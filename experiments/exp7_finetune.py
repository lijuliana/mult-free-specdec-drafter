"""Experiment 7: fine-tune the lookup tables against the target's next-token distribution.

Start from the experiment-2 tables (k-means centroids, ridge refit, 8-bit entries), turn every table
entry into a trainable float parameter, and train all of them jointly so the drafter's next-token
distribution matches the target's (KL(p_target || q_draft) on natural text). The encoders stay fixed,
so each input chunk still picks its centroid the same way; only the stored answers change.

A table layer has zero gradient with respect to its input (the centroid choice is piecewise
constant), so gradients would reach earlier layers only through the residual stream. With
straight-through estimation (default) the backward pass uses the original weight matrix as the
layer's input gradient, which lets early tables learn to correct what later tables need.

After training the tables are rounded back to 8 bits and evaluated with the same teacher-forced
acceptance and draft-round measurements as experiment 2, before and after training.

Usage (cloud GPU):
  python experiments/exp7_finetune.py --configs lut:nearest:8:attn@2-15 lut:nearest:8:down@4-15
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
import lut  # noqa: E402
from exp1_single_layer import Replaced, load_tokens  # noqa: E402
from exp2_acceptance import (Swapper, collect_all, draft_rounds, fit_layers, log, parse_config,  # noqa: E402
                             set_mod, teacher_forced)


class TrainLUT(nn.Module):
    """A lookup-table projection with trainable float entries and a fixed encoder."""

    def __init__(self, layer: lut.LUTLayer, lin: nn.Linear, ste: bool):
        super().__init__()
        assert layer.scale_in is None
        C, K, m = layer.q.shape
        T = layer.q.float() * layer.col_scale + layer.offset.unsqueeze(1)  # [C, K, m], what the 8-bit table computes
        self.T = nn.Parameter(T.reshape(C * K, m).contiguous())
        self.enc, self.C, self.K, self.bias, self.ste = layer.enc, C, K, layer.bias, ste
        self.lin = [lin]  # in a list so it is not registered as a submodule

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        with torch.no_grad():
            codes = self.enc.encode(x2.float())
            flat = codes + torch.arange(self.C, device=x.device) * self.K
            onehot = torch.zeros(x2.shape[0], self.C * self.K, device=x.device, dtype=torch.bfloat16)
            onehot.scatter_(1, flat, 1.0)
        y = onehot @ self.T.to(torch.bfloat16)
        if self.bias is not None:
            y = y + self.bias
        if self.ste:
            dense = F.linear(x2, self.lin[0].weight)
            y = y + (dense - dense.detach())
        return y.reshape(*shape[:-1], -1).to(x.dtype)

    @torch.no_grad()
    def to_lut(self) -> lut.LUTLayer:
        q, col_scale, offset = lut.quantize_tables(self.T.detach().float(), self.C)
        return lut.LUTLayer(self.enc, None, q, col_scale, offset, self.bias)


def kl_loss(target, drafter, ids):
    with torch.no_grad():
        lp_t = F.log_softmax(target(ids).logits[:, :-1].float(), -1)
    lp_d = F.log_softmax(drafter(ids).logits[:, :-1].float(), -1)
    return (lp_t.exp() * (lp_t - lp_d)).sum(-1).mean()


def train(target, drafter, mods, train_tokens, val_tokens, args, device):
    params = [m.T for m in mods]
    if args.optim == "adam":
        # Adam's step is about lr in absolute terms, so scale lr by each table's typical entry size
        groups = [{"params": [p], "lr": args.lr * p.detach().float().std().item()} for p in params]
        opt = torch.optim.Adam(groups, betas=(0.9, 0.99))
    else:
        opt = torch.optim.Adafactor(params, lr=args.lr)  # relative step size, small optimizer state
    base_lrs = [g["lr"] for g in opt.param_groups]
    n_steps = math.ceil(train_tokens.shape[0] / args.train_batch)
    t0, curve = time.time(), []
    for step in range(n_steps):
        ids = train_tokens[step * args.train_batch : (step + 1) * args.train_batch].to(device)
        frac = step / max(1, n_steps - 1)
        warm = min(1.0, (step + 1) / 50)
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * warm * 0.5 * (1 + math.cos(math.pi * frac))
        loss = kl_loss(target, drafter, ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % args.log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                val = sum(kl_loss(target, drafter, val_tokens[i : i + 1].to(device)).item()
                          for i in range(val_tokens.shape[0])) / val_tokens.shape[0]
            curve.append((step, loss.item(), val))
            log(f"  step {step}/{n_steps} train KL {loss.item():.3f} val KL {val:.3f} ({time.time() - t0:.0f}s)")
    return curve, time.time() - t0


def evaluate(target, drafter, ev, args, device, pad_id):
    tf = teacher_forced(target, drafter, ev, args.batch, device)
    dr = draft_rounds(target, drafter, ev, args.gamma, args.rounds, args.round_batch, device, args.seed, pad_id)
    return {**tf, "accepted_per_round": dr["accepted_per_round"], "alpha_hat": dr["alpha_hat"],
            "hist": " ".join(map(str, dr["hist"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--target", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--configs", nargs="+", default=["lut:nearest:8:attn@2-15", "lut:nearest:8:down@4-15"])
    ap.add_argument("--cal-tokens", type=int, default=200_000)
    ap.add_argument("--cal-rows", type=int, default=16_384)
    ap.add_argument("--train-tokens", type=int, default=2_000_000)
    ap.add_argument("--train-batch", type=int, default=2, help="sequences per step")
    ap.add_argument("--optim", choices=["adam", "adafactor"], default="adam")
    ap.add_argument("--lr", type=float, default=0.02, help="Adam: relative to each table's entry std")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--no-ste", action="store_true")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--eval-tokens", type=int, default=16_384)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=512)
    ap.add_argument("--round-batch", type=int, default=16)
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="results/exp7")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.drafter)
    drafter = AutoModelForCausalLM.from_pretrained(args.drafter, dtype=dtype).to(device).eval()
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=dtype).to(device).eval()
    for p in list(drafter.parameters()) + list(target.parameters()):
        p.requires_grad_(False)
    pad_id = tok.eos_token_id

    cal = load_tokens(tok, "train", args.cal_tokens, args.seq_len, args.seed)
    trn = load_tokens(tok, "train", args.train_tokens, args.seq_len, args.seed + 2)
    ev = load_tokens(tok, "test", args.eval_tokens, args.seq_len, args.seed + 1)
    val = load_tokens(tok, "validation", 4 * args.seq_len, args.seq_len, args.seed + 3)
    log(f"calibration {cal.numel()}, training {trn.numel()}, eval {ev.numel()}, validation {val.numel()} tokens")

    swapper = Swapper(drafter)
    cal_rows = collect_all(drafter, cal, args.batch, device, args.cal_rows, args.seed)
    del cal

    fields = ["config", "stage", "alpha_sample", "alpha_greedy", "kl", "nll", "accepted_per_round", "alpha_hat",
              "hist", "train_tokens", "train_seconds", "ste", "optim", "lr"]
    csv_path = out / "results.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    def write(row):
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)

    caches = {}
    meta = dict(vars(args), curves={})
    for cfg_s in args.configs:
        cfg = parse_config(cfg_s, swapper.n_layers)
        assert cfg["method"] == "lut"
        cache = caches.setdefault((cfg["encoder"], cfg["L"]), {})
        log(f"== {cfg_s}")
        fit_layers(swapper, cal_rows, cfg["encoder"], cfg["L"], cfg["names"], cfg["layer_ids"], args.lam, device, cache)
        common = dict(config=cfg_s, train_tokens=0, train_seconds=0, ste=not args.no_ste, optim=args.optim, lr=args.lr)

        # before: the experiment-2 drafter
        swapper.apply_lut(cache, cfg["names"], cfg["layer_ids"], device)
        r = evaluate(target, drafter, ev, args, device, pad_id)
        write({**common, "stage": "before", **r})
        log(f"  before: alpha={r['alpha_sample']:.3f} greedy={r['alpha_greedy']:.3f} KL={r['kl']:.3f} "
            f"tok/round={r['accepted_per_round']:.2f}")
        swapper.restore()

        # train
        mods = []
        for l in cfg["layer_ids"]:
            for n in cfg["names"]:
                m = TrainLUT(cache[(l, n)].to(device), swapper.originals[(l, n)], ste=not args.no_ste)
                set_mod(drafter, l, n, m)
                swapper.active.append((l, n))
                mods.append(m)
        n_params = sum(m.T.numel() for m in mods)
        log(f"  training {len(mods)} tables, {n_params / 1e6:.0f}M entries")
        curve, secs = train(target, drafter, mods, trn, val, args, device)
        meta["curves"][cfg_s] = curve

        # after: round the trained tables back to 8 bits and evaluate
        for m, (l, n) in zip(mods, [(l, n) for l in cfg["layer_ids"] for n in cfg["names"]]):
            set_mod(drafter, l, n, Replaced(m.to_lut().forward))
        del mods
        torch.cuda.empty_cache()
        r = evaluate(target, drafter, ev, args, device, pad_id)
        write({**common, "stage": "after", "train_tokens": trn.numel(), "train_seconds": round(secs), **r})
        log(f"  after:  alpha={r['alpha_sample']:.3f} greedy={r['alpha_greedy']:.3f} KL={r['kl']:.3f} "
            f"tok/round={r['accepted_per_round']:.2f} (trained {secs:.0f}s)")
        swapper.restore()
        torch.cuda.empty_cache()
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log("done")


if __name__ == "__main__":
    main()
