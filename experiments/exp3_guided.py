"""Experiment 3: target-guided row choice.

The tables are the experiment-2 tables and stay frozen. What changes is how each input chunk picks
its row. Every variant scores the 16 rows of a chunk and takes the best one:

  nearest   score = -||x_c - p_k||^2, the nearest centroid (experiment 2's encoder).
  recon     score = -||x_c W_c - T_ck||^2, the row whose stored answer is closest to the chunk's exact
            partial product. Uses only x, costs the same 16 L multiplies (via G_ck = W_c T_ck).
  learned   base score (the better of nearest and recon) + a learned linear term in x_c. Trained end
            to end against the target. This is the control: training without target features.
  joint     like learned, but the centroids and the table entries are trained too (soft-to-hard
            assignment as in LUT-NN), so the 16-row split itself can move. Nearest-centroid base.
  joint_guided   joint + the target-feature term below.
  warmstart the fix for guided: train `learned` on the first half of the training tokens, then
            continue on the second half twice from that same state, once without the context term
            (control) and once with it (context path at --ctx-lr, x path at lr/10). The context term
            starts at zero, so the guided run starts exactly at the control's starting point.
            The fusion projection is a fixed random projection (unless --train-fusion), the context
            vector is normalized to unit RMS, and the adapter's learning rate is divided by its
            fan-in: with a trained fusion and a plain learning rate the context term grew about
            1000x faster than the learning rate suggests and the loss rose instead of falling.
  guided    learned + a term from the target's features: hidden states of 5 evenly spaced target
            layers, normalized and concatenated, are projected to a context vector (as in DFlash),
            a per-projection adapter maps it to 32 numbers, and a per-chunk matrix turns those into a
            score for each of the 16 rows.

Training: the hard row choice is used in the forward pass and a softmax over the scores carries the
gradient (straight-through). The loss is KL(p_target || q_draft) on natural text. The learned terms
start at zero, so training starts from the base encoder.

Staleness: a drafter only has the target's features up to the last verified token. In teacher-forced
evaluation position t sees the target feature from position t - s (s = 1, and s = 3 for the middle of
a 5-token round); training draws s from 1..5. In draft rounds the feature of the last verified prefix
position is held for all drafted tokens.

Usage (cloud GPU):
  python experiments/exp3_guided.py --configs lut:nearest:8:attn@2-15 lut:nearest:8:down@4-15
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
from exp1_single_layer import load_tokens  # noqa: E402
from exp2_acceptance import Swapper, collect_all, fit_layers, log, parse_config, set_mod  # noqa: E402


class State:
    """Shared between the drafter's guided layers: the context feature per position and the positions
    of the current forward pass."""
    f_pos: torch.Tensor | None = None  # [B, T, Df]
    pos: torch.Tensor | None = None  # [S]


class Fusion(nn.Module):
    def __init__(self, layer_ids, d_target, d_out):
        super().__init__()
        self.layer_ids = layer_ids
        self.proj = nn.Linear(len(layer_ids) * d_target, d_out, bias=False)
        self.norm = nn.RMSNorm(d_out)

    def forward(self, hidden_states):
        hs = [hidden_states[i].float() for i in self.layer_ids]
        hs = [h / h.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() for h in hs]  # tame massive activations
        return self.norm(self.proj(torch.cat(hs, -1)))


class GuidedLUT(nn.Module):
    def __init__(self, layer, lin, state: State, base: str, learn_x: bool, use_ctx: bool, d_feat: int,
                 ctx_dim: int = 32, ste: bool = True, train_tables: bool = False):
        super().__init__()
        enc = layer.enc
        C, K, m = layer.q.shape
        L = enc.L
        T = (layer.q.float() * layer.col_scale + layer.offset.unsqueeze(1))  # [C, K, m]
        T = T.reshape(C * K, m).contiguous()
        self.train_tables = train_tables
        if train_tables:
            assert base == "nearest"
            self.T = nn.Parameter(T)
            self.P = nn.Parameter(enc.protos.float().clone())  # trainable centroids [C, K, L]
        else:
            self.register_buffer("T", T)
        if base == "nearest":
            A, a0 = enc.protos.float(), enc.protos.float().pow(2).sum(-1)  # [C, K, L], [C, K]
        else:
            Wc = lin.weight.detach().float().T.reshape(C, L, m)  # [C, L, m]
            A, a0 = torch.einsum("clm,ckm->ckl", Wc, T), T.pow(2).sum(-1)
        self.register_buffer("A", A.contiguous())
        self.register_buffer("a0", a0.contiguous())
        self.register_buffer("tau", torch.tensor(0.0))
        self.C, self.K, self.L, self.bias, self.state, self.ste = C, K, L, layer.bias, state, ste
        self.lin = [lin]
        self.Wx = nn.Parameter(torch.zeros(C, L, K)) if learn_x else None
        self.bx = nn.Parameter(torch.zeros(C, K)) if learn_x else None
        self.use_ctx = use_ctx
        self.ctx_on = use_ctx
        self.ctx_ratio = None  # size of the context term relative to the base score, for logging
        if use_ctx:
            self.adapter = nn.Linear(d_feat, ctx_dim)
            self.U = nn.Parameter(torch.zeros(C, ctx_dim, K))

    def logits(self, x2):
        xs = x2.view(-1, self.C, self.L)
        if self.train_tables:
            s = 2 * torch.einsum("ncl,ckl->nck", xs, self.P) - self.P.pow(2).sum(-1)
        else:
            s = 2 * torch.einsum("ncl,ckl->nck", xs, self.A) - self.a0
        if self.tau.item() == 0.0:
            self.tau.fill_(s.std().item())
        s = s / self.tau
        if self.Wx is not None:
            s = s + torch.einsum("ncl,clk->nck", xs, self.Wx) + self.bx
        if self.use_ctx and self.ctx_on:
            f = self.state.f_pos[:, self.state.pos]  # [B, S, Df]
            # unit-RMS context vector: its size cannot grow with training, only its direction
            z = F.rms_norm(self.adapter(f.reshape(-1, f.shape[-1])), (self.U.shape[1],))
            ctx = torch.einsum("nz,czk->nck", z, self.U)
            self.ctx_ratio = (ctx.detach().pow(2).mean() / s.detach().pow(2).mean()).sqrt()
            s = s + ctx
        return s

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).float()
        s = self.logits(x2)
        hard = F.one_hot(s.argmax(-1), self.K).float()
        if torch.is_grad_enabled() and (self.Wx is not None or self.use_ctx):
            soft = F.softmax(s, -1)
            onehot = hard + soft - soft.detach()
        else:
            onehot = hard
        y = onehot.reshape(x2.shape[0], self.C * self.K) @ self.T
        if self.bias is not None:
            y = y + self.bias
        if self.ste and torch.is_grad_enabled():
            dense = F.linear(x2, self.lin[0].weight.float())
            y = y + (dense - dense.detach())
        return y.reshape(*shape[:-1], -1).to(x.dtype)


def shift(f, s):
    """Position t sees the feature of position t - s (zeros before that)."""
    return torch.cat([torch.zeros_like(f[:, :s]), f[:, :-s]], 1)


def target_pass(target, fusion, ids):
    out = target(ids, output_hidden_states=True)
    f = fusion(out.hidden_states) if fusion is not None else None
    return out.logits, f


@torch.no_grad()
def teacher_forced(target, drafter, fusion, state, tokens, batch, device, s):
    tot = dict(alpha=0.0, greedy=0.0, kl=0.0, n=0)
    for i in range(0, tokens.shape[0], batch):
        ids = tokens[i : i + batch].to(device)
        logits_t, f = target_pass(target, fusion, ids)
        state.f_pos = shift(f, s) if f is not None else None
        lp_t = F.log_softmax(logits_t[:, :-1].float(), -1)
        lp_d = F.log_softmax(drafter(ids).logits[:, :-1].float(), -1)
        p_t = lp_t.exp()
        tot["alpha"] += torch.minimum(p_t, lp_d.exp()).sum(-1).sum().item()
        tot["greedy"] += (lp_t.argmax(-1) == lp_d.argmax(-1)).sum().item()
        tot["kl"] += (p_t * (lp_t - lp_d)).sum(-1).sum().item()
        tot["n"] += lp_t.shape[0] * lp_t.shape[1]
    n = tot["n"]
    return dict(alpha_sample=tot["alpha"] / n, alpha_greedy=tot["greedy"] / n, kl=tot["kl"] / n)


@torch.no_grad()
def draft_rounds(target, drafter, fusion, state, tokens, gamma, n_rounds, batch, device, seed, pad_id):
    g = torch.Generator().manual_seed(seed)
    accepted = rejected = rounds = 0
    for r in range(0, n_rounds, batch):
        b = min(batch, n_rounds - r)
        seq_idx = torch.randint(0, tokens.shape[0], (b,), generator=g)
        plen = int(torch.randint(32, tokens.shape[1] - gamma, (1,), generator=g))
        prefix = tokens[seq_idx, :plen].to(device)
        if fusion is not None:
            _, f = target_pass(target, fusion, prefix)
            # prefix positions see the previous position's feature; drafted positions hold the feature
            # of position plen - 2, the last token the target has processed when the round starts
            held = f[:, plen - 2 : plen - 1].expand(-1, gamma + 1, -1)
            state.f_pos = torch.cat([shift(f, 1)[:, :plen], held], 1)
        out = drafter.generate(prefix, max_new_tokens=gamma, min_new_tokens=gamma, do_sample=False,
                               pad_token_id=pad_id, attention_mask=torch.ones_like(prefix))
        draft = out[:, plen : plen + gamma]
        t_tok = target(out[:, : plen + gamma]).logits[:, plen - 1 : plen + gamma - 1].argmax(-1)
        acc = (t_tok == draft).long().cumprod(-1).sum(-1)
        accepted += acc.sum().item()
        rejected += (acc < gamma).sum().item()
        rounds += b
    return dict(accepted_per_round=accepted / rounds, alpha_hat=accepted / max(1, accepted + rejected))


def train(target, drafter, fusion, state, mods, tokens, val, args, device, use_ctx=True, x_lr=None, ctx_lr=None):
    groups, params = [], []
    for m in mods:
        for name, p in m.named_parameters():
            is_ctx = name == "U" or name.startswith("adapter")
            if is_ctx and not use_ctx:
                continue
            if name.startswith("adapter") and ctx_lr is not None:
                # Adam moves every weight by about lr, so a unit summing fan_in inputs moves by about
                # lr * fan_in: scale the adapter's rate down to U's fan-in (32)
                groups.append({"params": [p], "lr": ctx_lr * m.U.shape[1] / m.adapter.in_features})
            elif is_ctx and ctx_lr is not None:
                groups.append({"params": [p], "lr": ctx_lr})
            elif name in ("Wx", "bx") and x_lr is not None:
                groups.append({"params": [p], "lr": x_lr})
            elif name == "T":
                groups.append({"params": [p], "lr": args.table_lr * p.detach().std().item()})
            elif name == "P":
                groups.append({"params": [p], "lr": args.table_lr * p.detach().std().item()})
            else:
                groups.append({"params": [p], "lr": args.lr})
            params.append(p)
    if fusion is not None and use_ctx and args.train_fusion:
        groups.append({"params": list(fusion.parameters()), "lr": ctx_lr if ctx_lr is not None else args.lr})
        params += list(fusion.parameters())
    if fusion is not None and not args.train_fusion:
        for p in fusion.parameters():
            p.requires_grad_(False)
    if args.adam8bit:
        import bitsandbytes as bnb  # 8-bit optimizer state: needed to train all 112 tables in 48 GB
        opt = bnb.optim.Adam8bit(groups, betas=(0.9, 0.99))
    else:
        opt = torch.optim.Adam(groups, betas=(0.9, 0.99), foreach=False)  # no full-size temporaries
    base_lrs = [g["lr"] for g in opt.param_groups]
    n_steps = math.ceil(tokens.shape[0] / args.train_batch)
    gen = torch.Generator().manual_seed(args.seed)
    t0, curve = time.time(), []
    for step in range(n_steps):
        ids = tokens[step * args.train_batch : (step + 1) * args.train_batch].to(device)
        for gr, blr in zip(opt.param_groups, base_lrs):
            gr["lr"] = blr * min(1.0, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * step / max(1, n_steps - 1)))
        s = int(torch.randint(1, 6, (1,), generator=gen))
        with torch.no_grad():
            out = target(ids, output_hidden_states=True)
            lp_t = F.log_softmax(out.logits[:, :-1].float(), -1)
        if fusion is not None and use_ctx:
            state.f_pos = shift(fusion(out.hidden_states), s)
        lp_d = F.log_softmax(drafter(ids).logits[:, :-1].float(), -1)
        loss = (lp_t.exp() * (lp_t - lp_d)).sum(-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % args.log_every == 0 or step == n_steps - 1:
            v = teacher_forced(target, drafter, fusion, state, val, 1, device, 1)["kl"]
            ratios = [m.ctx_ratio.item() for m in mods if getattr(m, "ctx_ratio", None) is not None and m.ctx_on]
            r = sum(ratios) / len(ratios) if ratios else 0.0
            curve.append((step, loss.item(), v, r))
            log(f"  step {step}/{n_steps} train KL {loss.item():.3f} val KL {v:.3f} context/base {r:.3f} "
                f"({time.time() - t0:.0f}s)")
    return curve, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--target", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--configs", nargs="+", default=["lut:nearest:8:attn@2-15", "lut:nearest:8:down@4-15"])
    ap.add_argument("--variants", nargs="+", default=["nearest", "recon", "learned", "guided"])
    ap.add_argument("--cal-tokens", type=int, default=200_000)
    ap.add_argument("--cal-rows", type=int, default=16_384)
    ap.add_argument("--train-tokens", type=int, default=1_000_000)
    ap.add_argument("--train-batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--adam8bit", action="store_true", help="8-bit Adam state (bitsandbytes)")
    ap.add_argument("--train-fusion", action="store_true", help="also train the fusion projection (default: fixed random)")
    ap.add_argument("--ctx-lr", type=float, default=1e-4, help="context path learning rate in the warmstart variant")
    ap.add_argument("--table-lr", type=float, default=0.02, help="tables and centroids, relative to their std")
    ap.add_argument("--d-feat", type=int, default=1024)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--eval-tokens", type=int, default=16_384)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=512)
    ap.add_argument("--round-batch", type=int, default=16)
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="results/exp3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.drafter)
    drafter = AutoModelForCausalLM.from_pretrained(args.drafter, dtype=dtype).to(device).eval()
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=dtype).to(device).eval()
    for p in list(drafter.parameters()) + list(target.parameters()):
        p.requires_grad_(False)
    pad_id = tok.eos_token_id
    n_t = target.config.num_hidden_layers
    # DFlash: 5 layers spaced uniformly from the second layer to the third-to-last.
    # hidden_states[i + 1] is the output of layer i.
    feat_layers = [round(1 + j * (n_t - 3 - 1) / 4) + 1 for j in range(5)]
    log(f"target feature layers (hidden_states index): {feat_layers}")

    state = State()

    def pre_hook(_mod, a, kw):
        pos = kw.get("position_ids")
        if pos is not None:
            state.pos = pos[0]
        else:
            ids = kw.get("input_ids", a[0] if a else None)
            S = ids.shape[1] if ids is not None else kw["inputs_embeds"].shape[1]
            pkv = kw.get("past_key_values")
            past = pkv.get_seq_length() if pkv is not None else 0
            state.pos = torch.arange(past, past + S, device=device)
    drafter.model.register_forward_pre_hook(pre_hook, with_kwargs=True)

    cal = load_tokens(tok, "train", args.cal_tokens, args.seq_len, args.seed)
    trn = load_tokens(tok, "train", args.train_tokens, args.seq_len, args.seed + 2)
    ev = load_tokens(tok, "test", args.eval_tokens, args.seq_len, args.seed + 1)
    val = load_tokens(tok, "validation", 4 * args.seq_len, args.seq_len, args.seed + 3)
    swapper = Swapper(drafter)
    cal_rows = collect_all(drafter, cal, args.batch, device, args.cal_rows, args.seed)
    del cal

    fields = ["config", "variant", "base", "staleness", "alpha_sample", "alpha_greedy", "kl", "accepted_per_round",
              "alpha_hat", "train_seconds", "n_params"]
    csv_path = out_dir / "results.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()
    meta = dict(vars(args), feat_layers=feat_layers, curves={})
    caches = {}

    for cfg_s in args.configs:
        cfg = parse_config(cfg_s, swapper.n_layers)
        cache = caches.setdefault((cfg["encoder"], cfg["L"]), {})
        log(f"== {cfg_s}")
        fit_layers(swapper, cal_rows, cfg["encoder"], cfg["L"], cfg["names"], cfg["layer_ids"], args.lam, device, cache)
        keys = [(l, n) for l in cfg["layer_ids"] for n in cfg["names"]]
        best = {}
        def evaluate(variant, base, fusion, use_ctx, secs, n_params):
            for s in ([1, 3] if use_ctx else [1]):
                r = teacher_forced(target, drafter, fusion, state, ev, args.batch, device, s)
                dr = draft_rounds(target, drafter, fusion, state, ev, args.gamma, args.rounds, args.round_batch,
                                  device, args.seed, pad_id) if s == 1 else dict(accepted_per_round="", alpha_hat="")
                row = dict(config=cfg_s, variant=variant, base=base, staleness=s if use_ctx else "", **r, **dr,
                           train_seconds=round(secs), n_params=n_params)
                with open(csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fields).writerow(row)
                log(f"  {variant:14s} s={s}: alpha={r['alpha_sample']:.3f} greedy={r['alpha_greedy']:.3f} "
                    f"KL={r['kl']:.3f} tok/round={dr['accepted_per_round']}")

        for variant in args.variants:
            if variant == "warmstart":
                fusion = Fusion(feat_layers, target.config.hidden_size, args.d_feat).to(device)
                mods = []
                for (l, n) in keys:
                    m = GuidedLUT(cache[(l, n)].to(device), swapper.originals[(l, n)], state, "nearest", True, True,
                                  args.d_feat).to(device)
                    m.ctx_on = False
                    set_mod(drafter, l, n, m)
                    mods.append(m)
                swapper.active = list(keys)
                n_params = sum(p.numel() for m in mods for p in m.parameters()) + sum(p.numel() for p in fusion.parameters())
                teacher_forced(target, drafter, None, state, val[:1], 1, device, 1)
                half = trn.shape[0] // 2
                log("  warmstart phase A: learned row choice, no context")
                curve, secs = train(target, drafter, fusion, state, mods, trn[:half], val, args, device, use_ctx=False)
                meta["curves"][f"{cfg_s}/ws_A"] = curve
                evaluate("ws_learned_A", "nearest", None, False, secs, n_params)
                snap = [(m.Wx.detach().clone(), m.bx.detach().clone()) for m in mods]
                log("  warmstart phase B control: continue without context")
                curve, secs = train(target, drafter, fusion, state, mods, trn[half:], val, args, device, use_ctx=False,
                                    x_lr=args.lr / 10)
                meta["curves"][f"{cfg_s}/ws_B_control"] = curve
                evaluate("ws_learned_AB", "nearest", None, False, secs, n_params)
                with torch.no_grad():
                    for m, (w, b) in zip(mods, snap):
                        m.Wx.copy_(w)
                        m.bx.copy_(b)
                        m.ctx_on = True
                log("  warmstart phase B guided: continue with context")
                curve, secs = train(target, drafter, fusion, state, mods, trn[half:], val, args, device, use_ctx=True,
                                    x_lr=args.lr / 10, ctx_lr=args.ctx_lr)
                meta["curves"][f"{cfg_s}/ws_B_guided"] = curve
                evaluate("ws_guided_AB", "nearest", fusion, True, secs, n_params)
                swapper.restore()
                del mods, fusion
                state.f_pos = None
                torch.cuda.empty_cache()
                (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
                continue
            train_tables = variant.startswith("joint")
            if variant in ("nearest", "recon"):
                base, learn_x, use_ctx = variant, False, False
            elif train_tables:
                base, learn_x, use_ctx = "nearest", True, variant == "joint_guided"
            else:
                base = max(best, key=best.get) if best else "nearest"
                learn_x, use_ctx = True, variant == "guided"
            fusion = Fusion(feat_layers, target.config.hidden_size, args.d_feat).to(device) if use_ctx else None
            mods = []
            for (l, n) in keys:
                m = GuidedLUT(cache[(l, n)].to(device), swapper.originals[(l, n)], state, base, learn_x, use_ctx,
                              args.d_feat, train_tables=train_tables).to(device)
                set_mod(drafter, l, n, m)
                mods.append(m)
            swapper.active = list(keys)
            n_params = sum(p.numel() for m in mods for p in m.parameters()) + (
                sum(p.numel() for p in fusion.parameters()) if fusion is not None else 0)
            secs = 0.0
            if learn_x or use_ctx or train_tables:
                # one no-grad pass sets each layer's score scale before training
                teacher_forced(target, drafter, fusion, state, val[:1], 1, device, 1)
                log(f"  {variant} (base {base}): training {n_params / 1e6:.1f}M parameters")
                curve, secs = train(target, drafter, fusion, state, mods, trn, val, args, device)
                meta["curves"][f"{cfg_s}/{variant}"] = curve
            for s in ([1, 3] if use_ctx else [1]):
                r = teacher_forced(target, drafter, fusion, state, ev, args.batch, device, s)
                dr = draft_rounds(target, drafter, fusion, state, ev, args.gamma, args.rounds, args.round_batch,
                                  device, args.seed, pad_id) if s == 1 else dict(accepted_per_round="", alpha_hat="")
                row = dict(config=cfg_s, variant=variant, base=base, staleness=s if use_ctx else "", **r, **dr,
                           train_seconds=round(secs), n_params=n_params)
                with open(csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fields).writerow(row)
                log(f"  {variant:8s} base={base:7s} s={s}: alpha={r['alpha_sample']:.3f} greedy={r['alpha_greedy']:.3f} "
                    f"KL={r['kl']:.3f} tok/round={dr['accepted_per_round']}")
                if variant in ("nearest", "recon"):
                    best[variant] = r["alpha_sample"]
            swapper.restore()
            del mods, fusion
            state.f_pos = None
            torch.cuda.empty_cache()
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log("done")


if __name__ == "__main__":
    main()
