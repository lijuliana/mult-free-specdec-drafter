"""Experiment 1: single-layer reconstruction.

Replace one linear projection of the drafter with a lookup table (or a weight-quantized copy) and
measure (a) the layer's relative output error on held-out activations and (b) the effect on the
whole model's next-token distribution: KL(p_exact || p_replaced) and top-1 agreement, which is the
greedy acceptance proxy.

Usage (defaults reproduce the proposal's design):
  python experiments/exp1_single_layer.py --model meta-llama/Llama-3.2-1B --layer 8 \
      --targets mlp.down_proj self_attn.o_proj --Ls 8 16 32 --encoders nearest tree hash
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
import lut  # noqa: E402


def start_memory_guard(limit_gb: float):
    """Abort the whole process if resident memory passes limit_gb. Checked every 0.5 s."""
    limit = limit_gb * 1024**3

    def watch():
        while True:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
            if sys.platform != "darwin":
                rss *= 1024
            if rss > limit:
                print(f"\nMEMORY GUARD: resident memory {rss/1024**3:.1f} GB exceeded {limit_gb} GB, aborting", flush=True)
                os._exit(3)
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokens(tokenizer, split: str, n_tokens: int, seq_len: int, seed: int) -> torch.Tensor:
    """WikiText-103 raw, concatenated with EOS between documents, cut into [n_seq, seq_len]."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=20_000)
    ids: list[int] = []
    for row in ds:
        t = row["text"]
        if not t.strip():
            continue
        ids.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
        ids.append(tokenizer.eos_token_id)
        if len(ids) >= n_tokens:
            break
    n_seq = len(ids) // seq_len
    return torch.tensor(ids[: n_seq * seq_len]).view(n_seq, seq_len)


def get_module(model: nn.Module, layer: int, target: str) -> nn.Linear:
    mod = model.model.layers[layer]
    for part in target.split("."):
        mod = getattr(mod, part)
    assert isinstance(mod, nn.Linear)
    return mod


@torch.no_grad()
def collect_inputs(model, module: nn.Linear, tokens: torch.Tensor, batch: int, device, max_rows: int, seed: int):
    """Run the model and record the module's inputs; subsample to max_rows rows (float32, on CPU)."""
    rows = []
    captured = {}

    def hook(_m, inp, _out):
        captured["x"] = inp[0].detach().reshape(-1, inp[0].shape[-1]).float().cpu()

    h = module.register_forward_hook(hook)
    g = torch.Generator().manual_seed(seed)
    per_batch = max(1, max_rows // max(1, (tokens.shape[0] + batch - 1) // batch))
    for s in range(0, tokens.shape[0], batch):
        model(tokens[s : s + batch].to(device))
        x = captured["x"]
        idx = torch.randperm(x.shape[0], generator=g)[:per_batch]
        rows.append(x[idx])
    h.remove()
    return torch.cat(rows)[:max_rows]


class Replaced(nn.Module):
    """Wraps a LUT layer or a quantized weight so it can stand in for an nn.Linear."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def swap(model, layer, target, new: nn.Module | None, original: nn.Linear):
    parent = model.model.layers[layer]
    parts = target.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new if new is not None else original)


def _logp(model, ids):
    logits = model(ids).logits[:, :-1].float()
    logp = F.log_softmax(logits, -1)
    del logits
    return logp


@torch.no_grad()
def eval_model(model, tokens, batch, device, set_replaced=None):
    """Returns (mean KL(p_exact || p_replaced), top-1 agreement, mean NLL of the replaced model).

    The exact and replaced models are run in lockstep per batch, so no distribution is cached:
    at 128K vocabulary a cache of 32K positions would take 8 GB. set_replaced(True/False) swaps
    the module in and out; if it is None only the exact model's NLL is measured."""
    kls, agree, nll, n = 0.0, 0.0, 0.0, 0
    for s in range(0, tokens.shape[0], batch):
        ids = tokens[s : s + batch].to(device)
        tgt = ids[:, 1:]
        if set_replaced is None:
            logp = _logp(model, ids)
            nll += F.nll_loss(logp.reshape(-1, logp.shape[-1]), tgt.reshape(-1), reduction="sum").item()
            n += tgt.numel()
            del logp
            continue
        set_replaced(False)
        ref = _logp(model, ids)
        set_replaced(True)
        logp = _logp(model, ids)
        set_replaced(False)
        nll += F.nll_loss(logp.reshape(-1, logp.shape[-1]), tgt.reshape(-1), reduction="sum").item()
        kls += (ref.exp() * (ref - logp)).sum(-1).sum().item()
        agree += (ref.argmax(-1) == logp.argmax(-1)).sum().item()
        n += tgt.numel()
        del ref, logp
    return kls / n, agree / n, nll / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--targets", nargs="+", default=["mlp.down_proj", "self_attn.o_proj"])
    ap.add_argument("--Ls", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--encoders", nargs="+", default=["nearest", "tree", "hash"])
    ap.add_argument("--smooth", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--no-ridge-ablation", action="store_true", help="also run nearest encoder with plain prototype tables")
    ap.add_argument("--cal-tokens", type=int, default=1_000_000)
    ap.add_argument("--cal-rows", type=int, default=100_000)
    ap.add_argument("--eval-tokens", type=int, default=32_768)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="cpu (default; safe on a laptop), mps or cuda")
    ap.add_argument("--dtype", default="float32", help="float32 on CPU; bfloat16 on a GPU")
    ap.add_argument("--out", default="results/exp1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=3.0, help="ridge penalty toward prototype tables, relative to rows per entry")
    ap.add_argument("--max-rss-gb", type=float, default=6.0, help="abort if resident memory exceeds this")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    start_memory_guard(args.max_rss_gb)
    torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} dtype={dtype}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()

    t0 = time.time()
    cal = load_tokens(tok, "train", args.cal_tokens, args.seq_len, args.seed)
    ev = load_tokens(tok, "test", args.eval_tokens, args.seq_len, args.seed + 1)
    print(f"calibration {cal.numel()} tokens, eval {ev.numel()} tokens ({time.time()-t0:.0f}s)", flush=True)

    # Exact model perplexity.
    _, _, nll_exact = eval_model(model, ev, args.batch, device, None)
    print(f"exact nll {nll_exact:.4f} (ppl {torch.tensor(nll_exact).exp():.2f})", flush=True)

    rows = []
    csv_path = out_dir / "results.csv"
    fields = ["target", "method", "encoder", "L", "smooth", "ridge", "d", "m", "rel_err", "kl", "top1", "nll",
              "bytes_per_token", "storage_bytes", "fit_seconds"]

    for target in args.targets:
        module = get_module(model, args.layer, target)
        W = module.weight.detach().float().T.contiguous().cpu()  # [d, m]
        bias = module.bias.detach().float().cpu() if module.bias is not None else None
        d, m = W.shape
        print(f"\n== layer {args.layer} {target}: d={d} m={m}", flush=True)

        x_cal = collect_inputs(model, module, cal, args.batch, device, args.cal_rows, args.seed)
        x_ev = collect_inputs(model, module, ev, args.batch, device, min(args.cal_rows, 20_000), args.seed + 1)
        y_ev = x_ev @ W
        print(f"collected {x_cal.shape[0]} calibration rows, {x_ev.shape[0]} eval rows", flush=True)

        def record(method, encoder, L, smooth, ridge, rel_err, fit_s, new_module):
            def set_replaced(on: bool):
                swap(model, args.layer, target, new_module if on else None, module)

            kl, top1, nll = eval_model(model, ev, args.batch, device, set_replaced)
            bpt = lut.bytes_per_token(d, m, "lut" if method == "lut" else method, L)
            sb = lut.storage_bytes(d, m, "lut" if method == "lut" else method, L)
            row = dict(target=target, method=method, encoder=encoder, L=L, smooth=smooth, ridge=ridge, d=d, m=m,
                       rel_err=rel_err, kl=kl, top1=top1, nll=nll, bytes_per_token=bpt, storage_bytes=sb,
                       fit_seconds=fit_s)
            rows.append(row)
            print(f"{method:5s} enc={encoder or '-':8s} L={L or '-':>3} smooth={smooth} ridge={ridge} "
                  f"rel_err={rel_err:.4f} KL={kl:.5f} top1={top1:.4f} nll={nll:.4f} fit={fit_s:.0f}s", flush=True)
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)

        # Weight-quantization baselines.
        for method, bits, group in (("int8", 8, None), ("int4", 4, 128)):
            Wq = lut.rtn_quantize_weight(W, bits, group)
            rel = ((x_ev @ Wq - y_ev).norm() / y_ev.norm()).item()
            lin = nn.Linear(d, m, bias=bias is not None).to(device=device, dtype=dtype)
            lin.weight.data.copy_(Wq.T.to(dtype))
            if bias is not None:
                lin.bias.data.copy_(bias.to(dtype))
            record(method, None, None, 0, 0, rel, 0.0, lin)

        # Lookup tables.
        configs = [(enc, L, sm, 1) for L in args.Ls for enc in args.encoders for sm in args.smooth]
        if args.no_ridge_ablation:
            configs += [("nearest", L, sm, 0) for L in args.Ls for sm in args.smooth]
        for enc, L, sm, ridge in configs:
            t0 = time.time()
            layer = lut.build_lut_layer(enc, L, x_cal.to(device), W.to(device), bias, bool(sm), bool(ridge), args.lam).to(device)
            fit_s = time.time() - t0
            yhat = layer.forward(x_ev.to(device).to(dtype)).float().cpu()
            rel = ((yhat - y_ev).norm() / y_ev.norm()).item()
            record("lut", enc, L, sm, ridge, rel, fit_s, Replaced(layer.forward))

    with open(out_dir / "meta.json", "w") as f:
        json.dump(dict(vars(args), nll_exact=nll_exact, cal_tokens=int(cal.numel()), eval_tokens=int(ev.numel())), f, indent=2)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
