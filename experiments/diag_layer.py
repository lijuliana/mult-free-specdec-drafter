"""Layer-level diagnostic: where does the lookup-table error come from?

Collects inputs to one projection, then measures relative output error for variants that isolate
each error source: subvector length, number of prototypes, 8-bit table quantization, ridge vs plain
prototype tables, and an "oracle" that uses the true nearest prototype with unquantized ridge tables.
No model forward passes after collection, so it is cheap.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import lut  # noqa: E402
from exp1_single_layer import collect_inputs, get_module, load_tokens, start_memory_guard  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def rel(yhat, y):
    return ((yhat - y).norm() / y.norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--target", default="mlp.down_proj")
    ap.add_argument("--cal-tokens", type=int, default=50_000)
    ap.add_argument("--cal-rows", type=int, default=10_000)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    start_memory_guard(6.0)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    cal = load_tokens(tok, "train", args.cal_tokens, 512, 0)
    ev = load_tokens(tok, "test", 8192, 512, 1)
    module = get_module(model, args.layer, args.target)
    W = module.weight.detach().float().T.contiguous().cpu()
    d, m = W.shape
    x_cal = collect_inputs(model, module, cal, 1, device, args.cal_rows, 0)
    x_ev = collect_inputs(model, module, ev, 1, device, 4000, 1)
    del model
    y_ev = x_ev @ W
    print(f"{args.target}: d={d} m={m}; cal rows {x_cal.shape[0]}, eval rows {x_ev.shape[0]}")

    # activation statistics: how heavy-tailed and how concentrated in a few channels
    ch_max = x_cal.abs().amax(0)
    print(f"activation |x| per-channel max: median {ch_max.median():.2f}, max {ch_max.max():.2f}, "
          f"top-8 channels hold {(ch_max.topk(8).values.pow(2).sum() / ch_max.pow(2).sum()).item():.1%} of max-energy")
    print(f"rows: mean |x| {x_cal.abs().mean():.3f}, 99.9th pct {x_cal.abs().flatten().kthvalue(int(0.999 * x_cal.numel())).values:.2f}")

    for bits, g in ((8, None), (4, 128), (3, 128), (2, 128)):
        print(f"int{bits} g={g}: rel_err {rel(x_ev @ lut.rtn_quantize_weight(W, bits, g), y_ev):.4f}")

    def run(kind, L, smooth, ridge, quant, K=16, lam=1e-3):
        lut.K = K
        t0 = time.time()
        xc, Wc = x_cal, W
        scale = None
        if smooth:
            scale = lut.smooth_scales(xc, Wc)
            xc, Wc = xc / scale, Wc * scale.unsqueeze(1)
        enc = lut.fit_encoder(kind, xc, L)
        codes = enc.encode(xc)
        T0 = lut.proto_tables(lut.bucket_means(codes, xc, enc.C, L), Wc)
        T = lut.ridge_tables(codes, xc @ Wc, enc.C, lam, T0) if ridge else T0
        xe = x_ev / scale if smooth else x_ev
        ce = enc.encode(xe)
        C = enc.C
        flat = ce + torch.arange(C).unsqueeze(0) * K
        if quant:
            q, cs, off = lut.quantize_tables(T, C)
            tab = q.reshape(C * K, m).float()
            yhat = torch.nn.functional.embedding_bag(flat, tab, mode="sum") * cs + off.sum(0)
        else:
            yhat = torch.nn.functional.embedding_bag(flat, T, mode="sum")
        used = (torch.stack([torch.bincount(ce[:, c], minlength=K) for c in range(C)]) > 0).float().mean().item()
        print(f"{kind:8s} L={L:2d} K={K:3d} smooth={int(smooth)} ridge={int(ridge)} 8bit={int(quant)} lam={lam:g}: "
              f"rel_err {rel(yhat, y_ev):.4f}  (codes used {used:.0%}, {time.time()-t0:.0f}s)", flush=True)
        lut.K = 16

    print("\n-- plain prototype tables (bucket means dot W)")
    run("nearest", 8, False, False, True)
    print("-- ridge toward the prototype tables, penalty relative to rows per entry")
    for lam in (0.03, 0.3, 3.0, 30.0):
        run("nearest", 8, False, True, True, lam=lam)
    print("-- same for the cheap encoders")
    for kind in ("tree", "hash"):
        run(kind, 8, False, False, True)
        run(kind, 8, False, True, True, lam=3.0)
    print("-- shorter subvectors (2x bytes per token, 2x storage)")
    run("nearest", 4, False, False, True)
    run("nearest", 4, False, True, True, lam=3.0)
    print("-- longer subvectors (half the bytes)")
    run("nearest", 16, False, False, True)
    run("nearest", 16, False, True, True, lam=3.0)
    print("-- 256 prototypes per subspace, prototype tables only (16x storage; resolution check)")
    run("nearest", 8, False, False, True, K=256)


if __name__ == "__main__":
    main()
