"""Time one decode step of a whole drafter (batch 1) with 16-bit, 4-bit or lookup-table projections,
and turn the timings into speculative-decoding speedups.

The decode step is a Llama block stack written directly in PyTorch: RMSNorm, q/k/v projections,
attention over a KV cache of --ctx tokens, output projection, RMSNorm, gate/up, SiLU, down, and the
output head with argmax. The whole step is captured in one CUDA graph, so launch overhead is what a
production decoder would pay. Weights, tables and caches are random; timing depends only on shapes.

Projection back ends:
  bf16  cuBLAS matrix-vector product.
  int4  torch._weight_int4pack_mm (the tinygemm kernel used by gpt-fast), groups of 128.
  lut   two Triton kernels: nearest-centroid encoding of each 8-number chunk (16 centroids), then a
        fused gather-and-add of one 8-bit table row per chunk with per-column scale and offset.

Speedup per round with gamma drafted tokens, acceptance a and relative draft cost c:
  (1 - a^(gamma+1)) / ((1 - a) (gamma c + 1)).

Usage: python experiments/bench_drafter.py
"""

import json
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

dev = "cuda"
K, L = 16, 8
torch.backends.cuda.matmul.allow_tf32 = False

SHAPES = {  # hidden, intermediate, layers, heads, kv heads, head dim
    "llama-3.2-1b": (2048, 8192, 16, 32, 8, 64),
    "llama-3.2-3b": (3072, 8192, 28, 24, 8, 128),
    "llama-3.1-8b": (4096, 14336, 32, 32, 8, 128),
}
VOCAB = 128256


@triton.jit
def nearest_kernel(X, P, P2, codes, L: tl.constexpr, K: tl.constexpr, CH: tl.constexpr):
    pc = tl.program_id(0)
    cidx = pc * CH + tl.arange(0, CH)
    l = tl.arange(0, L)
    k = tl.arange(0, K)
    xs = tl.load(X + cidx[:, None] * L + l[None, :]).to(tl.float32)  # [CH, L]
    p = tl.load(P + (cidx[:, None, None] * K + k[None, :, None]) * L + l[None, None, :]).to(tl.float32)
    score = 2 * tl.sum(xs[:, None, :] * p, axis=2) - tl.load(P2 + cidx[:, None] * K + k[None, :])
    tl.store(codes + cidx, tl.argmax(score, axis=1).to(tl.int32))


@triton.jit
def gather_kernel(T, codes, scale, out, m, K: tl.constexpr, CH: tl.constexpr, BN: tl.constexpr):
    pn = tl.program_id(0)
    pc = tl.program_id(1)
    cols = pn * BN + tl.arange(0, BN)
    cidx = pc * CH + tl.arange(0, CH)
    rows = cidx.to(tl.int64) * K + tl.load(codes + cidx).to(tl.int64)
    v = tl.load(T + rows[:, None] * m + cols[None, :])
    part = tl.sum(v.to(tl.int32), axis=0).to(tl.float32) * tl.load(scale + cols)
    tl.atomic_add(out + cols, part)


class Proj:
    def __init__(self, d, m, kind):
        self.d, self.m, self.kind = d, m, kind
        if kind == "bf16":
            self.W = torch.randn(m, d, device=dev, dtype=torch.bfloat16) * 0.02
        elif kind == "int4":
            w = torch.randint(0, 256, (m, d // 2), device=dev, dtype=torch.uint8)
            self.W = torch._convert_weight_to_int4pack(w, 8)
            self.sz = torch.randn(d // 128, m, 2, device=dev, dtype=torch.bfloat16) * 0.01
        elif kind == "lut":
            C = d // L
            self.C = C
            self.T = torch.randint(-128, 127, (C * K, m), device=dev, dtype=torch.int8)
            self.P = torch.randn(C, K, L, device=dev, dtype=torch.float16)
            self.P2 = self.P.float().pow(2).sum(-1).contiguous()
            self.scale = torch.rand(m, device=dev) * 1e-3
            self.offset = torch.randn(m, device=dev) * 1e-2
            self.codes = torch.empty(C, device=dev, dtype=torch.int32)
            self.acc = torch.empty(m, device=dev)
            self.CH = 128 if C % 128 == 0 else 64
            self.BN = 64

    def __call__(self, x):  # x [1, d] bf16 -> [1, m] bf16
        if self.kind == "bf16":
            return F.linear(x, self.W)
        if self.kind == "int4":
            return torch._weight_int4pack_mm(x, self.W, 128, self.sz)
        nearest_kernel[(self.C // 32,)](x, self.P, self.P2, self.codes, L=L, K=K, CH=32)
        self.acc.copy_(self.offset)
        gather_kernel[(self.m // self.BN, self.C // self.CH)](self.T, self.codes, self.scale, self.acc, self.m,
                                                             K=K, CH=self.CH, BN=self.BN)
        return self.acc.to(torch.bfloat16).view(1, -1)


class Model:
    """kinds: function (layer, projection name) -> back end; head: back end for the output head."""

    def __init__(self, name, kinds, head="bf16", ctx=512):
        d, im, nl, nh, nkv, hd = SHAPES[name]
        self.d, self.nh, self.nkv, self.hd = d, nh, nkv, hd
        self.layers = []
        for l in range(nl):
            k = lambda n: kinds(l, n)  # noqa: E731
            self.layers.append(dict(
                q=Proj(d, nh * hd, k("q")), k=Proj(d, nkv * hd, k("k")), v=Proj(d, nkv * hd, k("v")),
                o=Proj(nh * hd, d, k("o")), gate=Proj(d, im, k("gate")), up=Proj(d, im, k("up")),
                down=Proj(im, d, k("down")),
                n1=torch.ones(d, device=dev, dtype=torch.bfloat16), n2=torch.ones(d, device=dev, dtype=torch.bfloat16),
                kc=torch.randn(1, nkv, ctx, hd, device=dev, dtype=torch.bfloat16),
                vc=torch.randn(1, nkv, ctx, hd, device=dev, dtype=torch.bfloat16)))
        self.norm = torch.ones(d, device=dev, dtype=torch.bfloat16)
        self.head = Proj(d, VOCAB, head)
        self.x = torch.randn(1, d, device=dev, dtype=torch.bfloat16)

    def step(self):
        x = self.x
        for ly in self.layers:
            h = F.rms_norm(x, (self.d,), ly["n1"])
            q = ly["q"](h).view(1, self.nh, 1, self.hd)
            ly["k"](h)
            ly["v"](h)  # new key and value; the cache write is omitted
            a = F.scaled_dot_product_attention(q, ly["kc"], ly["vc"], enable_gqa=True)
            x = x + ly["o"](a.reshape(1, -1))
            h = F.rms_norm(x, (self.d,), ly["n2"])
            x = x + ly["down"](F.silu(ly["gate"](h)) * ly["up"](h))
        return self.head(F.rms_norm(x, (self.d,), self.norm)).argmax(-1)


def time_step(model, reps=100):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            model.step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        model.step()
    for _ in range(10):
        g.replay()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps  # ms


def speedup(a, c, gamma=5):
    return (1 - a ** (gamma + 1)) / ((1 - a) * (gamma * c + 1))


def break_even(c, goal, gamma=5):
    lo, hi = 0.0, 0.999
    if speedup(hi, c, gamma) < goal:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if speedup(mid, c, gamma) < goal else (lo, mid)
    return hi


def main():
    print(torch.cuda.get_device_name(), "torch", torch.__version__, "triton", triton.__version__, flush=True)
    attn = {"q", "k", "v", "o"}
    drafters = {
        "16-bit": (lambda l, n: "bf16", "bf16"),
        "4-bit, all projections": (lambda l, n: "int4", "bf16"),
        "tables, all projections": (lambda l, n: "lut", "bf16"),
        "tables on attention 2-15, 16-bit rest": (lambda l, n: "lut" if (n in attn and l >= 2) else "bf16", "bf16"),
        "tables on attention 2-15, 4-bit rest": (lambda l, n: "lut" if (n in attn and l >= 2) else "int4", "bf16"),
        "tables on down 4-15 and attention 2-15, 4-bit rest":
            (lambda l, n: "lut" if ((n in attn and l >= 2) or (n == "down" and l >= 4)) else "int4", "bf16"),
        "tables on MLP, 4-bit attention": (lambda l, n: "int4" if n in attn else "lut", "bf16"),
        "tables on MLP 4-15, 4-bit rest": (lambda l, n: "lut" if (n not in attn and l >= 4) else "int4", "bf16"),
        "tables on attention, 4-bit MLP": (lambda l, n: "lut" if n in attn else "int4", "bf16"),
        "tables in layers 2-15, 4-bit layers 0-1": (lambda l, n: "lut" if l >= 2 else "int4", "bf16"),
        "4-bit, all projections, 4-bit head": (lambda l, n: "int4", "int4"),
        "tables in layers 2-15, 4-bit layers 0-1, 4-bit head": (lambda l, n: "lut" if l >= 2 else "int4", "int4"),
        "tables on MLP, 4-bit attention, 4-bit head": (lambda l, n: "int4" if n in attn else "lut", "int4"),
        "tables on attention 2-15, 4-bit rest, 4-bit head": (lambda l, n: "lut" if (n in attn and l >= 2) else "int4", "int4"),
        "tables, all projections, 4-bit head": (lambda l, n: "lut", "int4"),
    }
    res = {}
    for tname in ("llama-3.2-3b", "llama-3.1-8b"):
        m = Model(tname, lambda l, n: "bf16")
        res[tname] = time_step(m)
        print(f"target {tname}, 16-bit: {res[tname]:.3f} ms/token", flush=True)
        del m
        torch.cuda.empty_cache()
    for name, (kinds, head) in drafters.items():
        m = Model("llama-3.2-1b", kinds, head)
        res[name] = time_step(m)
        print(f"drafter {name}: {res[name]:.3f} ms/token", flush=True)
        del m
        torch.cuda.empty_cache()
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "bench_drafter.json", "w"), indent=2)

    # speedup and break-even acceptance against the 3B target, gamma = 5
    measured = {"16-bit": 0.74, "4-bit, all projections": 0.66}
    goal = speedup(0.66, res["4-bit, all projections"] / res["llama-3.2-3b"])
    print(f"\nagainst Llama-3.2-3B, gamma = 5. 4-bit drafter at acceptance 0.66: {goal:.2f}x")
    for name in drafters:
        c = res[name] / res["llama-3.2-3b"]
        be = break_even(c, goal)
        tag = f"needs acceptance {be:.2f} to match it" if be else "cannot match it at any acceptance"
        extra = f", speedup at its own acceptance {measured[name]}: {speedup(measured[name], c):.2f}x" if name in measured else ""
        print(f"  {name:58s} c = {c:.3f}; {tag}{extra}")


if __name__ == "__main__":
    main()
