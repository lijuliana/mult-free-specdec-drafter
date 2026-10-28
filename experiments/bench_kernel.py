"""Experiment 4 (first pass): time a fused lookup-table projection against a dense BF16 projection.

A Triton kernel reads each chunk's 4-bit code, gathers that 8-bit table row and adds it into an int32
accumulator, all in one pass. A second kernel also computes the code itself (4 random hyperplanes per
chunk). Both are timed at batch 1 with CUDA graphs against cuBLAS BF16 matrix-vector products, on one
Llama-3.2-1B down-projection and on all 112 block projections of the model. Tables and codes are random:
the timing depends only on shapes, and the kernel output is checked against a PyTorch reference.

Needs a CUDA GPU and Triton. Usage: python experiments/bench_kernel.py
"""
import torch, triton, triton.language as tl
torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
K, L = 16, 8

@triton.jit
def lut_kernel(T, codes, out, m, K: tl.constexpr, CH: tl.constexpr, BN: tl.constexpr):
    pn = tl.program_id(0); pc = tl.program_id(1)
    cols = pn * BN + tl.arange(0, BN)
    cidx = pc * CH + tl.arange(0, CH)
    code = tl.load(codes + cidx).to(tl.int64)
    rows = cidx.to(tl.int64) * K + code
    v = tl.load(T + rows[:, None] * m + cols[None, :])
    tl.atomic_add(out + cols, tl.sum(v.to(tl.int32), axis=0))

@triton.jit
def lut_hash_kernel(X, P, B, T, out, m, L: tl.constexpr, K: tl.constexpr, CH: tl.constexpr, BN: tl.constexpr):
    # same, but encodes the input inside the kernel: 4 hyperplanes per chunk -> 4-bit code
    pn = tl.program_id(0); pc = tl.program_id(1)
    cols = pn * BN + tl.arange(0, BN)
    cidx = pc * CH + tl.arange(0, CH)
    l = tl.arange(0, L)
    xs = tl.load(X + cidx[:, None] * L + l[None, :]).to(tl.float32)
    code = tl.zeros([CH], dtype=tl.int32)
    for b in tl.static_range(4):
        p = tl.load(P + (cidx[:, None] * 4 + b) * L + l[None, :]).to(tl.float32)
        s = tl.sum(xs * p, axis=1) - tl.load(B + cidx * 4 + b)
        code += (s > 0).to(tl.int32) << b
    rows = cidx.to(tl.int64) * K + code.to(tl.int64)
    v = tl.load(T + rows[:, None] * m + cols[None, :])
    tl.atomic_add(out + cols, tl.sum(v.to(tl.int32), axis=0))

class Proj:
    def __init__(s, d, m):
        s.d, s.m, s.C = d, m, d // L
        s.W = torch.randn(m, d, device=dev, dtype=torch.bfloat16)
        s.T = torch.randint(-128, 127, (s.C * K, m), device=dev, dtype=torch.int8)
        s.codes = torch.randint(0, K, (s.C,), device=dev, dtype=torch.int32)
        s.P = torch.randn(s.C, 4, L, device=dev, dtype=torch.float16)
        s.B = torch.zeros(s.C, 4, device=dev, dtype=torch.float32)
        s.out = torch.zeros(m, device=dev, dtype=torch.int32)
        s.y = torch.empty(m, device=dev, dtype=torch.bfloat16)
    def dense(s, x): torch.matmul(s.W, x, out=s.y)
    def lut(s, x, CH, BN):
        s.out.zero_(); lut_kernel[(s.m // BN, s.C // CH)](s.T, s.codes, s.out, s.m, K=K, CH=CH, BN=BN)
    def luth(s, x, CH, BN):
        s.out.zero_(); lut_hash_kernel[(s.m // BN, s.C // CH)](x, s.P, s.B, s.T, s.out, s.m, L=L, K=K, CH=CH, BN=BN)

def graph_time(fn, reps=200):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(10): g.replay()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); a.record()
    for _ in range(reps): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1e3  # microseconds

print(torch.cuda.get_device_name(), "torch", torch.__version__, "triton", triton.__version__)

# correctness
p = Proj(8192, 2048); x = torch.randn(8192, device=dev, dtype=torch.bfloat16)
p.lut(x, 32, 128)
ref = p.T.view(p.C, K, p.m)[torch.arange(p.C, device=dev), p.codes.long()].int().sum(0)
assert torch.equal(p.out, ref), "lut kernel mismatch"
xs = x.float().view(p.C, 1, L); bits = ((xs * p.P.float()).sum(-1) - p.B > 0).int()
hc = (bits * torch.tensor([1, 2, 4, 8], device=dev)).sum(-1)
p.luth(x, 32, 128)
ref = p.T.view(p.C, K, p.m)[torch.arange(p.C, device=dev), hc.long()].int().sum(0)
print("hash-encode exact match:", torch.equal(p.out, ref), "(fp rounding at plane boundaries can differ)")

# tune tile sizes on the down_proj shape
best = {}
for name, fn in (("lut", Proj.lut), ("luth", Proj.luth)):
    res = []
    for CH in (16, 32, 64, 128):
        for BN in (64, 128, 256, 512):
            ps = [Proj(8192, 2048) for _ in range(8)]
            t = graph_time(lambda: [fn(q, x, CH, BN) for q in ps]) / 8
            res.append((t, CH, BN)); del ps
    res.sort(); best[name] = res[0][1:]
    print(name, "best tiles CH,BN =", best[name], f"{res[0][0]:.1f} us")

# single layer: 8 distinct down_projs per graph so nothing stays in L2
xd = torch.randn(8192, device=dev, dtype=torch.bfloat16)
ps = [Proj(8192, 2048) for _ in range(8)]
td = graph_time(lambda: [q.dense(xd) for q in ps]) / 8
tl_ = graph_time(lambda: [q.lut(xd, *best["lut"]) for q in ps]) / 8
th = graph_time(lambda: [q.luth(xd, *best["luth"]) for q in ps]) / 8
bd, bt = 8192 * 2048 * 2, 8192 // L * 2048
print(f"\nDOWN_PROJ 8192->2048, batch 1")
print(f"  dense bf16 cuBLAS : {td:7.1f} us  ({bd/td/1e3:.0f} GB/s)")
print(f"  LUT lookup only   : {tl_:7.1f} us  ({bt/tl_/1e3:.0f} GB/s)  speedup {td/tl_:.1f}x")
print(f"  LUT + hash encode : {th:7.1f} us  speedup {td/th:.1f}x")
del ps; torch.cuda.empty_cache()

# all 112 block projections of Llama-3.2-1B
shapes = [(2048, 2048), (2048, 512), (2048, 512), (2048, 2048), (2048, 8192), (2048, 8192), (8192, 2048)]
layers = [[Proj(d, m) for d, m in shapes] for _ in range(16)]
xin = {d: torch.randn(d, device=dev, dtype=torch.bfloat16) for d in (2048, 8192)}
td = graph_time(lambda: [q.dense(xin[q.d]) for ly in layers for q in ly], reps=50)
tl_ = graph_time(lambda: [q.lut(xin[q.d], min(best["lut"][0], q.C), min(best["lut"][1], q.m)) for ly in layers for q in ly], reps=50)
th = graph_time(lambda: [q.luth(xin[q.d], min(best["luth"][0], q.C), min(best["luth"][1], q.m)) for ly in layers for q in ly], reps=50)
print(f"\nALL 112 PROJECTIONS of Llama-3.2-1B, batch 1 (one draft token)")
print(f"  dense bf16        : {td:7.1f} us")
print(f"  LUT lookup only   : {tl_:7.1f} us  speedup {td/tl_:.1f}x")
print(f"  LUT + hash encode : {th:7.1f} us  speedup {td/th:.1f}x")
