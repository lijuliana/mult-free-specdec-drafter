import sys; sys.path.insert(0, "experiments")
import torch, bench_drafter as B
class Null:
    def __init__(s, d, m): s.y = torch.zeros(1, m, device="cuda", dtype=torch.bfloat16)
    def __call__(s, x): return s.y
orig = B.Proj
def P(d, m, kind):
    return Null(d, m) if kind == "none" else orig(d, m, kind)
B.Proj = P
for name, kinds, head in [("no projections, 16-bit head", "none", "bf16"), ("no projections, no head", "none", "none")]:
    m = B.Model("llama-3.2-1b", lambda l, n: kinds, head)
    print(name, round(B.time_step(m), 3), "ms")
# per projection: 8 copies in one graph, per call
class Many:
    def __init__(s, d, m, kind): s.ps=[orig(d, m, kind) for _ in range(8)]; s.x=torch.randn(1,d,device="cuda",dtype=torch.bfloat16)
    def step(s):
        for p in s.ps: p(s.x)
for d, m, nm in [(2048,2048,"q/o 2048x2048"),(2048,512,"k/v 2048x512"),(2048,8192,"gate/up 2048x8192"),(8192,2048,"down 8192x2048"),(2048,128256,"head 2048x128256")]:
    row=[]
    for kind in ("bf16","int4","lut"):
        mm=Many(d,m,kind); row.append(f"{kind} {B.time_step(mm,50)/8*1000:.1f}us"); del mm; torch.cuda.empty_cache()
    print(nm, " | ".join(row))
