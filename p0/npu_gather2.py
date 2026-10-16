"""Re-test device-side expert gather with vendor-style W[ids_int32] (stacked table), print real cause.

  python3 npu_gather2.py none|fusion|mesh  [T] [K] [E]
"""
import sys, time, statistics, traceback, torch
mode = sys.argv[1]; T = int(sys.argv[2]) if len(sys.argv) > 2 else 1
K = int(sys.argv[3]) if len(sys.argv) > 3 else 8; E = int(sys.argv[4]) if len(sys.argv) > 4 else 64
import furiosa.torch  # noqa
from furiosa.torch import native_device as nd
if mode in ("fusion", "mesh"):
    nd.set_fusion(8)
if mode == "mesh":
    nd.set_mesh_kind(nd.MeshKind.Mesh(2))
from furiosa.torch.custom_ops import CompileModule
BF = torch.bfloat16; H, F = 4096, 1280; DEV = "rngd:0"


class GatherMoE(torch.nn.Module):
    def forward(self, x, G, ids, Wgu, Wd):
        wgu = Wgu[ids]                                    # [K,2F,H]  aten.index.Tensor, int32 ids
        wd = Wd[ids]                                      # [K,H,F]
        gu = torch.matmul(x.unsqueeze(0), wgu.transpose(1, 2))       # [K,T,2F]
        a = torch.nn.functional.silu(gu[..., :F]) * gu[..., F:]
        y = torch.matmul(a, wd.transpose(1, 2))           # [K,T,H]
        return (y * G.t().unsqueeze(-1)).sum(0)           # [T,H]


torch.manual_seed(0)
Wgu = (torch.randn(E, 2 * F, H) * .02).to(BF); Wd = (torch.randn(E, H, F) * .02).to(BF)
x = torch.randn(T, H).to(BF); G = (torch.rand(T, K) * 0.2).to(BF)
ids = torch.randperm(E)[:K].to(torch.int32)
t0 = time.time()
try:
    m = CompileModule.from_module(GatherMoE(), (x, G, ids, Wgu, Wd)); m.to(DEV)
except Exception as e:
    print(f"COMPILE FAIL mode={mode}: {type(e).__name__}: {e}"); traceback.print_exc()
    c = e.__cause__ or e.__context__
    while c is not None:
        print("  CAUSE:", type(c).__name__, str(c)[:1500]); c = c.__cause__ or c.__context__
    sys.exit(1)
print(f"compiled mode={mode} T={T} K={K} E={E} in {time.time()-t0:.1f}s")
Wg_d, Wd_d = Wgu.to(DEV), Wd.to(DEV)
ref = GatherMoE()(x.float(), G.float(), ids.long(), Wgu.float(), Wd.float())
got = m(x.to(DEV), G.to(DEV), ids.to(DEV), Wg_d, Wd_d).cpu().float()
print("rel_err", ((got - ref).norm() / ref.norm()).item())
xd, Gd = x.to(DEV), G.to(DEV)
idsets = [torch.randperm(E)[:K].to(torch.int32).to(DEV) for _ in range(8)]
for i in range(3): m(xd, Gd, idsets[i], Wg_d, Wd_d)
ts = []
for i in range(30):
    t = time.perf_counter(); m(xd, Gd, idsets[i % 8], Wg_d, Wd_d); ts.append(time.perf_counter() - t)
med = statistics.median(ts)
print(f"RESULT mode={mode} T={T} K={K} dev_ms={med*1e3:.3f} eff_GBps={K*3*H*F*2/med/1e9:.1f}")
# check other id sets numerically
for j in range(2):
    i2 = torch.randperm(E)[:K].to(torch.int32)
    r = GatherMoE()(x.float(), G.float(), i2.long(), Wgu.float(), Wd.float())
    g = m(xd, Gd, i2.to(DEV), Wg_d, Wd_d).cpu().float()
    print("rel_err ids", j, ((g - r).norm() / r.norm()).item())
