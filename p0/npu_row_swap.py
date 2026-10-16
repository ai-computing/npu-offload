"""Can we overwrite ONE expert row of a resident device table in place (for online expert swap)?

  python3 npu_row_swap.py
Checks: (1) table[i].copy_(host) on a device tensor, (2) gather graph sees the new row, other rows intact,
(3) latency per row (31.5MB), (4) alternatives if slice copy_ is unsupported.
"""
import time, statistics, traceback, torch
import furiosa.torch  # noqa
from furiosa.torch import native_device as nd
nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))
from furiosa.torch.custom_ops import CompileModule
BF = torch.bfloat16; H, F = 4096, 1280; DEV = "rngd:0"; S = 24; K = 8


class GExpG(torch.nn.Module):
    def forward(self, h, G, ids, Wgu, Wd):
        gu = torch.matmul(h.unsqueeze(0), Wgu[ids].transpose(1, 2))
        a = torch.nn.functional.silu(gu[..., :F]) * gu[..., F:]
        y = torch.matmul(a, Wd[ids].transpose(1, 2))
        return (y * G.t().unsqueeze(-1)).sum(0)


torch.manual_seed(0)
Wgu = (torch.randn(S, 2 * F, H) * .02).to(BF); Wd = (torch.randn(S, H, F) * .02).to(BF)
h = torch.randn(1, H).to(BF); G = torch.full((1, K), 0.125).to(BF)
m = CompileModule.from_module(GExpG(), (h, G, torch.zeros(K, dtype=torch.int32), Wgu, Wd)); m.to(DEV)
Wgu_d, Wd_d = Wgu.to(DEV), Wd.to(DEV)
hd, Gd = h.to(DEV), G.to(DEV)
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()


def check(ids, tag):
    ref = GExpG()(h.float(), G.float(), ids.long(), Wgu.float(), Wd.float())
    got = m(hd, Gd, ids.to(DEV), Wgu_d, Wd_d).cpu()
    print(f"{tag}: rel={rel(got, ref):.4f}", flush=True)
    return rel(got, ref)


ids = torch.arange(K, dtype=torch.int32)
check(ids, "before swap")
# new expert into row 3
ngu = (torch.randn(2 * F, H) * .02).to(BF); nd_ = (torch.randn(H, F) * .02).to(BF)
methods = {
    "slice_copy_": lambda i: (Wgu_d[i].copy_(ngu), Wd_d[i].copy_(nd_)),
    "index_put_": lambda i: (Wgu_d.index_put_((torch.tensor([i]).to(DEV),), ngu.unsqueeze(0).to(DEV)),
                             Wd_d.index_put_((torch.tensor([i]).to(DEV),), nd_.unsqueeze(0).to(DEV))),
    "setitem": lambda i: (Wgu_d.__setitem__(i, ngu.to(DEV)), Wd_d.__setitem__(i, nd_.to(DEV))),
}
ok_method = None
for name, fn in methods.items():
    try:
        fn(3)
        Wgu[3] = ngu; Wd[3] = nd_
        r = check(ids, f"after {name}")
        r2 = check(torch.tensor([8, 9, 10, 11, 12, 13, 14, 15], dtype=torch.int32), f"untouched rows ({name})")
        if r < 0.02 and r2 < 0.02:
            ok_method = name; break
    except Exception as e:
        print(f"{name} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
print("WORKING METHOD:", ok_method, flush=True)
if ok_method:
    fn = methods[ok_method]; ts = []
    for j in range(10):
        i = j % S
        t0 = time.perf_counter(); fn(i); torch.zeros(1).to(DEV).cpu(); ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    print(f"row swap latency: {med*1e3:.2f} ms  ({3*H*F*2/med/1e9:.1f} GB/s)", flush=True)
    # restore reference: rows 0..9 now hold ngu/nd_
    for j in range(10): Wgu[j] = ngu; Wd[j] = nd_
    check(ids, "after timed swaps")
