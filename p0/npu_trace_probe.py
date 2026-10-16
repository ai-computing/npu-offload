"""Probe furiosa.torch RNGDProfiler: do our compiled graphs emit NPU (TUC) spans? Export a Chrome trace.

  python3 npu_trace_probe.py [--mesh]
Writes results/trace_probe.json (open in chrome://tracing or ui.perfetto.dev).
"""
import sys, os, time, json, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "solar_hybrid"))
import furiosa.torch  # noqa
from furiosa.torch import native_device as nd
nd.set_fusion(8)
if "--mesh" in sys.argv or True:
    nd.set_mesh_kind(nd.MeshKind.Mesh(2))
from furiosa.torch.profiler import RNGDProfiler
from furiosa.torch.custom_ops import CompileModule
BF = torch.bfloat16; H, F = 4096, 1280; DEV = "rngd:0"; S = 16; K = 8; T = 1


class GExpG(torch.nn.Module):
    def forward(self, h, G, ids, Wgu, Wd):
        gu = torch.matmul(h.unsqueeze(0), Wgu[ids].transpose(1, 2))
        a = torch.nn.functional.silu(gu[..., :F]) * gu[..., F:]
        y = torch.matmul(a, Wd[ids].transpose(1, 2))
        return (y * G.t().unsqueeze(-1)).sum(0)


z = lambda *s: torch.zeros(*s, dtype=BF)
m = CompileModule.from_module(GExpG(), (z(T, H), z(T, K), torch.zeros(K, dtype=torch.int32), z(S, 2 * F, H), z(S, H, F)))
m.to(DEV)
print("profile_tensors:", m.edf_module.profile_tensors, flush=True)
print("stack sizes:", getattr(m.edf_module, "stack_sizes", None), "dram_shared:", m.edf_module.stack_is_dram_shared,
      "io_blocks:", getattr(m.edf_module, "io_block_sizes", None), m.edf_module.io_block_is_dram_shared, flush=True)
Wgu = (torch.randn(S, 2 * F, H) * .02).to(BF).to(DEV); Wd = (torch.randn(S, H, F) * .02).to(BF).to(DEV)
h = torch.randn(T, H).to(BF).to(DEV); G = torch.rand(T, K).to(BF).to(DEV); ids = torch.arange(K, dtype=torch.int32).to(DEV)
for _ in range(3): m(h, G, ids, Wgu, Wd)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "trace_probe.json")
with RNGDProfiler() as prof:
    for i in range(5):
        profs = m.generate_profiles(torch.device(DEV)) if m.edf_module.profile_tensors else None
        y = m(h, G, ids, Wgu, Wd, profiles=profs)
        y.cpu()
        if profs:
            prof.load_spans_from_profiles(m.edf, profs)
prof.export_chrome_trace(out)
ev = json.load(open(out))
names = {}
for e in ev:
    if e.get("ph") == "X":
        names.setdefault((e.get("pid"), e["name"]), []).append(e.get("dur", 0))
for (pid, n), ds in sorted(names.items(), key=lambda kv: -sum(kv[1]))[:30]:
    print(f"{str(pid)[:20]:20s} {n[:60]:60s} n={len(ds):3d} avg_us={sum(ds)/len(ds):9.1f}")
print("trace:", out)
