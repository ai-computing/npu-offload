"""Standalone check of expert_srv with real Solar experts: correctness vs host math + latency."""
import os, sys, time, statistics, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Weights, swiglu_ffn
import opt_client as OC
chip = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NL = 2
W = Weights(); c = W.cfg
assign = [list(range(32)) for _ in range(NL)]
hx = [[W.expert(l, e) for e in range(32)] for l in range(NL)]
wdir = "/dev/shm/solar_srv_test"
t0 = time.time(); OC.export_tables(hx, assign, NL, wdir); print(f"export {time.time()-t0:.1f}s", flush=True)
log = open("/home/gspark/workspace/npu_offload/logs/expert_srv_test.log", "w")
srv = OC.OptExpertServer(wdir, NL, chip=chip, log=log)
print(f"server ready in {srv.t_load:.1f}s", flush=True)
torch.manual_seed(0)
x = (torch.randn(c.H) * 0.5).to(torch.bfloat16)
for l in range(NL):
    slots = [3, 17, 25, 31, 0, 8, 12, 29][: 8 if l == 0 else 5]
    got = srv.ffn(l, x, slots)
    for k, s in enumerate(slots):
        ref = swiglu_ffn(x[None], *hx[l][s]).float()[0]
        rel = ((got[k] - ref).norm() / ref.norm()).item()
        print(f"layer {l} slot {s}: rel_err={rel:.4f}", flush=True)
ts = []; sv = []
for i in range(200):
    t = time.perf_counter(); srv.ffn(i % NL, x, [(i + k) % 32 for k in range(8)]); ts.append(time.perf_counter() - t); sv.append(srv.last_us())
print(f"client roundtrip p50={statistics.median(ts)*1e6:.0f}us  server-side p50={statistics.median(sv):.0f}us", flush=True)
srv.close()
