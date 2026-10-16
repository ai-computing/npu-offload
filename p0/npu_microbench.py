"""P0 NPU microbench for Solar-Open-100B expert offload.

(1) H2D / D2H latency + bandwidth vs size (torch .to(rngd) / .cpu())
(2) tiny-graph round trip (upload x -> run -> download y)
(3) Solar expert FFN (H=4096, F=1280, SwiGLU) with weights as resident runtime inputs:
    k experts per launch (k=1,2,4,8), batch rows T. Each expert's weights are separate input
    tensors, so swapping which expert runs = passing a different resident tensor (no gather).

  python3 npu_microbench.py [--fusion] [--dev rngd:0] [--out results/npu_microbench.jsonl]
"""
import argparse, json, os, statistics, sys, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--dev", default="rngd:0")
ap.add_argument("--fusion", action="store_true", help="set_fusion(8)+Mesh(2) before anything")
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--tokens", default="1,8,32")
ap.add_argument("--ks", default="1,2,4,8")
ap.add_argument("--skip", default="", help="comma list of sections to skip: xfer,rt,ffn")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "npu_microbench.jsonl"))
a = ap.parse_args()

import furiosa.torch  # noqa: F401
if a.fusion:
    from furiosa.torch import native_device as nd
    nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))
from furiosa.torch.custom_ops import CompileModule

BF = torch.bfloat16
H, F = 4096, 1280
DEV = a.dev
skip = set(filter(None, a.skip.split(",")))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
tag = dict(dev=DEV, fusion=a.fusion)


def emit(rec):
    rec = {**tag, **rec}
    print(json.dumps(rec), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


def timeit(fn, iters, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts)


# ---------------- (1) transfers ----------------
if "xfer" not in skip:
    for nbytes in [4 << 10, 64 << 10, 1 << 20, 16 << 20, 128 << 20, 512 << 20, 1 << 30]:
        n = nbytes // 2
        h = torch.randn(n, dtype=torch.float32).to(BF)
        it = max(3, min(a.iters, (256 << 20) // nbytes * 4))
        med, mn = timeit(lambda: h.to(DEV), it)
        d = h.to(DEV)
        med2, mn2 = timeit(lambda: d.cpu(), it)
        emit(dict(sec="xfer", bytes=nbytes, h2d_ms=med * 1e3, h2d_GBps=nbytes / med / 1e9,
                  d2h_ms=med2 * 1e3, d2h_GBps=nbytes / med2 / 1e9))
        del d


# ---------------- (2) tiny graph round trip ----------------
class _Add(torch.nn.Module):
    def forward(self, x, y): return x + y


if "rt" not in skip:
    m = CompileModule.from_module(_Add(), (torch.zeros(H, dtype=BF), torch.zeros(H, dtype=BF)))
    m.to(DEV)
    yd = torch.zeros(H, dtype=BF).to(DEV)
    xh = torch.randn(H).to(BF)
    xd = xh.to(DEV)
    med_run, _ = timeit(lambda: m(xd, yd), a.iters)                      # resident in, device out
    med_rt, mn_rt = timeit(lambda: m(xh.to(DEV), yd).cpu(), a.iters)      # host in -> host out
    emit(dict(sec="roundtrip", n=H, run_only_ms=med_run * 1e3, h2d_run_d2h_ms=med_rt * 1e3,
              h2d_run_d2h_min_ms=mn_rt * 1e3))


# ---------------- (3) expert FFN ----------------
def make_ffn(k):
    class FFN(torch.nn.Module):
        # x [T,H], g [T,k] routing weights; per expert: wgu [2F,H], wd [H,F]
        def forward(self, x, g, *ws):
            out = None
            for i in range(k):
                wgu, wd = ws[2 * i], ws[2 * i + 1]
                gu = x @ wgu.t()                      # [T,2F]
                a_ = torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]
                y = (a_ @ wd.t()) * g[:, i:i + 1]     # [T,H]
                out = y if out is None else out + y
            return out
    return FFN()


if "ffn" not in skip:
    torch.manual_seed(0)
    NE = 16  # resident pool of distinct experts to rotate through (pointer swap)
    pool = [((torch.randn(2 * F, H) * 0.02).to(BF), (torch.randn(H, F) * 0.02).to(BF)) for _ in range(NE)]
    t0 = time.perf_counter()
    pool_d = [(wgu.to(DEV), wd.to(DEV)) for wgu, wd in pool]
    emit(dict(sec="upload_pool", n_experts=NE, MB=NE * 3 * H * F * 2 / 1e6, s=time.perf_counter() - t0))
    for T in [int(t) for t in a.tokens.split(",")]:
        for k in [int(x) for x in a.ks.split(",")]:
            x = (torch.randn(T, H)).to(BF); g = torch.full((T, k), 1.0 / k).to(BF)
            ex = [w for i in range(k) for w in pool[i]]
            t0 = time.perf_counter()
            m = CompileModule.from_module(make_ffn(k), (x, g, *ex)); m.to(DEV)
            comp = time.perf_counter() - t0
            # correctness vs torch on host (first call)
            ref = make_ffn(k)(x.float(), g.float(), *[w.float() for w in ex])
            xd, gd = x.to(DEV), g.to(DEV)
            got = m(xd, gd, *[w for i in range(k) for w in pool_d[i]]).cpu().float()
            rel = ((got - ref).norm() / ref.norm()).item()
            # rotate expert sets each call -> proves pointer swap w/o recompile
            sets = [[w for i in range(s, s + k) for w in pool_d[i % NE]] for s in range(0, NE, k)]
            c = [0]
            def run():
                ws = sets[c[0] % len(sets)]; c[0] += 1
                return m(xd, gd, *ws)
            med_dev, _ = timeit(run, a.iters)
            def run_rt():
                ws = sets[c[0] % len(sets)]; c[0] += 1
                return m(x.to(DEV), g.to(DEV), *ws).cpu()
            med_rt, _ = timeit(run_rt, a.iters)
            wbytes = k * 3 * H * F * 2
            emit(dict(sec="ffn", T=T, k=k, compile_s=comp, rel_err=rel, dev_ms=med_dev * 1e3,
                      host_rt_ms=med_rt * 1e3, eff_GBps=wbytes / med_dev / 1e9))
