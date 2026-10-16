"""P0 CPU microbench: Solar expert FFN (H=4096, F=1280 SwiGLU) on host CPU (torch/oneDNN, AMX bf16).

Measures k distinct experts (cold weights, rotated through a pool larger than L3) x T tokens.
Run under numactl to pin, e.g.:
  numactl --cpunodebind=0 --membind=0 python3 cpu_microbench.py --threads 3
  python3 cpu_microbench.py --threads 24
"""
import argparse, json, os, statistics, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=24)
ap.add_argument("--tokens", default="1,8,32")
ap.add_argument("--ks", default="1,4,8")
ap.add_argument("--pool", type=int, default=64, help="#distinct experts rotated (64*31.5MB=2GB >> L3)")
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--label", default="")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "cpu_microbench.jsonl"))
a = ap.parse_args()
torch.set_num_threads(a.threads)
H, F, BF = 4096, 1280, torch.bfloat16
os.makedirs(os.path.dirname(a.out), exist_ok=True)

torch.manual_seed(0)
pool = [((torch.randn(2 * F, H) * 0.02).to(BF), (torch.randn(H, F) * 0.02).to(BF)) for _ in range(a.pool)]


def ffn(x, ws, g):
    out = torch.zeros(x.shape[0], H, dtype=torch.float32)
    for i, (wgu, wd) in enumerate(ws):
        gu = x @ wgu.t()
        act = torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]
        out += (act @ wd.t()).float() * g
    return out


with torch.no_grad():
    for T in [int(t) for t in a.tokens.split(",")]:
        for k in [int(x) for x in a.ks.split(",")]:
            x = torch.randn(T, H).to(BF)
            c = [0]
            def run():
                s = c[0]; c[0] += k
                return ffn(x, [pool[(s + i) % a.pool] for i in range(k)], 1.0 / k)
            for _ in range(3): run()
            ts = []
            for _ in range(a.iters):
                t0 = time.perf_counter(); run(); ts.append(time.perf_counter() - t0)
            med = statistics.median(ts)
            wbytes = k * 3 * H * F * 2
            rec = dict(sec="cpu_ffn", label=a.label, threads=a.threads, T=T, k=k, ms=med * 1e3,
                       eff_GBps=wbytes / med / 1e9)
            print(json.dumps(rec), flush=True)
            with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
