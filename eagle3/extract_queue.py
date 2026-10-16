"""Feature producer for queue training: shard `s` of `n` of data.pt -> chunk files in a queue dir (e.g. /dev/shm).

usage: TORCH_DEVICE_BACKEND_AUTOLOAD=0 CUDA_VISIBLE_DEVICES=<gpu> python3 extract_queue.py data.pt s n qdir [chunk] [maxq]
Writes qdir/s{s}_{i:05d}.pt ({ids, mask, aux, fin}); waits while the shard has >= maxq files queued; qdir/s{s}.done at the end.
"""
import glob, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract  # noqa: E402
from model import Weights, Rotary  # noqa: E402


def groups_of(samples, chunk):
    out, cur, n = [], [], 0
    for x in samples:
        if n + len(x[0]) > chunk and cur: out.append(cur); cur, n = [], 0
        cur.append(x); n += len(x[0])
    if cur: out.append(cur)
    return out


def main():
    data, s, n, q = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    chunk = int(sys.argv[5]) if len(sys.argv) > 5 else 98304
    maxq = int(sys.argv[6]) if len(sys.argv) > 6 else 6
    os.makedirs(q, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", "0.4")))
    W = Weights(); rot = Rotary(W.cfg.raw)
    embed = W.get("model.embed_tokens.weight"); norm = W.get("model.norm.weight").to(extract.DEV)
    groups = groups_of(torch.load(data)[s::n], chunk)
    start = int(os.environ.get("START", 0))
    print(f"shard {s}/{n}: {len(groups)} chunks", flush=True)
    for gi in range(start, len(groups)):
        f = os.path.join(q, f"s{s}_{gi:05d}.pt")
        while len(glob.glob(os.path.join(q, f"s{s}_*.pt"))) >= maxq: time.sleep(2)
        t0 = time.time()
        with torch.no_grad():
            aux, fin, _ = extract.forward_chunk(W, rot, embed, norm, groups[gi])
        torch.save(dict(ids=[a for a, _ in groups[gi]], mask=[m for _, m in groups[gi]], aux=aux, fin=fin.cpu()), f + ".tmp")
        os.rename(f + ".tmp", f)
        torch.cuda.empty_cache()
        print(f"[s{s} {gi + 1}/{len(groups)}] {time.time() - t0:.0f}s", flush=True)
    open(os.path.join(q, f"s{s}.done"), "w").write(str(len(groups)))


if __name__ == "__main__":
    main()
