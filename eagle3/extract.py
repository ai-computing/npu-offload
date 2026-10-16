"""Solar-Open-100B bf16 teacher features for EAGLE-3, by layer streaming on one GPU.

A chunk of ~CHUNK tokens (many sequences, packed) goes through the 48 layers one layer at a time; each layer's
weights (attention + shared + 128 experts, ~4.3 GB) are loaded to the GPU once per chunk (the next layer is
prefetched on a background thread). Per token we keep the outputs of AUX layers (EAGLE-3 low/mid/high features)
and the final-norm hidden state (target logits = final @ lm_head^T, recomputed at training time).

usage: CUDA_VISIBLE_DEVICES=1 python3 extract.py data.pt out_dir [chunk_tokens]
       python3 extract.py --check      (argmax vs the NPU engine's exact tokens on data_p2)
"""
import os, sys, threading, time, queue
import torch
import torch.nn.functional as Fn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "solar_hybrid"))
from model import Weights, Rotary, rotate_half, BF  # noqa: E402

AUX = (1, 23, 44)
DEV = "cuda"


NAMES = dict(in_norm="input_layernorm.weight", post_norm="post_attention_layernorm.weight",
             wq="self_attn.q_proj.weight", wk="self_attn.k_proj.weight", wv="self_attn.v_proj.weight",
             wo="self_attn.o_proj.weight", sg="mlp.shared_experts.gate_proj.weight",
             su="mlp.shared_experts.up_proj.weight", sd="mlp.shared_experts.down_proj.weight",
             router="mlp.gate.weight", bias="mlp.gate.e_score_correction_bias")


def alloc_set(W, pin):
    c = W.cfg
    d = {k: torch.empty_like(W.get("model.layers.0." + n), dtype=torch.float32 if k in ("router", "bias") else BF,
                             device="cpu" if pin else DEV) for k, n in NAMES.items()}
    mk = lambda *sh: torch.empty(*sh, dtype=BF, device="cpu" if pin else DEV)
    d.update(eg=mk(c.E, c.F, c.H), eu=mk(c.E, c.F, c.H), ed=mk(c.E, c.H, c.F))
    return {k: v.pin_memory("cuda") for k, v in d.items()} if pin else d


def fill_layer(W, l, buf, pool):
    """copy layer l's weights into the pinned set `buf` (parallel memcpy from the safetensors mmap)."""
    p = f"model.layers.{l}."
    jobs = [(buf[k], p + n) for k, n in NAMES.items()]
    for e in range(W.cfg.E):
        q = f"{p}mlp.experts.{e}."
        jobs += [(buf["eg"][e], q + "gate_proj.weight"), (buf["eu"][e], q + "up_proj.weight"), (buf["ed"][e], q + "down_proj.weight")]
    list(pool.map(lambda j: j[0].copy_(W.get(j[1])), jobs))


def rms(x, w, eps):
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def route(c, h, router, bias):
    s = Fn.linear(h.float(), router).sigmoid()
    idx = torch.topk(s + bias, k=c.topk, dim=-1, sorted=False)[1]
    w = s.gather(1, idx)
    if c.norm_topk: w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return idx, w * c.scale


def layer_fwd(c, L, x, cos, sin, seqs):
    """x [N,H] bf16 packed; seqs = list of (start, len)."""
    N = x.shape[0]
    h = rms(x, L["in_norm"], c.eps)
    q = (h @ L["wq"].t()).view(N, c.nh, c.hd)
    k = (h @ L["wk"].t()).view(N, c.nkv, c.hd)
    v = (h @ L["wv"].t()).view(N, c.nkv, c.hd)
    q = q * cos[:, None] + rotate_half(q) * sin[:, None]
    k = k * cos[:, None] + rotate_half(k) * sin[:, None]
    a = torch.empty(N, c.nh, c.hd, dtype=BF, device=x.device)
    for s0, n in seqs:
        qs = q[s0:s0 + n].transpose(0, 1)[None]; ks = k[s0:s0 + n].transpose(0, 1)[None]; vs = v[s0:s0 + n].transpose(0, 1)[None]
        o = Fn.scaled_dot_product_attention(qs, ks, vs, is_causal=True, enable_gqa=True)
        a[s0:s0 + n] = o[0].transpose(0, 1)
    x2 = x + a.view(N, -1) @ L["wo"].t()
    h2 = rms(x2, L["post_norm"], c.eps)
    y = x2 + (Fn.silu(h2 @ L["sg"].t()) * (h2 @ L["su"].t())) @ L["sd"].t()
    idx, w = route(c, h2, L["router"], L["bias"])
    flat = idx.reshape(-1)
    order = torch.argsort(flat)
    tok = order // c.topk
    counts = torch.bincount(flat, minlength=c.E).tolist()
    acc = torch.zeros(N, c.H, dtype=torch.float32, device=x.device)
    wf = w.reshape(-1)[order]
    o0 = 0
    for e, n in enumerate(counts):
        if n == 0: continue
        t = tok[o0:o0 + n]
        he = h2[t]
        ye = (Fn.silu(he @ L["eg"][e].t()) * (he @ L["eu"][e].t())) @ L["ed"][e].t()
        acc.index_add_(0, t, ye.float() * wf[o0:o0 + n, None])
        o0 += n
    return y + acc.to(BF)


class Prefetch:
    """two pinned staging sets filled on a background thread; one GPU set overwritten per layer."""
    def __init__(self, W):
        from concurrent.futures import ThreadPoolExecutor
        self.W = W
        from safetensors import safe_open
        for f in set(W.map.values()):   # open every shard up front (W.get from many threads)
            if f not in W._h: W._h[f] = safe_open(os.path.join(W.dir, f), framework="pt")
        self.pin = [alloc_set(W, True), alloc_set(W, True)]
        self.gpu = alloc_set(W, False)
        self.pool = ThreadPoolExecutor(16)

    def start(self, n):
        self.q = queue.Queue(maxsize=1)
        self.free = [threading.Semaphore(1), threading.Semaphore(1)]
        def run():
            for l in range(n):
                self.free[l % 2].acquire()
                fill_layer(self.W, l, self.pin[l % 2], self.pool)
                self.q.put(l)
        threading.Thread(target=run, daemon=True).start()

    def get(self):
        l = self.q.get()
        for k, v in self.pin[l % 2].items(): self.gpu[k].copy_(v, non_blocking=True)
        torch.cuda.synchronize()
        self.free[l % 2].release()
        return self.gpu


PF = None


def forward_chunk(W, rot, embed, norm, samples):
    c = W.cfg
    ids = torch.cat([s for s, _ in samples]).long()
    seqs, pos, s0 = [], [], 0
    for s, _ in samples:
        seqs.append((s0, len(s))); pos.append(torch.arange(len(s))); s0 += len(s)
    cos, sin = rot(torch.cat(pos))
    cos, sin = cos.to(DEV), sin.to(DEV)
    x = embed[ids].to(DEV)
    aux = []
    global PF
    if PF is None: PF = Prefetch(W)
    PF.start(c.L)
    for l in range(c.L):
        t0 = time.time()
        L = PF.get()
        t1 = time.time()
        x = layer_fwd(c, L, x, cos, sin, seqs)
        torch.cuda.synchronize()
        if os.environ.get("VERBOSE"): print(f"  layer {l}: wait {t1 - t0:.2f}s compute {time.time() - t1:.2f}s", flush=True)
        if l in AUX: aux.append(x.cpu())
    fin = rms(x, norm, c.eps)
    return torch.cat(aux, -1), fin, seqs


def main():
    W = Weights(); c = W.cfg
    rot = Rotary(c.raw)
    embed = W.get("model.embed_tokens.weight")
    norm = W.get("model.norm.weight").to(DEV)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", "0.7")))
    if sys.argv[1] == "--check":
        import json
        ids = json.load(open("../track2_opt/solar_engine/data_p2/prompt.json"))["ids"]
        gen = [22, 4677, 6604, 12806, 4355, 7738, 14324, 5619, 4355, 15962, 8134, 4395]
        seq = torch.tensor(ids + gen, dtype=torch.int32)
        t0 = time.time()
        aux, fin, _ = forward_chunk(W, rot, embed, norm, [(seq, None)])
        lm = W.get("lm_head.weight").to(DEV)
        pred = (fin[len(ids) - 1:] @ lm.t()).argmax(-1).tolist()
        print("pred", pred[:len(gen)])
        print("gold", gen)
        print("match", sum(a == b for a, b in zip(pred, gen)), "/", len(gen), f"({time.time() - t0:.0f}s)")
        return
    samples = torch.load(sys.argv[1])
    out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 96 * 1024
    groups, cur, n = [], [], 0
    for s in samples:
        if n + len(s[0]) > chunk and cur: groups.append(cur); cur, n = [], 0
        cur.append(s); n += len(s[0])
    if cur: groups.append(cur)
    for gi, g in enumerate(groups):
        f = os.path.join(out, f"chunk{gi:04d}.pt")
        if os.path.exists(f): continue
        t0 = time.time()
        aux, fin, seqs = forward_chunk(W, rot, embed, norm, g)
        torch.save(dict(ids=[s for s, _ in g], mask=[m for _, m in g], aux=aux, fin=fin.cpu()), f + ".tmp")
        os.rename(f + ".tmp", f)
        print(f"[{gi + 1}/{len(groups)}] {sum(len(s) for s, _ in g)} tok in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
