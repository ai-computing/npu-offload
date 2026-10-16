"""EAGLE-3 draft head for Solar-Open-100B, trained offline on teacher features from extract.py.

Draft (one decoder layer, SpecForge-style):
  g_t   = fc([aux_low, aux_mid, aux_high]_t)                       target features at position t (3H -> H)
  in_t  = [input_norm(embed(x_{t+1})), hidden_norm(g_t)]          (2H)
  h_t   = g_t + o(attn(in))  ;  h_t = h_t + mlp(post_norm(h_t))     (q/k/v read 2H)
  q(x_{t+2}) = softmax(lm_head_D(norm(h_t)))                       draft vocab D (most frequent target tokens)
Training-time test (TTT): step j>0 feeds the step j-1 outputs h as the hidden input (shifted one position);
queries see the step-0 keys causally plus, on the diagonal, their own position's keys from steps 1..j.
Loss at step j, position t: CE(target p_{t+j+1} restricted to D, q), masked to assistant positions.

usage: CUDA_VISIBLE_DEVICES=1 python3 train.py feat_dir out.pt [epochs] [ttt]
"""
import glob, math, os, random, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as Fn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "solar_hybrid"))
from model import Weights, Rotary, rotate_half, BF  # noqa: E402

H, NH, NKV, HD, FF, DV = 4096, 32, 8, 128, 8192, 32768
DEV = "cuda"


class RMS(nn.Module):
    def __init__(self, n, eps=1e-5):
        super().__init__(); self.w = nn.Parameter(torch.ones(n)); self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (self.w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps))).to(x.dtype)


class Draft(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * H, H, bias=False)
        self.in_norm, self.hid_norm, self.post_norm, self.norm = RMS(H), RMS(H), RMS(H), RMS(H)
        self.q = nn.Linear(2 * H, NH * HD, bias=False)
        self.k = nn.Linear(2 * H, NKV * HD, bias=False)
        self.v = nn.Linear(2 * H, NKV * HD, bias=False)
        self.o = nn.Linear(NH * HD, H, bias=False)
        self.gate = nn.Linear(H, FF, bias=False); self.up = nn.Linear(H, FF, bias=False); self.down = nn.Linear(FF, H, bias=False)
        self.lm = nn.Linear(H, DV, bias=False)

    def qkv(self, emb, hid, cos, sin):
        """emb, hid [B,T,H] -> q [B,NH,T,HD], k/v [B,NKV,T,HD] (roped)."""
        x = torch.cat([self.in_norm(emb), self.hid_norm(hid)], -1)
        B, T, _ = x.shape
        q = self.q(x).view(B, T, NH, HD).transpose(1, 2)
        k = self.k(x).view(B, T, NKV, HD).transpose(1, 2)
        v = self.v(x).view(B, T, NKV, HD).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k, v

    def out(self, hid, a):
        """residual hid [B,T,H], attention output a [B,NH,T,HD] -> h [B,T,H], logits [B,T,DV]."""
        B, _, T, _ = a.shape
        h = hid + self.o(a.transpose(1, 2).reshape(B, T, -1))
        p = self.post_norm(h)
        h = h + self.down(Fn.silu(self.gate(p)) * self.up(p))
        return h, self.lm(self.norm(h))


def attend(q, ks, vs, kmask):
    """q [B,NH,T,HD]; ks/vs: list of [B,NKV,T,HD] (index 0: causal over positions, others: diagonal only);
    kmask [B,T] valid keys. Returns [B,NH,T,HD]."""
    rep = NH // NKV
    k0 = ks[0].repeat_interleave(rep, 1); v0 = vs[0].repeat_interleave(rep, 1)
    T = q.shape[2]
    sc = (q @ k0.transpose(-1, -2)).float() * HD ** -0.5                     # [B,NH,T,T]
    causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    sc = sc.masked_fill(~(causal[None, None] & kmask[:, None, None, :]), float("-inf"))
    diag = [((q * kj.repeat_interleave(rep, 1)).sum(-1, keepdim=True).float() * HD ** -0.5) for kj in ks[1:]]
    allsc = torch.cat([sc] + diag, -1)
    p = torch.softmax(allsc, -1).to(q.dtype)
    a = p[..., :T] @ v0
    for i, vj in enumerate(vs[1:]):
        a = a + p[..., T + i:T + i + 1] * vj.repeat_interleave(rep, 1)
    return a


def load_feats(d):
    seqs = []
    for f in sorted(glob.glob(os.path.join(d, "chunk*.pt"))):
        c = torch.load(f)
        o = 0
        for ids, m in zip(c["ids"], c["mask"]):
            n = len(ids)
            seqs.append((ids.long(), m, c["aux"][o:o + n], c["fin"][o:o + n]))
            o += n
    return seqs


def batches(seqs, tok_budget, rng):
    idx = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    out, cur = [], []
    for i in idx:
        L = len(seqs[i][0])
        if cur and (len(cur) + 1) * max(L, len(seqs[cur[0]][0])) > tok_budget:
            out.append(cur); cur = []
        cur.append(i)
    if cur: out.append(cur)
    rng.shuffle(out)
    return out


def make_batch(seqs, bi, embed, lm_d, rot):
    """returns emb [B,T,H] (embeds of x_{t+1}), aux [B,T,3H], tgt_logits [B,T,DV] (target at t+1), lmask [B,T], kmask, cos, sin"""
    L = max(len(seqs[i][0]) for i in bi) - 1
    B = len(bi)
    emb = torch.zeros(B, L, H, dtype=BF); aux = torch.zeros(B, L, 3 * H, dtype=BF); fin = torch.zeros(B, L + 1, H, dtype=BF)
    lmask = torch.zeros(B, L + 1, dtype=torch.bool); kmask = torch.zeros(B, L, dtype=torch.bool)
    for r, i in enumerate(bi):
        ids, m, a, f = seqs[i]
        n = len(ids) - 1
        emb[r, :n] = embed[ids[1:]]; aux[r, :n] = a[:n]; fin[r, :n + 1] = f; lmask[r, :n + 1] = m; kmask[r, :n] = True
    cos, sin = rot(torch.arange(L + 8))
    fin = fin.to(DEV)
    with torch.no_grad():
        tl = (fin @ lm_d.t()).float()                                        # [B,L+1,DV] target logits on D
    return emb.to(DEV), aux.to(DEV), tl, lmask.to(DEV), kmask.to(DEV), cos.to(DEV)[None, None], sin.to(DEV)[None, None]


def step_loss(model, emb, aux, tl, lmask, kmask, cos, sin, ttt):
    B, L, _ = emb.shape
    hid = model.fc(aux)
    ks, vs, losses, accs = [], [], [], []
    for j in range(ttt):
        e = emb if j == 0 else torch.cat([emb[:, j:], torch.zeros_like(emb[:, :j])], 1)
        q, k, v = model.qkv(e, hid, cos[:, :, j:j + L], sin[:, :, j:j + L])
        ks.append(k); vs.append(v)
        a = attend(q, ks, vs, kmask)
        h, lg = model.out(hid, a)
        # position t at step j predicts x_{t+j+2}: target logits at t+j+1
        T = L - j
        tgt = tl[:, j + 1:j + 1 + T]
        msk = lmask[:, j + 1:j + 1 + T] & kmask[:, :T]
        lp = Fn.log_softmax(lg[:, :T].float(), -1)
        p = torch.softmax(tgt, -1)
        ce = -(p * lp).sum(-1)
        losses.append((ce * msk).sum() / msk.sum().clamp(min=1))
        accs.append(((lg[:, :T].argmax(-1) == tgt.argmax(-1)) & msk).sum().float() / msk.sum().clamp(min=1))
        # index t stays the chain start: step j+1 reads this step's output h_t with the token one further on
        hid = h
    w = [0.8 ** j for j in range(ttt)]
    return sum(wi * l for wi, l in zip(w, losses)) / sum(w), [a.item() for a in accs]


def main():
    feat, out = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    ttt = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", "0.7")))
    W = Weights()
    rot = Rotary(W.cfg.raw)
    embed = W.get("model.embed_tokens.weight")
    seqs = load_feats(feat)
    rng = random.Random(0)
    rng.shuffle(seqs)
    nval = max(8, len(seqs) // 50)
    val, seqs = seqs[:nval], seqs[nval:]
    # draft vocab: most frequent target-argmax tokens on the training data
    lm = W.get("lm_head.weight").to(DEV)
    cnt = torch.zeros(lm.shape[0], dtype=torch.long, device=DEV)
    for ids, m, a, f in seqs:
        am = (f.to(DEV) @ lm.t()).argmax(-1)
        cnt += torch.bincount(am, minlength=lm.shape[0])
    d2t = torch.sort(torch.topk(cnt, DV).indices).values
    cover = cnt[d2t].sum().item() / cnt.sum().item()
    print(f"{len(seqs)} train / {len(val)} val seqs; draft vocab {DV} covers {cover:.4f} of target argmaxes", flush=True)
    lm_d = lm[d2t].contiguous(); del lm
    model = Draft().to(DEV)
    with torch.no_grad():
        model.lm.weight.copy_(lm_d.float())
    model = model.to(BF)
    for p in model.parameters(): p.data = p.data.float()  # fp32 master weights, bf16 autocast
    opt = torch.optim.AdamW(model.parameters(), lr=float(os.environ.get("LR", "2e-4")), betas=(0.9, 0.95), weight_decay=0.0)
    budget = int(os.environ.get("TOK_BUDGET", "16384"))
    total = epochs * len(batches(seqs, budget, random.Random(1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    step, t0 = 0, time.time()
    for ep in range(epochs):
        for bi in batches(seqs, budget, rng):
            b = make_batch(seqs, bi, embed, lm_d, rot)
            with torch.autocast("cuda", dtype=BF):
                loss, accs = step_loss(model, *b, ttt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step(); sched.step(); step += 1
            if step % 50 == 0:
                print(f"ep {ep} step {step}/{total} loss {loss.item():.3f} acc {[round(a, 3) for a in accs]} "
                      f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        # validation
        model.eval(); va = []
        with torch.no_grad(), torch.autocast("cuda", dtype=BF):
            for bi in batches(val, budget, random.Random(2)):
                _, accs = step_loss(model, *make_batch(val, bi, embed, lm_d, rot), ttt)
                va.append(accs)
        model.train()
        print(f"== epoch {ep} val acc per step {[round(sum(x[j] for x in va) / len(va), 3) for j in range(ttt)]}", flush=True)
        torch.save(dict(state=model.state_dict(), d2t=d2t.cpu(), cfg=dict(H=H, NH=NH, NKV=NKV, HD=HD, FF=FF, DV=DV)), out)


if __name__ == "__main__":
    main()
