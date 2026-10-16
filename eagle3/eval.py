"""Offline acceptance of an EAGLE-3 draft on Solar's own exact greedy generations (NPU engine outputs).

For each eval prompt dir (prompt.json + gen.txt "TOKENS [...]"), teacher features come from extract.py's
layer-streaming forward. A chain drafted from position t (target has produced x_{t+1}) is accepted while the
draft's greedy token equals Solar's next greedy token; while it matches, feeding ground truth == feeding the
draft's own token, so one TTT-style pass gives every position's accepted length.
tau = 1 + mean accepted drafts per verify step (the target always contributes one token).

usage: CUDA_VISIBLE_DEVICES=1 python3 eval.py draft.pt evalgen/e* [--k 6]
"""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract  # noqa: E402
from train import Draft, attend, H, BF, DEV  # noqa: E402
from model import Weights, Rotary  # noqa: E402


def main():
    args = sys.argv[1:]
    k = 6
    if "--k" in args:
        i = args.index("--k"); k = int(args[i + 1]); del args[i:i + 2]
    ck = torch.load(args[0])
    dirs = args[1:]
    W = Weights(); rot = Rotary(W.cfg.raw)
    embed = W.get("model.embed_tokens.weight")
    norm = W.get("model.norm.weight").to(DEV)
    model = Draft().to(DEV)
    model.load_state_dict({kk: v for kk, v in ck["state"].items()})
    model = model.to(BF).eval()
    d2t = ck["d2t"].to(DEV)
    tot_acc, tot_pos, hist = 0, 0, [0] * (k + 1)
    for d in dirs:
        ids = json.load(open(f"{d}/prompt.json"))["ids"]
        gen = json.loads(open(f"{d}/gen.txt").read().split("TOKENS", 1)[1])
        seq = torch.tensor(ids + gen, dtype=torch.int32)
        aux, fin, _ = extract.forward_chunk(W, rot, embed, norm, [(seq, None)])
        x = seq.long()
        L = len(x) - 1
        emb = embed[x[1:]].to(DEV)[None]
        cos, sin = rot(torch.arange(L + k + 2))
        cos, sin = cos.to(DEV)[None, None], sin.to(DEV)[None, None]
        kmask = torch.ones(1, L, dtype=torch.bool, device=DEV)
        ok = torch.zeros(k, L, dtype=torch.bool, device=DEV)
        with torch.no_grad():
            hid = model.fc(aux[:L].to(DEV)[None])
            ks, vs = [], []
            for j in range(k):
                e = emb if j == 0 else torch.cat([emb[:, j:], torch.zeros_like(emb[:, :j])], 1)
                q, kk, vv = model.qkv(e, hid, cos[:, :, j:j + L], sin[:, :, j:j + L])
                ks.append(kk); vs.append(vv)
                h, lg = model.out(hid, attend(q, ks, vs, kmask))
                pred = d2t[lg[0].argmax(-1)]                       # position t, step j -> x_{t+j+2}
                T = L - j - 1
                ok[j, :T] = pred[:T] == x[j + 2:j + 2 + T].to(DEV)
                hid = h
        # chains start where the target has just produced a generated token: t+1 >= len(ids)
        starts = range(len(ids) - 1, len(x) - 1 - k)
        accs = []
        for t in starts:
            a = 0
            while a < k and ok[a, t]: a += 1
            accs.append(a); hist[a] += 1
        tot_acc += sum(accs); tot_pos += len(accs)
        print(f"{d}: {len(accs)} chains, tau {1 + sum(accs) / max(1, len(accs)):.2f}, step-1 acc {ok[0, len(ids) - 1:len(x) - 2].float().mean():.3f}", flush=True)
    print(f"ALL: tau(k={k}) = {1 + tot_acc / tot_pos:.3f}; accepted-length histogram {hist}")


if __name__ == "__main__":
    main()
