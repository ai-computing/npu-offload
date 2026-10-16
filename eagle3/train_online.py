"""Online EAGLE-3 training: extract Solar features for a chunk (extract.forward_chunk, layer streaming) and train
the draft on it right away (PASSES times), so the 32 KB/token features are never stored.

usage: TORCH_DEVICE_BACKEND_AUTOLOAD=0 CUDA_VISIBLE_DEVICES=1 python3 train_online.py data.pt init.pt out.pt
env: CHUNK (tokens, 98304), PASSES (2), TOK_BUDGET (6144), LR (1.5e-4), TTT (5), GPU_FRAC (0.72)
"""
import math, os, random, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract  # noqa: E402
from train import Draft, batches, make_batch, step_loss, BF, DEV  # noqa: E402
from model import Weights, Rotary  # noqa: E402


def main():
    data, init, out = sys.argv[1:4]
    chunk = int(os.environ.get("CHUNK", 98304)); passes = int(os.environ.get("PASSES", 2))
    budget = int(os.environ.get("TOK_BUDGET", 6144)); ttt = int(os.environ.get("TTT", 5))
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", "0.72")))
    W = Weights(); rot = Rotary(W.cfg.raw)
    embed = W.get("model.embed_tokens.weight")
    norm = W.get("model.norm.weight").to(DEV)
    ck = torch.load(init)
    d2t = ck["d2t"].to(DEV)
    lm_d = W.get("lm_head.weight")[ck["d2t"]].to(DEV)
    model = Draft().to(DEV)
    model.load_state_dict(ck["state"])
    model = model.float()
    opt = torch.optim.AdamW(model.parameters(), lr=float(os.environ.get("LR", "1.5e-4")), betas=(0.9, 0.95), weight_decay=0.0)
    samples = torch.load(data)
    groups, cur, n = [], [], 0
    for s in samples:
        if n + len(s[0]) > chunk and cur: groups.append(cur); cur, n = [], 0
        cur.append(s); n += len(s[0])
    if cur: groups.append(cur)
    est = int(len(groups) * passes * 1.15 * chunk / budget)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 100) * max(0.02, 0.5 * (1 + math.cos(math.pi * min(1.0, s / est)))))
    rng = random.Random(0)
    lm_full = None
    step, t0 = 0, time.time()
    for gi, g in enumerate(groups):
        te = time.time()
        with torch.no_grad():
            aux, fin, _ = extract.forward_chunk(W, rot, embed, norm, g)
        fin = fin.cpu()
        seqs, o = [], 0
        for ids, m in g:
            k = len(ids); seqs.append((ids.long(), m, aux[o:o + k], fin[o:o + k])); o += k
        del aux
        torch.cuda.empty_cache()
        # draft-vocab coverage of the target argmax on this chunk (loss positions)
        if gi % 10 == 0:
            if lm_full is None: lm_full = W.get("lm_head.weight")
            f = torch.cat([s[3][s[1]] for s in seqs[:16]]).to(DEV)
            am = (f @ lm_full.to(DEV).t()).argmax(-1)
            cov = torch.isin(am, d2t).float().mean().item()
            del f, am; torch.cuda.empty_cache()
        tt = time.time()
        accs_sum, nb = None, 0
        for _ in range(passes):
            for bi in batches(seqs, budget, rng):
                b = make_batch(seqs, bi, embed, lm_d, rot)
                with torch.autocast("cuda", dtype=BF):
                    loss, accs = step_loss(model, *b, ttt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step(); sched.step(); step += 1
                accs_sum = accs if accs_sum is None else [a + c for a, c in zip(accs_sum, accs)]; nb += 1
        del seqs; torch.cuda.empty_cache()
        print(f"[{gi + 1}/{len(groups)}] step {step} loss {loss.item():.3f} acc {[round(a / nb, 3) for a in accs_sum]} "
              f"cov {cov:.3f} lr {sched.get_last_lr()[0]:.2e} extract {tt - te:.0f}s train {time.time() - tt:.0f}s "
              f"total {(time.time() - t0) / 3600:.2f}h", flush=True)
        if (gi + 1) % 20 == 0 or gi + 1 == len(groups):
            torch.save(dict(state={k: v.to(BF) for k, v in model.state_dict().items()}, d2t=ck["d2t"], cfg=ck.get("cfg")), out + ".tmp")
            os.replace(out + ".tmp", out)


if __name__ == "__main__":
    main()
