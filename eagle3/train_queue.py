"""Consumer: train the EAGLE-3 draft on feature chunks as they appear in a queue dir; deletes each after use.

usage: TORCH_DEVICE_BACKEND_AUTOLOAD=0 CUDA_VISIBLE_DEVICES=1 python3 train_queue.py qdir n_shards total_chunks init.pt out.pt
env: PASSES (1), TOK_BUDGET (6144), LR (1.2e-4), TTT (5), GPU_FRAC (0.45), SAVE_EVERY (25)
"""
import glob, math, os, random, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Draft, batches, make_batch, step_loss, BF, DEV  # noqa: E402
from model import Weights, Rotary  # noqa: E402


def main():
    q, nshard, total, init, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
    passes = int(os.environ.get("PASSES", 1)); budget = int(os.environ.get("TOK_BUDGET", 6144))
    ttt = int(os.environ.get("TTT", 5)); save_every = int(os.environ.get("SAVE_EVERY", 25))
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", "0.45")))
    W = Weights(); rot = Rotary(W.cfg.raw)
    embed = W.get("model.embed_tokens.weight")
    ck = torch.load(init)
    lm_d = W.get("lm_head.weight")[ck["d2t"]].to(DEV)
    model = Draft().to(DEV); model.load_state_dict(ck["state"]); model = model.float()
    opt = torch.optim.AdamW(model.parameters(), lr=float(os.environ.get("LR", "1.2e-4")), betas=(0.9, 0.95), weight_decay=0.0)
    est = int(total * passes * 1.15 * 98304 / budget)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 100) * max(0.03, 0.5 * (1 + math.cos(math.pi * min(1.0, s / est)))))
    rng = random.Random(0)
    step, done, t0 = 0, 0, time.time()
    while True:
        files = sorted(glob.glob(os.path.join(q, "s*_*.pt")), key=os.path.getmtime)
        if not files:
            if len(glob.glob(os.path.join(q, "s*.done"))) >= nshard: break
            time.sleep(1); continue
        f = files[0]
        c = torch.load(f); os.remove(f)
        seqs, o = [], 0
        for ids, m in zip(c["ids"], c["mask"]):
            k = len(ids); seqs.append((ids.long(), m, c["aux"][o:o + k], c["fin"][o:o + k])); o += k
        del c
        tt = time.time(); acc_sum, nb = None, 0
        for _ in range(passes):
            for bi in batches(seqs, budget, rng):
                b = make_batch(seqs, bi, embed, lm_d, rot)
                with torch.autocast("cuda", dtype=BF):
                    loss, accs = step_loss(model, *b, ttt)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step(); sched.step(); step += 1
                acc_sum = accs if acc_sum is None else [a + b2 for a, b2 in zip(acc_sum, accs)]; nb += 1
        del seqs; torch.cuda.empty_cache()
        done += 1
        print(f"[{done}/{total}] {os.path.basename(f)} step {step} loss {loss.item():.3f} acc {[round(a / nb, 3) for a in acc_sum]} "
              f"lr {sched.get_last_lr()[0]:.2e} train {time.time() - tt:.0f}s total {(time.time() - t0) / 3600:.2f}h", flush=True)
        if done % save_every == 0:
            torch.save(dict(state={k: v.to(BF) for k, v in model.state_dict().items()}, d2t=ck["d2t"], cfg=ck.get("cfg")), out + ".tmp")
            os.replace(out + ".tmp", out)
    torch.save(dict(state={k: v.to(BF) for k, v in model.state_dict().items()}, d2t=ck["d2t"], cfg=ck.get("cfg")), out)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
