"""v3 data: Glaive reasoning-v1 (general-domain R1 traces, Apache-2.0) in Solar's format + all of data_v2.
usage: python3 prep3.py out.pt [n_glaive] [max_len]"""
import glob, os, random, sys
import pandas as pd
import torch
from transformers import AutoTokenizer
from prep2 import THINK, build


def main():
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 45000
    max_len = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    d = sorted(glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--upstage--Solar-Open-100B/snapshots/*')))[-1]
    tk = AutoTokenizer.from_pretrained(d)
    g = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("raw/glaive_*.parquet"))]).sample(frac=1.0, random_state=0)
    samples = []
    for r in g.itertuples():
        if len(samples) >= n: break
        mt = THINK.match(r.response)
        if not mt or len(mt.group(1).strip()) < 200: continue
        samples.append(build(tk, r.prompt, mt.group(1), mt.group(2), max_len))
    ng = len(samples)
    samples += torch.load("data_v2.pt")
    random.Random(2).shuffle(samples)
    ntok = sum(len(s) for s, _ in samples)
    torch.save(samples, out)
    print(f"glaive {ng}, total {len(samples)} samples, {ntok} tokens -> {out}")


if __name__ == "__main__":
    main()
