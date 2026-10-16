"""Tokenize chat data with Solar's chat template -> list of (ids, loss_mask) for EAGLE-3 training.

loss_mask marks assistant-content tokens (and the closing <|end|>): the positions whose next-token
predictions the draft head is trained on.
usage: python3 prep.py out.pt [n_ultrachat] [n_koalpaca] [max_len]
"""
import glob, random, sys
import pandas as pd
import torch
from transformers import AutoTokenizer

BEGIN, END, CONTENT = 20, 21, 23


def assistant_mask(ids, tk):
    asst = tk.encode("assistant", add_special_tokens=False)
    m = [0] * len(ids)
    i, n = 0, len(ids)
    while i < n:
        if ids[i] == BEGIN and ids[i + 1:i + 1 + len(asst)] == asst:
            j = i + 1 + len(asst)
            while j < n and ids[j] != END:
                m[j] = 1; j += 1
            if j < n: m[j] = 1
            i = j
        i += 1
    return m


def main():
    out = sys.argv[1]
    n_uc = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    n_ko = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    max_len = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    d = sorted(glob.glob('/home/user/.cache/huggingface/hub/models--upstage--Solar-Open-100B/snapshots/*'))[-1]
    tk = AutoTokenizer.from_pretrained(d)
    rng = random.Random(0)
    uc = pd.read_parquet("raw/ultrachat_train_sft0.parquet")
    ko = pd.read_parquet("raw/koalpaca.parquet")
    convs = [list(m) for m in uc.sample(n=n_uc, random_state=0)["messages"]]
    convs += [[{"role": "user", "content": r.instruction}, {"role": "assistant", "content": r.output}]
              for r in ko.sample(n=n_ko, random_state=0).itertuples()]
    rng.shuffle(convs)
    samples, ntok, nloss = [], 0, 0
    for c in convs:
        c = [{"role": x["role"], "content": x["content"]} for x in c]
        ids = tk.apply_chat_template(c, tokenize=True, add_generation_prompt=False)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)[:max_len]
        m = assistant_mask(ids, tk)
        if sum(m) < 8: continue
        samples.append((torch.tensor(ids, dtype=torch.int32), torch.tensor(m, dtype=torch.bool)))
        ntok += len(ids); nloss += sum(m)
    torch.save(samples, out)
    print(f"{len(samples)} samples, {ntok} tokens, {nloss} loss tokens -> {out}")


if __name__ == "__main__":
    main()
