"""Reasoning-style data in Solar's generation format:
  prompt (chat template, add_generation_prompt) + <|think|>{reasoning}<|end|><|begin|>assistant<|content|>{answer}<|end|>
Loss mask = every generated token. Mixed with prep.py's chat data (data_v1.pt).
usage: python3 prep2.py out.pt [n_nemotron] [n_mot] [max_len]
"""
import glob, json, random, re, sys
import pandas as pd
import torch
from transformers import AutoTokenizer

THINK = re.compile(r"^\s*<think>(.*?)</think>\s*(.*)$", re.S)


def build(tk, user, reasoning, answer, max_len):
    p = tk.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True)
    p = list(p["input_ids"] if hasattr(p, "keys") else p)
    g = tk.encode(f"<|think|>{reasoning.strip()}<|end|><|begin|>assistant<|content|>{answer.strip()}<|end|>", add_special_tokens=False)
    ids = (p + g)[:max_len]
    m = ([0] * len(p) + [1] * len(g))[:max_len]
    return torch.tensor(ids, dtype=torch.int32), torch.tensor(m, dtype=torch.bool)


def main():
    out = sys.argv[1]
    n_nem = int(sys.argv[2]) if len(sys.argv) > 2 else 8574
    n_mot = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    max_len = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    d = sorted(glob.glob('/home/gspark/.cache/huggingface/hub/models--upstage--Solar-Open-100B/snapshots/*'))[-1]
    tk = AutoTokenizer.from_pretrained(d)
    samples = []
    rows = [json.loads(l) for l in open("raw/nemotron_chat.jsonl")]
    rows = [r for r in rows if r["reasoning"] == "on"]
    random.Random(0).shuffle(rows)
    for r in rows[:n_nem]:
        msgs = r["input"] if isinstance(r["input"], list) else eval(r["input"])
        if len(msgs) != 1 or msgs[0]["role"] != "user": continue
        mt = THINK.match(r["output"])
        if not mt or len(mt.group(1).strip()) < 200: continue
        samples.append(build(tk, msgs[0]["content"], mt.group(1), mt.group(2), max_len))
    n1 = len(samples)
    mot = pd.read_parquet("raw/mot_all0.parquet").sample(frac=1.0, random_state=0)
    for msgs in mot["messages"][:n_mot]:
        if len(msgs) < 2 or msgs[0]["role"] != "user": continue
        mt = THINK.match(msgs[1]["content"])
        if not mt: continue
        samples.append(build(tk, msgs[0]["content"], mt.group(1), mt.group(2), max_len))
    samples += torch.load("data_v1.pt")
    random.Random(1).shuffle(samples)
    ntok = sum(len(s) for s, _ in samples); nloss = sum(int(m.sum()) for _, m in samples)
    torch.save(samples, out)
    print(f"nemotron {n1}, mot {len(samples) - n1 - len(torch.load('data_v1.pt'))}, total {len(samples)} samples, {ntok} tokens, {nloss} loss tokens -> {out}")


if __name__ == "__main__":
    main()
