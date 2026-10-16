"""Export host-side inputs for the Rust solar_engine: yarn RoPE cos/sin table (bf16, [S,128]) and prompt ids."""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Weights, Rotary, BF
from transformers import AutoTokenizer
out = os.path.expanduser("~/workspace/npu_offload/track2_opt/solar_engine/data")
os.makedirs(out, exist_ok=True)
W = Weights(); rot = Rotary(W.cfg.raw)
S = 8192
cos, sin = rot(torch.arange(S))
cos.contiguous().view(torch.int16).numpy().tofile(f"{out}/rope_cos.bin")
sin.contiguous().view(torch.int16).numpy().tofile(f"{out}/rope_sin.bin")
tk = AutoTokenizer.from_pretrained(W.dir)
prompt = sys.argv[1] if len(sys.argv) > 1 else "서울에서 주말에 가볼 만한 곳을 다섯 군데 추천하고 각각 이유를 설명해줘."
ids = tk.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
json.dump({"ids": ids, "model_dir": W.dir}, open(f"{out}/prompt.json", "w"))
print("exported", len(ids), "prompt ids; model dir", W.dir)
