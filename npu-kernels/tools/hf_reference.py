#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""HF reference for the generated RNGD layer engine: greedy tokens + top logits per step, and router statistics.

  python hf_reference.py <model_dir> <out_dir> --prompts prompts.json --new 32 [--calib calib.json]

prompts.json: {"name": "text", ...} (chat-templated as one user turn). Writes, per prompt, <out_dir>/<name>/prompt.json
(ids), ref.json (greedy tokens, top-20 logits per step). With --calib, also <out_dir>/expert_counts.json: per layer, how
often each expert was in the top-k over prefill + greedy decode of the calibration prompts (for the hot-expert table).
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def router_hooks(model, counts):
    """count top-k expert choices per layer from the router (mlp.gate) outputs."""
    k = model.config.num_experts_per_tok
    hooks = []
    for li, layer in enumerate(model.model.layers):
        gate = layer.mlp.gate

        def hook(mod, inp, out, li=li):
            t = out
            if isinstance(out, tuple):          # newer routers return (logits, weights, indices)
                ints = [o for o in out if torch.is_tensor(o) and not o.is_floating_point()]
                if ints:
                    for e in ints[0].reshape(-1).tolist():
                        counts[li][e] += 1
                    return
                t = out[0]
            for e in t.float().topk(k, dim=-1).indices.reshape(-1).tolist():
                counts[li][e] += 1
        hooks.append(gate.register_forward_hook(hook))
    return hooks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--calib")
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.bfloat16)
    model.eval()
    cfg = model.config

    def ids_of(text):
        msgs = [{"role": "user", "content": text}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return tok(text, add_special_tokens=False).input_ids

    def run(ids, n):
        x = torch.tensor([ids])
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=n, do_sample=False, output_logits=True,
                                 return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
        toks = out.sequences[0, len(ids):].tolist()
        tops = []
        for lg in out.logits:
            v, i = lg[0].float().topk(20)
            tops.append({"ids": i.tolist(), "logits": v.tolist()})
        return toks, tops

    os.makedirs(a.out_dir, exist_ok=True)
    if a.calib:
        counts = [[0] * cfg.num_experts for _ in range(cfg.num_hidden_layers)]
        hooks = router_hooks(model, counts)
        for name, text in json.load(open(a.calib)).items():
            ids = ids_of(text)
            run(ids, a.new)
            print("calib", name, len(ids), flush=True)
        for h in hooks:
            h.remove()
        json.dump({"counts": counts, "prompts": list(json.load(open(a.calib)))}, open(os.path.join(a.out_dir, "expert_counts.json"), "w"))
    for name, text in json.load(open(a.prompts)).items():
        ids = ids_of(text)
        toks, tops = run(ids, a.new)
        d = os.path.join(a.out_dir, name)
        os.makedirs(d, exist_ok=True)
        json.dump({"ids": ids, "text": text}, open(os.path.join(d, "prompt.json"), "w"))
        json.dump({"tokens": toks, "top": tops, "text": tok.decode(toks)}, open(os.path.join(d, "ref.json"), "w"))
        print("ref", name, len(ids), toks[:8], repr(tok.decode(toks))[:120], flush=True)


if __name__ == "__main__":
    main()
