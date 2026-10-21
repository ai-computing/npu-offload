#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Tie-aware token comparison of a layer-engine run against the HF reference.

  python compare_ref.py ref.json engine.json

Step i matches when the engine's argmax equals the HF token, or when the two tokens are tied in the HF logits at bf16
precision (|l_hf - l_engine_token| <= one bf16 ulp of the HF max logit; HF computes the LM head in bf16). In a
teacher-forced run (FORCE=ref.json) every step is compared; in a free run, steps after the first divergence are not
comparable and are counted separately.
"""
import json
import math
import sys


def ulp_bf16(v):
    if v == 0:
        return 2.0 ** -133
    return 2.0 ** (math.floor(math.log2(abs(v))) - 7)


def compare(ref, eng):
    out = {"steps": 0, "exact": 0, "tie": 0, "mismatch": 0, "after_divergence": 0, "details": []}
    diverged = False
    for i, (th, te) in enumerate(zip(ref["tokens"], eng["tokens"])):
        out["steps"] += 1
        if diverged and not eng.get("forced"):
            out["after_divergence"] += 1
            continue
        if th == te:
            out["exact"] += 1
            continue
        top = dict(zip(ref["top"][i]["ids"], ref["top"][i]["logits"]))
        lh = top[th]
        le = top.get(te)
        gap = None if le is None else lh - le
        tie = le is not None and gap <= ulp_bf16(lh) + 1e-6
        out["tie" if tie else "mismatch"] += 1
        out["details"].append({"step": i, "hf": th, "engine": te, "hf_logit": lh, "hf_logit_engine_token": le,
                               "gap": gap, "tie": tie})
        diverged = True
    out["match_tie_aware"] = out["mismatch"] == 0
    return out


if __name__ == "__main__":
    r = compare(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])))
    print(json.dumps(r, indent=1))
    sys.exit(0 if r["match_tie_aware"] else 1)
