"""Tiered expert placement: which experts live in which accelerator's HBM, the rest on the host.

A placement is `[layer][tier] -> [expert ids in slot order]` (the engine's placement.json). Tiers are the
accelerators in engine order with a fixed number of expert slots per layer (set by the compiled kernels).

Policies (per layer, experts ranked by profile count, rank 0 = hottest):
  "top"      tier 0 takes ranks [0, s0), tier 1 the next s1, ...
  "mix:<h>"  tier 1 takes ranks [h, h+s1); tier 0 takes ranks [0, h) and [h+s1, h+s1+s0-h). Keeps a few of the
             hottest experts on tier 0 (which also runs attention) and gives the bulk of hot traffic to tier 1,
             balancing the per-layer critical path (Solar on 2x RNGD: mix:4 was fastest).
"""
import json


def ranks(row):
    return sorted(range(len(row)), key=lambda e: (-row[e], e))


def plan(counts, slots, policy="top"):
    out = []
    for row in counts:
        r = ranks(row)
        if policy == "top":
            tiers, o = [], 0
            for s in slots: tiers.append(r[o:o + s]); o += s
        elif policy.startswith("mix:"):
            h = int(policy.split(":")[1])
            assert len(slots) == 2 and 0 <= h <= slots[0]
            s0, s1 = slots
            tiers = [r[:h] + r[h + s1:h + s1 + s0 - h], r[h:h + s1]]
        else:
            raise ValueError(f"unknown policy {policy}")
        out.append(tiers)
    return out


def hit_rates(counts, placement):
    """fraction of routed traffic served by each tier (and the host) under `counts`."""
    tiers = len(placement[0])
    tot, per = 0.0, [0.0] * (tiers + 1)
    for row, pl in zip(counts, placement):
        s = sum(row); tot += s
        placed = set()
        for t, es in enumerate(pl):
            per[t] += sum(row[e] for e in es); placed.update(es)
        per[-1] += s - sum(row[e] for e in placed)
    return [p / tot for p in per]


def save(placement, path):
    json.dump(placement, open(path, "w"))


def load(path):
    return json.load(open(path))


def main():
    import argparse
    from .spec import ModelSpec
    from .profile import counts_from_routes
    ap = argparse.ArgumentParser(description="plan a tiered expert placement")
    ap.add_argument("--model", required=True, help="HF model dir (config.json)")
    ap.add_argument("--profile", required=True, help="routes.pt profile")
    ap.add_argument("--slots", default="24,32", help="expert slots per layer per accelerator")
    ap.add_argument("--policy", default="mix:4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = ModelSpec.from_hf_config(a.model)
    c = counts_from_routes(a.profile, m.layers, m.experts)
    p = plan(c, [int(s) for s in a.slots.split(",")], a.policy)
    save(p, a.out)
    print("tier hit fractions (profile):", [round(x, 4) for x in hit_rates(c, p)])


if __name__ == "__main__":
    main()


def class_priors(placement, by_class):
    """expected accelerator hit rate per request class: the gateway's affinity prior for a worker."""
    return {cls: round(1.0 - hit_rates(c, placement)[-1], 4) for cls, c in by_class.items()}
