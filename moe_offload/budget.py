"""Resource budget -> expert slots per layer per accelerator (the 'resource declaration' half of placement).

Each tier declares its HBM capacity and what else it hosts (dense layers, KV cache, lm_head, runtime reserve);
what remains, divided over layers, bounds the expert slots. Compiled kernels fix the slot count, so the planner
checks the requested slots against this bound rather than choosing freely.
"""
from dataclasses import dataclass, field

GiB = 1 << 30


@dataclass
class Tier:
    name: str
    hbm_bytes: int
    hosts_dense: bool = False
    hosts_kv: bool = False
    hosts_head: bool = False
    reserve_bytes: int = 1 * GiB          # runtime scratch, program images, alignment
    slots: int = 0                        # compiled expert slots per layer (0 = unknown)
    extra: dict = field(default_factory=dict)


def fixed_bytes(spec, t, ctx):
    b = t.reserve_bytes
    if t.hosts_dense: b += spec.layers * spec.dense_layer_bytes
    if t.hosts_kv: b += spec.kv_bytes(ctx)
    if t.hosts_head: b += spec.head_bytes
    return b + sum(t.extra.values())


def max_slots(spec, t, ctx=8192):
    free = t.hbm_bytes - fixed_bytes(spec, t, ctx)
    return max(0, free // (spec.layers * spec.expert_bytes))


def report(spec, tiers, ctx=8192):
    rows = []
    for t in tiers:
        cap = max_slots(spec, t, ctx)
        used = fixed_bytes(spec, t, ctx) + t.slots * spec.layers * spec.expert_bytes
        rows.append(dict(tier=t.name, slots=t.slots, max_slots=cap, fits=t.slots <= cap,
                         used_gib=round(used / GiB, 2), hbm_gib=round(t.hbm_bytes / GiB, 2)))
    host = spec.total_expert_bytes - sum(t.slots for t in tiers) * spec.layers * spec.expert_bytes
    return rows, host


def rngd_solar_2chip(lm_on_b=True):
    """The current Solar-Open-100B engine layout on 2x RNGD (47.5 GiB HBM each)."""
    hbm = int(47.5 * GiB)
    return [Tier("rngd0", hbm, hosts_dense=True, hosts_kv=True, hosts_head=not lm_on_b, slots=24, reserve_bytes=GiB // 2),
            Tier("rngd1", hbm, hosts_head=lm_on_b, slots=32, reserve_bytes=GiB // 2)]
