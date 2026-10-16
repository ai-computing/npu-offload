"""Expert activation profiles: counts[layer][expert] of routed (token, expert) pairs."""
import json


def counts_from_routes(path, layers, experts):
    """p0/expert_profile.py format: {"recs": [{"prefill": [L x (T,k)], "decode": ...}, ...]} (torch.save)."""
    import torch
    d = torch.load(path, weights_only=False)
    c = torch.zeros(layers, experts)
    for r in d["recs"]:
        for key in ("prefill", "decode"):
            x = r.get(key)
            if x is None: continue
            for l in range(layers):
                c[l] += torch.bincount(x[l].flatten().long(), minlength=experts).float()
    return c.tolist()


def counts_from_trace(path, layers, experts):
    """JSON lines {"layer": l, "experts": [...]} as written by an engine trace."""
    c = [[0.0] * experts for _ in range(layers)]
    for line in open(path):
        r = json.loads(line)
        for e in r["experts"]: c[r["layer"]][e] += 1
    return c


def merge(*profiles, weights=None):
    weights = weights or [1.0] * len(profiles)
    L, E = len(profiles[0]), len(profiles[0][0])
    return [[sum(w * p[l][e] for p, w in zip(profiles, weights)) for e in range(E)] for l in range(L)]


DOMAIN_CLASS = {"ko_chat": "ko", "ko_tech": "ko", "ko_write": "ko", "ko_code": "ko",
                "code_py": "code", "code_cpp": "code"}


def counts_by_class(path, layers, experts, domain_class=None):
    """per request class ('ko' / 'code' / 'en') counts from a routes.pt profile whose records carry a 'domain'."""
    import torch
    dc = domain_class or DOMAIN_CLASS
    d = torch.load(path, weights_only=False)
    out = {}
    for r in d["recs"]:
        cls = dc.get(r.get("domain", ""), "en")
        c = out.setdefault(cls, torch.zeros(layers, experts))
        for key in ("prefill", "decode"):
            x = r.get(key)
            if x is None: continue
            for l in range(layers):
                c[l] += torch.bincount(x[l].flatten().long(), minlength=experts).float()
    return {k: v.tolist() for k, v in out.items()}
