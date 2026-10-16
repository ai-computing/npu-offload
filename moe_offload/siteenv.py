"""Site-specific settings (paths, gateway URLs) come from the environment, optionally seeded from an untracked
`cluster.local.env` (KEY=VALUE lines; see cluster.example.env). Already-set variables win over the file.
Lookup: $MOE_OFFLOAD_ENV, ./cluster.local.env, <repo root>/cluster.local.env."""
import os

_loaded = False


def load():
    global _loaded
    if _loaded: return
    _loaded = True
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (os.environ.get("MOE_OFFLOAD_ENV"), "cluster.local.env", os.path.join(root, "cluster.local.env")):
        if p and os.path.isfile(p):
            for line in open(p):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line: continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


def require(name, what):
    load()
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"[moe_offload] {name} is not set ({what}). Set it in the environment or in "
                         f"cluster.local.env (copy cluster.example.env).")
    return v
