"""EDF artifact cache (Phase 5) — persist a compiled CompileModule so restart skips the ~30min compile.

torch.save on a CompileModule fails (native `ir.Scatter` isn't picklable), but the EDF exposes a native
`serialize()`/`Edf.deserialize()`, and the `module_weight` (ExportedProgramWeight, pure torch) pickles.
So an artifact = { <key>.edf : edf.serialize() bytes,  <key>.w : torch.save(module_weight) }. On load
we deserialize the EDF, reload the weight, and rebuild `CompileModule(EdfModule(edf), weight)` — no
recompile. Verified in a fresh process (probe_artifact_cache.py): identical output, load in seconds.

Content-addressed by (model, dims, bucket, tactic) so each unique config compiles once, ever.
furiosa imports are lazy so this module stays importable on NPU-free hosts (serving_app --mock).
"""
import os, time, hashlib
import torch


def cache_key(**kw):
    return hashlib.blake2b(repr(sorted(kw.items())).encode(), digest_size=12).hexdigest()


def path_for(cache_dir, key):
    return os.path.join(cache_dir, key)


def exists(path):
    return os.path.exists(path + ".edf") and os.path.exists(path + ".w")


def save(module, path):
    """Persist a compiled CompileModule as (<path>.edf blob, <path>.w weight). Returns bytes written."""
    blob = module.edf.serialize()                       # native EDF serialization (binaries+scatter+shapes)
    tmp = path + ".edf.tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, path + ".edf")
    torch.save(module.module_weight, path + ".w.tmp")   # ExportedProgramWeight (flatten/unflatten specs)
    os.replace(path + ".w.tmp", path + ".w")
    return os.path.getsize(path + ".edf") + os.path.getsize(path + ".w")


def load(path, device=None):
    """Rebuild a compiled CompileModule from a saved artifact. No recompile."""
    from furiosa.native_torch import ir
    from furiosa.torch.custom_ops import CompileModule
    from furiosa.torch.custom_ops.edf import EdfModule
    with open(path + ".edf", "rb") as f:
        edf = ir.Edf.deserialize(f.read())
    weight = torch.load(path + ".w", weights_only=False)
    m = CompileModule(EdfModule(edf), weight)
    if device is not None:
        m.to(device)
    return m


def get_or_compile(cache_dir, key, compile_fn, device=None):
    """Return (module, mode, seconds). mode ∈ {'load','compile'}. compile_fn() -> CompileModule."""
    os.makedirs(cache_dir, exist_ok=True)
    path = path_for(cache_dir, key)
    if exists(path):
        t0 = time.time(); m = load(path, device); return m, "load", time.time() - t0
    t0 = time.time(); m = compile_fn()
    try:
        save(m, path)
    except Exception as e:                              # never fail the request path over caching
        print(f"[artifact_cache] WARN: could not persist EDF ({e}); cold-start cache disabled for {key}")
    return m, "compile", time.time() - t0


if __name__ == "__main__":
    k1 = cache_key(model="llama", H=4096, bucket=64, tactic="Decode")
    k2 = cache_key(bucket=64, tactic="Decode", H=4096, model="llama")
    k3 = cache_key(model="llama", H=4096, bucket=128, tactic="Decode")
    assert k1 == k2 and k1 != k3
    print(f"artifact_cache key self-test: PASS ✅ (stable={k1}, differs on bucket)")
