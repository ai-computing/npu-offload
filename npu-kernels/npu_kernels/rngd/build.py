"""Generate (if needed) and build the expert server binary for a tiling plan."""
import hashlib
import os
import subprocess

from .generate import _TPL, crate_name, generate


def cache_root():
    """crates and targets live under $NPU_KERNELS_CACHE (default ~/.cache/npu_kernels)."""
    return os.path.expanduser(os.environ.get("NPU_KERNELS_CACHE", "~/.cache/npu_kernels"))


def build_server(plan, emulate: bool) -> str:
    """generate the crate for `plan` (regenerated when the template changes) and build its `server` binary:
    `cargo furiosa-opt build` for RNGD, plain `cargo build` for the furiosa-opt CPU emulator. Returns its path."""
    name = crate_name(plan)
    crate = os.path.join(cache_root(), "crates", name)
    digest = hashlib.blake2b(digest_size=8)
    for root, _, files in sorted(os.walk(_TPL)):
        for f in sorted(files):
            digest.update(f.encode()); digest.update(open(os.path.join(root, f), "rb").read())
    stamp = os.path.join(crate, ".template")
    if not os.path.exists(stamp) or open(stamp).read() != digest.hexdigest():
        generate(plan, crate, crate=name)
        with open(stamp, "w") as f:
            f.write(digest.hexdigest())
    target = os.path.join(cache_root(), "target", "cpu" if emulate else "npu")
    exe = os.path.join(target, "release", "server")
    env = dict(os.environ, CARGO_TARGET_DIR=target)
    cmd = ["cargo", "build", "--release", "--bin", "server"] if emulate else \
        ["cargo", "furiosa-opt", "build", "--release", "--bin", "server"]
    r = subprocess.run(cmd, cwd=crate, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"npu_kernels.rngd: building {name} failed:\n{r.stderr[-4000:]}")
    dst = os.path.join(target, f"server_{name}")       # every plan builds `server`: keep a per-plan copy
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(exe):
        tmp = dst + ".tmp"
        with open(exe, "rb") as a, open(tmp, "wb") as b:
            b.write(a.read())
        os.chmod(tmp, 0o755)
        os.replace(tmp, dst)
    return dst
