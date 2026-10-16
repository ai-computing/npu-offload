"""Compile all bucket graphs once (CPU only) and persist EDFs to the artifact cache."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import runtime as R
from model import Weights
import artifact_cache as AC
W = Weights(); c = W.cfg
npu = R.Npu(c, "rngd:0")
npu.dev = None  # don't load to device
t00 = time.time()
for T in R.T_BUCKETS:
    for kind, Ks in (("qkv", [0]), ("post", [0]), ("exp", list(R.K_BUCKETS))):
        for K in Ks:
            # reuse Npu._compile's key + compile_fn without device load
            orig_to = R.Npu._compile
            import types
            key = (kind, T, K)
            t0 = time.time()
            try:
                m = npu._compile(kind, T, K)
            except Exception as e:
                print("ERR", key, type(e).__name__, str(e)[:200], flush=True)
            print(f"done {key} {time.time()-t0:.0f}s", flush=True)
print(f"ALL DONE {time.time()-t00:.0f}s")
