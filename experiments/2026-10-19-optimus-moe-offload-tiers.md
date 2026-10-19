# OptimusPrime MoE offload: RNGD furiosa-opt tier, native dispatcher, tier matrix (Qwen3-30B-A3B)

- **Date**: 2026-10-19
- **Task**: OptimusPrime MoE offload (generalized RNGD/GPU/CPU tiers)
- **Host / device**: this box, 2x RNGD (fw 2026.4.0, furiosa-opt 0.9), A100 80GB (GPU0), 2x Xeon Silver 4410Y; torch 2.10.0+cu128, transformers 5.1.0
- **Commit**: aicomp `feat/optimus-moe-offload` (local; 77ed99d rngd-opt, 1645390 native dispatcher, 39e6739 compile option, 4ea3e0a devices)

## Question

How fast is the generalized OptimusPrime offload path (model-agnostic, per-shape generated RNGD kernels) on
Qwen3-30B-A3B across GPU / RNGD / CPU tier mixes, where does a layer's time go, and how far is it from the
Solar-Open PoC engine?

## Setup

- `rngd-opt` backend: furiosa-opt kernel generated per expert shape (kernels/rngd_opt, same tiling rule as the PoC
  engine), Rust expert server per chip, per-layer expert table resident in HBM (32 or 64 slots), pipe protocol.
- Native host dispatcher (csrc/dispatch.c): blocking tiers (cpu, rngd-opt) run on persistent C threads, no GIL.
- Profile prompts: 2 (EN MoE/softmax, KO capital). Prompt "Explain mixture of experts in two sentences.", greedy,
  KV cache, 64 tokens (incl. prefill), warm-up 4 tokens.

## Command

```bash
cd ~/workspace/aicomp/opt_prime/examples
TORCH_DEVICE_BACKEND_AUTOLOAD=0 CUDA_VISIBLE_DEVICES=0 MOE_OFFLOAD_PROFILE=1 torchrun --nproc_per_node=1 \
  moe_offload_generate.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --tiers "<tiers>" --kv-cache \
  --max-new-tokens 64 --prompt "Explain mixture of experts in two sentences." --profile-prompt "<P>" \
  [--compile default --compile-skip-kv] [--check --experts-impl eager]
python3 moe_offload_rngd_opt_check.py --shape qwen3|solar [--experts 64] [--device N]     # kernel silicon check
```

## Results

Kernel (silicon, one decode token, random weights; rel. error vs cpu backend):

| shape | check | k=1 | k=8 | fit |
|---|---|---:|---:|---|
| Qwen3 H2048 F768, 32 slots | PASS 0.0045 | 189 us | 325 us | ~170 us + 0.48 TB/s |
| Qwen3, 64 slots | PASS 0.0060 | 186 us | 337 us | same |
| Solar H4096 F1280 | PASS 0.0054 | 263 us | 640 us | ~210 us + 0.58 TB/s |
| furiosa-torch `rngd` (Qwen3) | rel 4.5e-3 | 1.21-1.34 ms | 1.53-1.57 ms | ~1.2 ms + 0.2 TB/s |

Server phases (avg per run): upload x 37 us, write slots 7, launch 68, read back 61. Patched furiosa-opt-rt
(PIN_BOUNCE/MOVDIR64B): upload 9 us, end-to-end -14 us only.

End to end, 64 tokens (native dispatcher; "before" = Python worker threads):

| experts on | before tok/s | native tok/s | accel. share |
|---|---:|---:|---:|
| cpu only | 7.12 | 7.67 | 0 |
| rngd-opt x1 (32) + cpu | 8.94 | 9.77 | 54% |
| rngd-opt x2 (32) + cpu | 10.23 | 12.80 | 81% |
| rngd-opt x1 (64) + cpu | 10.91 | 12.56 | 77% |
| rngd-opt x2 (64) + cpu | 10.85 | **15.42** | 94% |
| gpu 15GB + cpu | 10.33 | 11.11 | 63% |
| gpu 15GB + rngd-opt (64) + cpu | - | 13.21 | |
| gpu 15GB + rngd-opt x2 (64) + cpu | 8.76 | 14.49 | |
| all experts on GPU (no offload, 80GB) | | 19.5 | 100% |

torch.compile of the stage module with the KV-cache attention eager (`--compile default --compile-skip-kv`), 48 tok:
all-GPU 19.5 -> **24.27 tok/s** (48/48 = HF), rngd-opt x2 (64) + cpu 15.42 -> **18.42 tok/s**. Compiling the whole
stage (KV module included) recompiles every step: 0.30 tok/s.

Per-layer profile, rngd-opt x2 (64) + cpu: outside (rest of the model) 864 us, wait 301, sync 103, split+merge+launch
83 (compiled: outside 614, wait 318).

Accuracy vs HF (eager experts): all-GPU 48/48; rngd-opt (64) + cpu 32/32 and 48/48; any config with the native CPU
kernel holding experts diverges at token 8 for this prompt (cpu-only too, also without the dispatcher); with the
torch CPU kernel 32/32 (cpu only) and 44/48 (rngd x2 + cpu). Compile does not change agreement.

Solar-Open-100B bf16 on the same path (no compile, 64 tokens, 3 profile prompts; same model as the PoC engine):

| experts on | tok/s | accel. pairs | per layer (us) | vs HF (eager experts) |
|---|---:|---:|---|---|
| rngd-opt x2 (40 GB each, ~28 slots/layer) + cpu | 8.56 | 74% | wait 1189, outside 1049, sync 108 | 64/64 PASS |
| gpu 45 GB + rngd-opt x2 + cpu | 10.18 | 88% | outside 1090, wait 470, launch 293, sync 118 | diverges at token 2: bf16 tie (gap 0.125 = 1 ulp) |
| PoC solar_engine (all on RNGD + CPU cold, Rust) | 31-33 | | ~650 total | exact |

torch.compile on Solar first failed (Inductor: export's _assert_tensor_metadata on the group mask, float32 vs int64);
fixed by stripping those asserts in the IR (aicomp 57646c9); tiny solar_open compiled matches HF 8/8.

Raw logs: session task outputs (not kept); numbers copied from the runs.

## Conclusion

The generated per-shape RNGD kernel is correct on silicon and 4.7x faster per call than furiosa-torch graphs; the
fixed ~170 us per call (DMA upload + read back + launch + pipe) dominates small k. Moving work off Python threads
(native dispatcher) gave +20-40% end to end. Best exact config: rngd-opt x2 (64 slots) + cpu, 15.4 tok/s, 18.4 with
compiled dense layers. Remaining per-layer cost is the OptimusPrime dense path (~0.6-0.9 ms) and the GPU<->host
routing sync (~0.1 ms) - the PoC (all on RNGD, one Rust process) does a Solar layer in ~0.65 ms. Next: CUDA-graph
partition around the offload op, native CPU kernel rounding. Same-model check: Solar-Open-100B runs exact on this
path but at 8.6-10.2 tok/s vs the PoC's 31-33 - the per-layer GPU<->host round trip, the Python dense path and the
out-of-process NPU calls are structural. Next: block-level restructuring (ModelSpec -> kernel providers -> planner
-> in-process segment executors) so an all-on-RNGD configuration like the PoC is one planner output.

## Addendum: engine selection, CUDA graphs, ATOM hybrid probes (same day)

- `auto_generate.py --model upstage/Solar-Open-100B` (aicomp feat/optimus-moe-offload): block spec 0.4 s (meta),
  candidates optimus (no estimate) / rngd-layer (manifest 31 tok/s) -> rngd-layer chosen; engine decode **29.4 tok/s**,
  14.65 tok/s incl. 69-token prompt (the PoC engine prefills token by token); 49/64 tokens equal to HF (tie check not
  run in this example).
- Qwen3-30B all-GPU with `--compile reduce-overhead --compile-skip-kv` (static shapes): 19.60 tok/s, 48/48 = HF; slower
  than default compile (24.27): the CUDA graphs are split around the eager KV module and CPU ops, and the GPU expert
  tier's pair split stays in Python. Dynamic shapes + reduce-overhead kept re-recording (0% GPU for >5 min).
- ATOM (etri-001, rbln0, rebel-compiler 0.11): graph call overhead p50 61 us (x*2), 207 us (4096x4096 GEMV), 664 us
  (4096x16384) -> per-layer segments are affordable on ATOM (furiosa-torch: ~1.2 ms per call).
- ATOM custom kernels exist: `rebel.triton` (Triton 3.2 dialect + RBLN libdevice), registered as
  `torch.library.triton_op("rbln_triton_ops::<name>")`; vllm-rbln ships attention kernels this way and the compiler has
  a generic converter (custom_converter: extend_triton_op -> triton_custom, reads the kernel's metadata from the Triton
  cache). A minimal SwiGLU-gate kernel in a compiled FFN did not convert yet (TVM frontend got a dict instead of a
  tensor); next: reproduce one vllm-rbln kernel end to end as a known-good baseline, then adapt.
- ATOM custom-kernel follow-up: the compiler reads hints from ~/.triton/cache_rbln/<md5(op name + input shapes)>/<op>.json,
  written by rblib.write_rtosa from the kernel *function* name, so op name and kernel name must match (fixed: frontend
  then accepts the op). Every user kernel then fails at `RBLNCompileError: Graph Generation: [DEVICE_GRAPH_CONVERSION]`
  - mul-only, sigmoid, exp bodies, 2-D host_layout "1:2", torch.compile(rbln_backend, strict) and
  rebel.compile_from_torch alike - including a previously written identity spike on this host (scripts/spike/
  spike12c, TP=1). vllm-rbln's attention kernels compile through dedicated converters. Conclusion: with
  rebel-compiler 0.11 user-defined rbln_triton_ops are not usable on ATOM without vendor guidance; the ATOM hybrid is
  graph-level for now (RBLN graphs, ~60 us per call).
