# Step 3: RNGD layer engine for more model geometries

Goal: the native RNGD layer engine (today: npu-offload `solar_engine`, Solar-Open-100B only, 29-33 tok/s exact) built
from a `ModelSpec`, so `opt_prime.exec.select_engine` picks it for other MoE models. First target: Qwen3-30B-A3B.

## What depends on the geometry

The engine's kernels are furiosa-opt Rust with the geometry in two places: axis constants (`axes![...]`) and tiling
expressions inside the mappings (`m![H / 2048]`, `m![RW / 512 % 2, H / 32]`, slice groups of 32 per kv head, a packed
host buffer of 33 x 128 f32, ...). The tiling expressions assume Solar's sizes fill the chip (2 clusters x 256
slices); smaller models leave slices idle or do not tile at all, so each kernel needs a plan function, the way
`generate.plan(H, F)` derives the expert kernel's tiling.

| axis (Solar) | meaning | Solar-Open-100B | Qwen3-30B-A3B |
|---|---|---:|---:|
| H | hidden | 4096 | 2048 |
| Q = heads x D | q width | 8192 | 4096 |
| P = kv_heads x D | k / v width | 1024 | 512 |
| N | kv heads | 8 | 4 |
| G | q heads per kv head | 8 | 8 |
| D | head dim | 128 | 128 |
| RW = 2 x N x D | [Wk; Wv] rows per GEMV chunk | 2048 | 1024 |
| Qc = Q / RW | Wq chunks | 4 | 4 |
| Hr x Hc = H | o_proj rows x chunks | 1024 x 4 | 1024 x 2 |
| E, top-k, F | experts | 128, 8, 1280 | 128, 8, 768 |
| shared expert | | F 1280 | none |
| vocab | lm_head | 196608 | 151936 |

## Kernel by kernel

| kernel | Solar file | change for Qwen3 |
|---|---|---|
| q/k/v GEMV (`attn/gemv.rs`) | RC = RW/1024 clusters, RS = (RW/512 % 2, H/32) slices | RW = 1024 gives one cluster: re-split rows (RS = RW/256 % 4 x H/32) so both clusters and 256 slices are used |
| RoPE (`attn/rope.rs`) | one kv head per 32-slice group (`SliceN32 = m![N, 1 # 32]`) | N = 4: 64-slice groups, or two heads per group |
| **q/k RMSNorm (new)** | not in Solar | per-head RMSNorm over D before RoPE (Qwen3 `q_norm` / `k_norm`); same reduction pattern as `rmsnorm.rs` on D = 128 |
| attention over KV chunks (`attention.rs`) | T = 512 token chunks, G = 8 | G, D, T unchanged; N halves (fewer slice groups) |
| o_proj + residual + post-norm (`epilogue.rs`) | Hr / 512 clusters | H = 2048: Hc = 2 |
| router (`dense/shared.rs`, router part) | logits GEMV [E, H] on chip; top-k on the host | GEMV retiled for H = 2048; the scoring (Qwen3 softmax vs Solar sigmoid + group) is host code: RouterSpec.scoring decides, no new kernel |
| shared expert (`dense/shared.rs`) | F 1280 | absent: skipped when MoESpec.shared_ffn == 0 |
| routed experts (`ffn_k8`, tables) | E tables per chip | already generated per shape (`generate.plan(2048, 768)` passes emulation and silicon) |
| combine (`unified_a/layer.rs`) | H % 128 per slice, `CC = H / 2048` | H = 2048: one cluster x 16 slices, or 64 elements per slice |
| packed host buffer (`Pk = 33, R = 128`) | H / 128 rows + gates + logits | rows = H / 128 + 1 (gates) -> Pk = 17 for H = 2048; router logits E = 128 fit one row |
| lm_head (`dense/lm_head.rs`, `lm_half.rs`) | vocab split over 2 chips, W = 196608 | W = 151936 = 2^7 x 1187: tile by 128-row blocks with a padded tail |
| host engine constants (`host.rs`, `chip.rs`, ...) | `const H = 4096` etc. | generated from ModelSpec (layer count, dims, scoring, shared expert, vocab) |

## Order and checks

1. Plan functions per kernel (pure Python next to `generate.plan`), unit-tested on Solar (must reproduce today's
   constants) and Qwen3.
2. Templates: q/k/v GEMV + RoPE + q/k norm first (largest change), checked with `cargo run --bin eval -- check` on
   the CPU emulator against a torch oracle per kernel, then on silicon (`cargo furiosa-opt run`).
3. Remaining kernels (attention, epilogue, combine, router GEMV, lm_head), same checks.
4. Host engine generated from ModelSpec; `engine_manifest.json` written from `engine_geometry(spec)` so the selector
   offers it. End to end: Qwen3-30B-A3B tokens vs HF (tie-aware), tok/s vs the Optimus path (15-18 tok/s today).

## Status (2026-10-22, npu-offload#3)

Implemented as `layer.py` (`LayerGeometry`, `plan_layer`, `generate_layer`) + `layer_template/`. What differs from the
table above:

- q/k/v GEMV: one rule for every geometry, the expert gate/up rule (A = 8192 / H row groups x H / 32 column blocks,
  RPS = RW / (2A) rows per slice). Solar: A 2, RPS 512 (= today's `RS`); Qwen3: A 4, RPS 128, both clusters busy.
- q path: q goes through HBM as `[N, G, D]` and is normalized / rotated one head per 256 / (N G) slices (Qwen3: 8);
  k and v one head per 256 / N slices (Qwen3: 64). For q/k-norm models q and k are divided by the deferred input-norm
  factor before the head norm: the norm's eps is not scale invariant, and skipping this broke layers 0-2 of Qwen3
  (embeddings ~1e-2; relative error 0.2-0.8 per layer before the fix, 0.003-0.03 after).
- attention: N / 2 kv heads per cluster, `SA = [N % (N/2), 1 # (8 / (N/2)), T / 16]`, otherwise unchanged.
- o_proj: Ao = 8192 / Q row groups (Qwen3: 2, RPSo 256); the output goes through HBM into an epilogue layout with one
  slice per cluster holding H / 2 (one 4 KB hop instead of geometry-specific reshapes).
- post-norm: HF rounding (`bf16(bf16(x2 / rms) * w)`); the single-rounding form failed the Solar-geometry router check
  on small activations.
- router: GEMV on chip in the same launch (logits out with x2 and h2); scoring (softmax top-k for Qwen3) on the host.
- LM head: generic GEMV (`_lm_plan`, rows padded to a multiple of the chunk; Qwen3 38 x 4032 rows).
- experts: the expert generator's kernel with a resident hot table of E_T slots per layer (ids packed 7 bits per slot
  when E_T > 64); experts outside the table run on the CPU (bf16 weights from the safetensors mmap), overlapped with
  the chip.
- host engine: `bin/engine` (one process, one card): embedding, top-k, CPU experts, HF-order bf16 combine, argmax.
  Shared expert, sigmoid/group routing (Solar) and two-chip placement are not in the generated engine; Solar keeps the
  hand-written engine.

Limits: head_dim 128, G = 8 q heads per kv head, kv heads 4/8/16, H | 8192, Q | 8192, context <= 512 x max(chunks)
(default 2048), decode only, one request.

Results: `../../../experiments/2026-10-22-rngd-layer-templates-qwen3.md`.
