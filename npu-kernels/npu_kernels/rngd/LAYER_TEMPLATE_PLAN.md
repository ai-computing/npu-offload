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
