# RNGD layer engine from a model geometry: Qwen3-30B-A3B decode vs HF

- **Date**: 2026-10-22
- **Task**: npu-offload#3 (RNGD layer engine: shape templates for attention / norm / router)
- **Host / device**: RNGD+ATOM host (`npu-node-1`), one RNGD card through a Kubernetes Job (`furiosa.ai/rngd: 1`), fw /
  driver 2026.4.0, furiosa-opt 0.9.0 (rust nightly-2026-05-01), 2x AMD EPYC 9254 (job: 24-32 cores); HF reference on
  CPU, torch 2.10.0+cpu, transformers 5.1.0 (`furiosaai/furiosa-llm:latest` image)
- **Commit**: branch `feat/3-layer-templates`, template code 0ea64af (runs below)

## Question

Can the RNGD layer engine be generated from a model geometry instead of hand-written for Solar-Open-100B, and does the
generated engine decode Qwen3-30B-A3B with the same tokens as HF? At what rate?

## Setup

- `npu_kernels.rngd.layer`: `plan_layer(geometry)` derives every tiling constant (q/k/v and o_proj GEMV splits, head
  slice groups, attention cluster split, router and LM-head GEMVs); `generate_layer` writes a crate from
  `layer_template/`: `attn_block_c{1,2,4}` (one launch per layer: rmsnorm, q/k/v GEMV, per-head q/k RMSNorm, RoPE,
  KV-cache write, attention over 512-token chunks, o_proj, residual, post-norm, router logits), `ffn_k` (expert
  generator kernel, resident table of E_T experts per layer), `lm_head`, `bin/eval`, `bin/engine`.
- Engine (one process, one card): embedding, softmax top-8 and renormalization on the host; the E_T most used experts
  of each layer resident on the card, the others on the CPU (bf16 weights, rayon, overlapped with the chip); expert
  outputs combined in HF order (ascending expert id, bf16 accumulation); greedy argmax. Prompt fed token by token.
- Hot experts: per-layer top-k counts of 8 calibration prompts (prefill + 32 greedy tokens), disjoint from the 3 test
  prompts (`npu-kernels/tools/prompts_calib.json`, `prompts_test.json`).
- Correctness criterion: per step, engine argmax == HF greedy token, or the two tokens are within one bf16 ulp in the
  HF logits (`tools/compare_ref.py`). Teacher-forced runs (HF tokens fed back) compare every step; free runs compare
  until the first difference.
- Solar-Open-100B: the hand-written engine (`track2_opt/solar_engine`) is not modified. The Solar geometry is run
  through the same template for the kernel checks only (its shared expert, sigmoid routing and two-chip placement are
  not in the generated engine).

## Command

```bash
# crate (dev pod, CPU only): plan + generate, CPU-emulator check, NPU build
python -c "from npu_kernels.rngd.layer import *; generate_layer(plan_layer(QWEN3_30B_A3B, E_T=88), 'lq3e88')"
(cd lq3e88 && cargo run --release --bin eval -- all && cargo furiosa-opt build --release --bin eval --bin engine)
# HF reference + hot-expert counts (CPU)
python npu-kernels/tools/hf_reference.py <Qwen3-30B-A3B-Instruct-2507> ref --prompts prompts_test.json \
  --calib prompts_calib.json --new 32          # and --new 128 (no --calib) into ref128
# device job (one card)
./eval all bench
CPU_THREADS=20 REPEAT=3 PLACEMENT=ref/expert_counts.json OUT_DIR=res ./engine <model_dir> 32 ref/p_en ref/p_ko ref/p_code
python npu-kernels/tools/compare_ref.py ref/p_en/ref.json res/p_en.free.json
```

## Results

Kernel checks against the host oracle (HF semantics, random weights; pass = < 1% of outputs outside 2-3% rel + abs
tolerance), CPU emulator and silicon, both geometries: PASS for `attn_block_c1/c2/c4` (k row, v row, x2, h2, router
logits; contexts 301, 1024, 1300, plus a small-activation case), `ffn_k` (8 slots) and `lm_head`.

Kernel time on silicon (p50 of 50 launches, one token):

| kernel | Qwen3-30B-A3B | Solar-Open-100B geometry | hand-written Solar engine |
|---|---:|---:|---:|
| attn_block_c1 (+ router) | 82 us | 173 us | 160-190 us (attention block only) |
| attn_block_c2 | 87 us | 178 us | |
| attn_block_c4 | 102 us | 188 us | |
| ffn_k, 8 experts | 132 us | 274 us | |
| lm_head | 565 us (1.11 TB/s) | 1442 us | |

Qwen3-30B-A3B decode, tokens vs HF (32 greedy tokens, 3 prompts):

| hot experts / layer | prompt | forced | free | decode ms/tok (free, 3 runs) | tok/s | hit rate |
|---|---|---|---|---:|---:|---:|
| 64 | p_en | 32/32 exact | 32/32 exact | 41.9-43.3 | 23.1-23.9 | 0.747 |
| 64 | p_ko | 32/32 exact | 32/32 exact | 39.1-39.5 | 25.3-25.5 | 0.811 |
| 64 | p_code | 32/32 exact | 32/32 exact | 41.8-41.9 | 23.9 | 0.717 |
| 88 | p_en | 32/32 exact | 32/32 exact | 36.7-38.6 | 25.9-27.3 | 0.864 |
| 88 | p_ko | 32/32 exact | 32/32 exact | 35.0-35.6 | 28.1-28.6 | 0.927 |
| 88 | p_code | 32/32 exact | 32/32 exact | 36.0-37.4 | 26.7-27.8 | 0.858 |

128 greedy tokens (E_T = 88; HF stops p_en at EOS after 86 tokens):

| prompt | forced | free | decode ms/tok (free, 2 runs) | tok/s |
|---|---|---|---:|---:|
| p_en | 85/86 exact, 1 within one ulp | 45 exact, then the same one-ulp step | 36.6-37.2 | 26.9-27.3 |
| p_ko | 128/128 exact | 128/128 exact | 33.6-34.1 | 29.4-29.8 |
| p_code | 128/128 exact | 128/128 exact | 40.0-40.4 | 24.7-25.0 |

The p_en step 45: HF logits 35.75 (token 16876) vs 35.5 (token 702); engine 35.75 (702) vs 35.5 (16876): the two
candidates swap by one bf16 ulp. Under a strict equal-logit tie rule this step would count as a mismatch.

Time per forward step (E_T 88, p_code free run, ms, averaged over prefill and decode steps): experts on chip 12.3 (ids write + launch + 32 KB read, 48 layers),
attention launches 5.7, downloads of x2 / h2 / logits 4.8, CPU experts beyond the chip 2.6, combine 1.7, upload x 1.6,
LM head 0.7. Loading (48 layers, 39.9 GB of expert tables + attention weights + LM head into HBM) takes 96-99 s.

HBM use at E_T 88: experts 39.9 GB, attention + router 1.8 GB, KV cache (2048 tokens) 0.8 GB, LM head 0.6 GB.

Raw logs: Job logs `npo-lq3-run3`, `npo-lq3-r128`, `npo-lq3-eng2`, `npo-lq3-chk` (shared store, `npo/logs/`).

Debug history: the first silicon run matched only 31/32, 31/32 and 20/32 steps. `CHECK=1` (host oracle per layer, same
input) showed relative errors of 0.06-0.8 in layers 0-2 only: q and k were head-normalized before the deferred
input-norm factor was applied, so eps (1e-6) acted on vectors scaled by rms(x) ~ 1e-2. Dividing first fixed it
(errors 0.003-0.03 in every layer). The host-oracle path (`HOSTPATH=1`) matched HF 32/32 on two prompts before the fix.

## Conclusion

The layer engine is generated from the geometry: the same template builds Qwen3-30B-A3B and the Solar-Open-100B
shapes, and the generated Qwen3 engine decodes with HF's tokens (every compared step exact except one one-ulp swap in
the 342 teacher-forced steps of the 128-token runs) at 25-30 tok/s on one RNGD card with 88 hot experts per layer, vs 15-18 tok/s for Qwen3 through
the generic OptimusPrime offload path. Next: one download per layer (x2 / h2 / logits packed), device-side weighted
expert sum, a server mode for OptimusPrime, and the Solar-specific pieces (shared expert, sigmoid routing, two chips)
before the hand-written Solar engine can be replaced.
