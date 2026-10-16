# Solar-Open-100B decode on furiosa-opt — shared spec

Goal: ≥30 tok/s decode (≤33 ms/token) for Solar-Open-100B bf16 on 2× RNGD with hot experts in HBM and
cold experts on host CPU. Every kernel here is T=1 decode, bf16 weights, `#[device(chip = 1, pe = 8)]`
(full chip), toolchain `cargo furiosa-opt 0.9.0`, nightly-2026-05-01 (copy rust-toolchain.toml from
`track2_opt/moe_opt`). Use `track2_opt/upstream/furiosa-opt-examples/src/transformer` (Qwen3 decode step:
projection / attention / decoder / lm_head) as the starting point, and `track2_opt/moe_opt/src/kernel/ffn.rs`
(Solar expert FFN, ~1.08 TB/s) as the GEMV reference.

## Model dims (Solar-Open-100B)
H = 4096 hidden; 64 q heads, 8 kv heads, head_dim D = 128 → Q = 8192 (= N 8 × G 8 × D 128), KV = 1024 (N 8 × D 128).
Shared expert F = 1280 (SwiGLU). Routed experts: 128, top-8, F = 1280. Router: Wr [128, H], logits in fp32.
Vocab W = 196608 (= 24 × 8192, already a multiple of 8192). rms eps = 1e-5. RoPE: rotate_half on full D=128,
cos/sin come from the host per position (yarn-scaled table), bf16 [D].
Upstream example differences: H 1024→4096, G 2→8, Q 2048→8192, no q/k RMSNorm in Solar, MLP replaced.

## Reference math (HF transformers solar_open, bf16 semantics)
```
h  = rmsnorm(x, w_in)                           # fp32 math, cast to bf16, then * w (bf16)
q,k,v = h@Wq^T, h@Wk^T, h@Wv^T                  # [8192], [1024], [1024]
q,k = q*cos + rotate_half(q)*sin, same for k    # per head, D=128
kcache[pos], vcache[pos] = k, v
a  = softmax(q·K^T / sqrt(128)) V               # GQA: q head i uses kv head i // 8
x2 = x + a @ Wo^T                               # Wo [4096, 8192]
h2 = rmsnorm(x2, w_post)
y  = x2 + Wsd @ (silu(Wsg@h2) * (Wsu@h2))       # shared expert
logits = h2(fp32) @ Wr^T (fp32)                 # router input for host top-k
x_next = y + Σ_k g_k * expert_k(h2)             # routed (NPU hot experts + CPU cold experts)
final: rmsnorm(x, w_final) @ Wlm^T              # [196608]
```
A host oracle of all of this exists in `npu_offload/solar_hybrid/model.py` + `runtime.py` (verified vs HF).

## Deliverables per kernel
- Kernel in its own crate under `track2_opt/solar_kernels/<name>/` (copy Cargo.toml/rust-toolchain from moe_opt).
- A test/bench binary: CPU-emulation check vs a scalar f32 oracle with random weights at real dims
  (tolerance like moe_opt host_util::check), and NPU `bench` mode printing launch p50 (µs) and GB/s.
- Report: device fn signature (HBM layouts of every input/output, including any pre-permuted weight layout and
  the host permutation function), measured NPU p50, correctness, and known limits.

## Performance targets (per layer, decode, full chip)
attention block (rmsnorm+qkv+rope+kv write + attention + o_proj + residual + post-rmsnorm) ≲ 250 µs;
shared expert + router ≲ 50 µs; lm_head ≲ 2 ms. Weight-streaming GEMVs should approach ≥0.9 TB/s.

## NPU etiquette
Only use the chip you were assigned (FURIOSA_OPT_NPUS / FURIOSA_VISIBLE_CHIPS). Never kill processes you did
not start. If busy, retry later; do CPU-emulation work meanwhile.
