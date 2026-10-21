# npu-kernels

Kernel generators and device kernels for NPUs, consumed by orchestration layers (OptimusPrime `moe_offload`,
`exec`) through thin backend adapters. The orchestration side never needs a vendor toolchain to import; this package
needs one only to build kernels.

| module | device | what |
|---|---|---|
| `npu_kernels.rngd` | FuriosaAI RNGD (furiosa-opt 0.9) | routed-expert FFN kernel per (H, F) + expert server; weight-streaming GEMV per (rows, cols); decoder-layer engine per model geometry (`plan_layer` / `generate_layer`) |

Verified (2026-10-19, RNGD fw 2026.4.0): expert kernels for Qwen3-30B-A3B (H2048 F768) and Solar-Open-100B (H4096
F1280) pass the CPU emulator and silicon (k=1..8 experts 186-640 us per token); GEMV q 4096x2048 23 us (724 GB/s),
o 2048x4096 709 GB/s. `LAYER_TEMPLATE_PLAN.md` describes extending this to whole decoder layers.

Layer engine (2026-10-22, one RNGD card, fw 2026.4.0): `generate_layer(plan_layer(geometry))` writes a crate with the
attention block + router (one launch per layer: rmsnorm, q/k/v GEMV, per-head q/k RMSNorm, RoPE, KV-cache write,
attention over 512-token chunks, o_proj, residual, post-norm, router logits), the routed-expert kernel with a resident
hot-expert table, the LM head, a kernel check (`eval`) and a decode engine (`engine`). Qwen3-30B-A3B decodes with
tokens equal to HF transformers (3 prompts x 128 greedy tokens) at 26-28 tok/s (88 hot experts per layer on the card,
the rest on the CPU); the Solar-Open-100B geometry passes the same kernel checks. Details:
`../experiments/2026-10-22-rngd-layer-templates-qwen3.md`.

Lives in [ai-computing/npu-offload](https://github.com/ai-computing/npu-offload) (subdirectory `npu-kernels/`), next
to the track-2 PoC engine the kernels come from.

```bash
pip install "git+https://github.com/ai-computing/npu-offload#subdirectory=npu-kernels"
# or, from a checkout:
pip install -e .                    # Python side
pytest tests                        # plans (no toolchain)
python -c "from npu_kernels.rngd import plan, generate; generate(plan(2048, 768), '/tmp/k')" && \
  (cd /tmp/k && cargo run --release --bin eval -- check)          # CPU emulator

# decoder-layer engine for a model geometry (HF config.json -> crate)
python -c "import json; from npu_kernels.rngd import geometry_from_hf, plan_layer, generate_layer; \
  generate_layer(plan_layer(geometry_from_hf(json.load(open('<model_dir>/config.json'))), E_T=88), '/tmp/l')"
(cd /tmp/l && cargo run --release --bin eval -- all)                     # kernels vs host oracle, CPU emulator
(cd /tmp/l && cargo furiosa-opt run --release --bin eval -- all bench)   # RNGD
# decode: HF reference + hot-expert counts, the engine (teacher-forced and free runs), tie-aware compare
python tools/hf_reference.py <model_dir> ref --prompts tools/prompts_test.json --calib tools/prompts_calib.json --new 32
(cd /tmp/l && cargo furiosa-opt build --release --bin engine)
PLACEMENT=ref/expert_counts.json OUT_DIR=res <target_dir>/release/engine <model_dir> 32 ref/p_en ref/p_ko ref/p_code
python tools/compare_ref.py ref/p_en/ref.json res/p_en.free.json
```
