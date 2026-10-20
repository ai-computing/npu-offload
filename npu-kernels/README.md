# npu-kernels

Kernel generators and device kernels for NPUs, consumed by orchestration layers (OptimusPrime `moe_offload`,
`exec`) through thin backend adapters. The orchestration side never needs a vendor toolchain to import; this package
needs one only to build kernels.

| module | device | what |
|---|---|---|
| `npu_kernels.rngd` | FuriosaAI RNGD (furiosa-opt 0.9) | routed-expert FFN kernel per (H, F) + expert server; weight-streaming GEMV per (rows, cols) |

Verified (2026-10-19, RNGD fw 2026.4.0): expert kernels for Qwen3-30B-A3B (H2048 F768) and Solar-Open-100B (H4096
F1280) pass the CPU emulator and silicon (k=1..8 experts 186-640 us per token); GEMV q 4096x2048 23 us (724 GB/s),
o 2048x4096 709 GB/s. `LAYER_TEMPLATE_PLAN.md` describes extending this to whole decoder layers.

Lives in [ai-computing/npu-offload](https://github.com/ai-computing/npu-offload) (subdirectory `npu-kernels/`), next
to the track-2 PoC engine the kernels come from.

```bash
pip install "git+https://github.com/ai-computing/npu-offload#subdirectory=npu-kernels"
# or, from a checkout:
pip install -e .                    # Python side
pytest tests                        # plans (no toolchain)
python -c "from npu_kernels.rngd import plan, generate; generate(plan(2048, 768), '/tmp/k')" && \
  (cd /tmp/k && cargo run --release --bin eval -- check)          # CPU emulator
```
