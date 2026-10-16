# npu-offload

**Run Mixture-of-Experts models that do not fit in accelerator memory, at interactive speed.**

npu-offload keeps the *hot* experts of a routed-MoE model in accelerator HBM, computes the *cold* experts on the
host CPU, and moves only activations over PCIe. It ships a model-agnostic control plane (`moe_offload`: placement
planning, memory budgeting, an OpenAI-compatible serving worker and an expert-affinity gateway) and a reference data
plane for **Solar-Open-100B (bf16, 205 GB) on two FuriosaAI RNGD cards (2 x 47.5 GiB HBM)**.

The approach follows [PolyStrata](https://huggingface.co/spaces/VecSzn/PolyStrata) (profile-guided expert placement,
GPU hits / CPU misses) and serves the result the way [llm-d](https://github.com/llm-d/llm-d) serves models: workers
behind a scheduler that scores endpoints by cache affinity and load.

---

## Highlights

- **Exact decoding.** bf16 end to end; outputs match a host reference on the prompts checked (the only divergences were exact bf16 logit ties).
- **~35 tok/s single-stream** for a 100B-parameter MoE whose weights are 2x larger than total HBM.
- **Profile-guided tiered placement** of experts across accelerators, with a planner that reports expected hit rates.
- **Memory budgeting** per accelerator (dense layers, KV cache, LM head, reserve → expert slots).
- **OpenAI-compatible worker** (`/v1/completions`, `/v1/chat/completions`, reasoning split) with vLLM-compatible
  Prometheus gauges, so existing llm-d scorers work unchanged.
- **Expert-affinity routing**: a gateway that sends each request to the worker whose resident experts best match it,
  the MoE analogue of prefix-cache-aware routing.
- **Low-latency host↔device transfers** on RNGD (MOVDIR64B posted writes, page-locked bounce buffers).

## Results

Solar-Open-100B, bf16, greedy decoding, 32 generated tokens, median of 5 rounds.
Host: 2x Xeon Silver 4410Y (24 cores, AMX/AVX-512), 1 TB DDR5, 2x RNGD (PCIe Gen5), FuriosaAI driver 2026.4.

| Configuration | Decode | Throughput | Accelerator hit rate |
|---|---:|---:|---:|
| All experts on CPU (baseline) | 435 ms/token | 2.3 tok/s | 0 |
| Hot experts on 2x RNGD, tuned (4 prompts, 3 unseen) | 30.7 – 32.1 ms/token | 31.2 – 32.6 tok/s | 0.82 – 0.89 |
| + MOVDIR64B uploads, pinned DMA reads | **28.7 ms/token** | **34.9 tok/s** | 0.86 |
| Two interleaved streams (aggregate) | 64.9 ms per pair step | 30.8 tok/s | — |
| Approximate: int8 cold experts, drop experts ranked ≥ 4 if cold | 29.1 ms/token | 34.3 tok/s | 92 % top-1 agreement |

Where the time goes (per layer): accelerator A attention + shared expert ≈ 175 µs, then the slower of accelerator B's
hot experts (≈ 240 µs) and the CPU's cold experts (≈ 237 µs, bound by DRAM bandwidth). Measurement records are in
[`experiments/`](experiments) and [`PLAN.md`](PLAN.md).

## How it works

```
                 ┌──────────── gateway (expert affinity − load) ────────────┐
   client ─────▶ │ classify request → pick worker → proxy OpenAI request     │
                 └───────────────┬──────────────────────────────────────────┘
                                 │ HTTP
                 ┌───────────────▼──────────────┐
                 │ worker (OpenAI API, /metrics)│
                 └───────────────┬──────────────┘
                                 │ JSON lines (stdin/stdout)
   ┌─────────────────────────────▼───────────────────────────────────────────────┐
   │ engine, per layer:                                                          │
   │   accelerator A: combine → attention → shared expert → router logits        │
   │   host: top-k routing, split experts into  A-resident | B-resident | cold   │
   │   in parallel:  A hot experts  ‖  B hot experts  ‖  CPU cold experts         │
   │   activations only cross PCIe; weights never move                           │
   └─────────────────────────────────────────────────────────────────────────────┘
```

1. **Profile** which experts each layer actually routes to on representative prompts.
2. **Budget** each accelerator's HBM and derive how many expert slots per layer fit.
3. **Place** the most frequently used experts in those slots (`top` or `mix:<h>` policies). On Solar, ~44 % of the
   experts serve 82–89 % of routed tokens.
4. **Serve**: the engine runs attention and hot experts on the accelerators and cold experts on NUMA-local host
   memory with AVX-512 BF16 kernels, all three in parallel per layer.

## Requirements

| Component | Needed for |
|---|---|
| Python ≥ 3.10 | control plane, worker, gateway (standard library only) |
| `torch`, `transformers` | tokenizer in the worker, profiling, planning from HF checkpoints (`pip install .[model]`) |
| FuriosaAI RNGD x2, driver 2026.4, `cargo-furiosa-opt` 0.9 | the reference Solar engine |
| x86-64 CPU with AVX-512 BF16, ≥ 256 GB RAM | cold experts on the host |
| Optional: a GPU | training the experimental EAGLE-3 draft head |

## Installation

```bash
git clone https://github.com/ai-computing/npu-offload.git
cd npu-offload
pip install -e .            # control plane, CLI `moe-offload`
pip install -e .[model]     # + torch / transformers

# reference engine (Solar-Open-100B on 2x RNGD)
cd track2_opt/solar_engine
cp ~/.cargo/registry/src/*/furiosa-opt-rt-0.9.0/prebuilt/* vendor/furiosa-opt-rt/prebuilt/
cargo furiosa-opt build --release --bin solar_engine
```

The engine also needs per-model inputs (RoPE tables, prompt template, placement); generate them with
`python3 solar_hybrid/export_engine_inputs.py` and `moe-offload plan`.

## Quick start

```bash
MODEL=~/.cache/huggingface/hub/models--upstage--Solar-Open-100B/snapshots/<rev>

# 1. how many expert slots fit, how much stays on the host
moe-offload budget --model $MODEL

# 2. plan a placement from a routing profile
moe-offload plan --model $MODEL --profile p0/results/routes.pt --slots 24,32 --policy mix:4 \
    --out track2_opt/solar_engine/data/placement.json

# 3. serve
moe-offload gateway --port 8080 &
moe-offload worker --engine track2_opt/solar_engine/target/release/solar_engine \
    --data track2_opt/solar_engine/data --model $MODEL --port 8101 --gateway http://127.0.0.1:8080

# 4. query (OpenAI API)
curl -s localhost:8080/v1/chat/completions \
    -d '{"messages":[{"role":"user","content":"Explain MoE offloading in two sentences."}],"max_tokens":512}'
```

Without accelerators, everything above the engine runs against a mock: `moe-offload worker --mock`.

### Python API

```python
from moe_offload.spec import ModelSpec
from moe_offload.profile import counts_from_routes
from moe_offload.placement import plan, hit_rates
from moe_offload.engine import EngineProcess, SOLAR_ENV

spec = ModelSpec.from_hf_config(MODEL)
counts = counts_from_routes("routes.pt", spec.layers, spec.experts)
placement = plan(counts, slots=[24, 32], policy="mix:4")
print(hit_rates(counts, placement))        # [tier0, tier1, host] fractions of routed tokens

eng = EngineProcess("solar_engine", "data/", env=SOLAR_ENV)
tokens, stats = eng.run(prompt_ids, max_new=64)
print(stats["decode_ms_per_tok"], stats["hit_rate"])
```

## Serving and llm-d integration

| Endpoint (worker) | |
|---|---|
| `POST /v1/completions` | `prompt` or `prompt_token_ids`; `max_tokens` |
| `POST /v1/chat/completions` | chat template applied; `reasoning_content` split from `content` |
| `GET /metrics` | `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`, `moe_offload_expert_hit_rate`, `moe_offload_decode_ms_per_token`, per-class hit rate |
| `GET /health`, `/v1/models` | |

The gateway reuses the worker registry and endpoint picker of an llm-d style RNGD gateway (`$RNGD_LLMD`) and adds an
**expert-affinity scorer**: each worker reports the hit rate of every request by class (Korean / code / English),
the gateway keeps an EWMA per (worker, class) seeded with the hit rates the worker's placement predicts, and routes
to the best `affinity − load` score. With workers carrying different placements, requests go where their experts
already live. A Kubernetes sketch (Deployment + `InferencePool`) is in
[`moe_offload/deploy/`](moe_offload/deploy/k8s-moe-worker.yaml).

## Engine configuration

Environment variables of the reference engine (the worker sets the tuned values from `engine.SOLAR_ENV`):

| Variable | Default (tuned) | Effect |
|---|---|---|
| `URES` | 1 | accelerator A keeps the next layer's weights resident in on-chip memory |
| `LM_ON_B` | 1 | LM head on accelerator B |
| `CHIP_CORES` / `CPU_THREADS` / `OMP_PLACES` | 7 / 17 / `{7}:17:1` | cores for accelerator host threads vs. cold-expert threads |
| `OMP_WAIT_POLICY` | passive | spinning OpenMP threads slow the accelerator host threads |
| `MMIO_VEC`, `MMIO_LIMIT` | 2, 32768 | uploads ≤ 32 KiB through the BAR window with MOVDIR64B |
| `PIN_BOUNCE` | 1 | small DMA reads through page-locked buffers (needs `/dev/dma_heap/system` access) |
| `SQPOLL_OFF`, `DMA_IDLE_US` | 1, 50 | DMA submission thread behaviour |
| `COLD_INT8`, `TAIL_SKIP` | off | approximate modes (int8 cold experts, skip low-ranked cold experts) |
| `NSTREAM` | 1 | 2 = two interleaved decode streams |

## Experimental: speculative decoding

Solar-Open-100B has no MTP head and no compatible small draft model, so [`eagle3/`](eagle3) trains an EAGLE-3 draft
head from teacher features extracted by streaming the bf16 model through a GPU layer by layer
(`extract.py` matches the engine's tokens exactly). Current acceptance length on Solar's own outputs is
τ ≈ 1.5 (k = 6), below the ≈ 2.5 needed to pay for multi-token verification; training on larger reasoning-style
corpora is in progress. The engine does not use the draft yet.

## Repository layout

```
moe_offload/            control plane library (pip package, CLI `moe-offload`)
  spec.py profile.py placement.py budget.py engine.py worker.py cli.py
  llmd/                 request classes, affinity scoring, gateway
  tests/                NPU-free tests (`moe-offload selftest`)
  deploy/               Kubernetes sketch
track2_opt/solar_engine reference data plane (Rust, furiosa-opt): engine, kernels, CPU kernels (C), patched runtime
track2_opt/solar_kernels  standalone kernel crates with CPU-emulation checks and NPU benchmarks
solar_hybrid/           Python reference model (HF-faithful) and input export
p0/                     microbenchmarks and the routing profile
eagle3/                 EAGLE-3 draft training pipeline (experimental)
experiments/            one record per measurement (setup, exact command, results)
PLAN.md, STATUS.md      design notes, decisions and running status
```

## Development

```bash
moe-offload selftest                 # placement, budget, reasoning split, classifier, gateway e2e (no hardware)
cd track2_opt/solar_engine && cargo furiosa-opt build --release --bin solar_engine
REPEAT=5 ./target/release/solar_engine data_p2 32      # benchmark mode: prints tokens, ms/token, per-stage profile
```

Benchmarks are noisy on shared hosts (±10–15 %): use `REPEAT`, check the load average, and record runs in
`experiments/` using [`experiments/TEMPLATE.md`](experiments/TEMPLATE.md).

## Limitations

- One decode stream per engine; requests queue in the worker. No prefix cache: prompts are processed token by token.
- The reference engine is specific to Solar-Open-100B: model dimensions and slot counts are compile-time constants of
  its kernels. The control plane accepts any routed-MoE config, but a new model needs its own data plane build.
- Accelerators are opened exclusively; other processes using the same RNGD cards block engine start-up.
- Expert-affinity routing is validated with mock workers; a single host with two cards runs one real worker.
- The Kubernetes manifests are untested.

## License

Not yet decided; until then this repository is for internal research use. `track2_opt/solar_engine/vendor/furiosa-opt-rt`
is a modified copy of FuriosaAI's `furiosa-opt-rt` 0.9.0 (Apache-2.0); see its
[`VENDOR_PATCHES.md`](track2_opt/solar_engine/vendor/furiosa-opt-rt/VENDOR_PATCHES.md). Model weights and datasets
are not included and keep their own licenses.

## Acknowledgements

PolyStrata for the hot/cold expert offloading design, llm-d for the serving architecture, and FuriosaAI for the
furiosa-opt toolchain.
