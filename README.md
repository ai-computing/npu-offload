<div align="center">

<h1>npu-offload</h1>

<h3>Run Mixture-of-Experts models that do not fit in accelerator memory, at interactive speed</h3>

<p>
  <strong>English</strong> · <a href="README.ko.md">한국어</a>
</p>

<p>
  <a href="https://github.com/ai-computing/ai-rm"><img src="https://img.shields.io/badge/ai--rm-L2%20memory%20%26%20transport-22d3ee" alt="ai-rm layer: L2 memory and transport"></a>
  <a href="https://github.com/ai-computing/rm-core/blob/main/PROTOCOL.md"><img src="https://img.shields.io/badge/protocol-v0.1-7c5cff" alt="Protocol v0.1"></a>
  <a href="ai-rm.yaml"><img src="https://img.shields.io/badge/maturity-prototype-f59e0b" alt="Maturity: prototype"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&amp;logoColor=white" alt="Python 3.10+"></a>
  <a href="track2_opt/solar_engine"><img src="https://img.shields.io/badge/engine-Rust%20%C2%B7%20furiosa--opt-dea584?logo=rust&amp;logoColor=white" alt="Engine: Rust with furiosa-opt"></a>
  <a href="#-license"><img src="https://img.shields.io/badge/license-undecided-64748b" alt="License: undecided"></a>
</p>

<p>
  <a href="#-highlights"><b>Highlights</b></a> ·
  <a href="#-results"><b>Results</b></a> ·
  <a href="#-quick-start"><b>Quick start</b></a> ·
  <a href="#-ai-rm-integration"><b>ai-rm integration</b></a> ·
  <a href="#-documentation"><b>Docs</b></a>
</p>

</div>

---

**npu-offload** keeps the *hot* experts of a routed-MoE model in accelerator HBM, computes the *cold* experts on the
host CPU, and moves only activations over PCIe. It ships a model-agnostic control plane (`moe_offload`: placement
planning, memory budgeting, an OpenAI-compatible serving worker and an expert-affinity gateway) and a reference data
plane for **Solar-Open-100B (bf16, 205 GB) on two FuriosaAI RNGD cards (2 x 47.5 GiB HBM)**. In
[ai-rm](https://github.com/ai-computing/ai-rm) it is the **L2 memory & transport** component for expert offloading
between NPU HBM and host DRAM, registered as the workload runner `airm.workloads:npu-offload`.

It is also a **migration source for the Optimus Prime engine**: expert offloading is planned to become an Optimus
Prime module ([consolidation plan](https://github.com/ai-computing/ai-rm/blob/master/docs/engine-integration.md)),
keeping the workload and memory-tier boundary (it is not a `KVTransport`). Until that consolidation is accepted, this
repository remains the executable path.

The approach follows [PolyStrata](https://huggingface.co/spaces/VecSzn/PolyStrata) (profile-guided expert placement,
GPU hits / CPU misses) and serves the result the way [llm-d](https://github.com/llm-d/llm-d) serves models: workers
behind a scheduler that scores endpoints by cache affinity and load.

> [!NOTE]
> **Prototype.** The reference engine is specific to Solar-Open-100B, numbers are decode-only, the Helm chart has
> only been rendered (no container image yet), and expert-affinity routing is validated with mock workers.

## ✨ Highlights

<table>
<tr>
<td width="33%" valign="top">
<b>Exact decoding</b><br>
bf16 end to end; outputs match a host reference on the prompts checked (the only divergences were exact bf16 logit ties).
</td>
<td width="33%" valign="top">
<b>~35 tok/s single-stream</b><br>
For a 100B-parameter MoE whose weights are 2x larger than total HBM, with hot experts on 2x RNGD and cold experts on the CPU.
</td>
<td width="33%" valign="top">
<b>Profile-guided tiered placement</b><br>
Experts are placed across accelerators from a routing profile; the planner reports expected hit rates per tier.
</td>
</tr>
<tr>
<td width="33%" valign="top">
<b>Memory budgeting</b><br>
Per-accelerator budget (dense layers, KV cache, LM head, reserve → expert slots), declared as rm-core memory tiers.
</td>
<td width="33%" valign="top">
<b>OpenAI-compatible worker</b><br>
<code>/v1/completions</code>, <code>/v1/chat/completions</code> with reasoning split, and vLLM-compatible Prometheus gauges so existing llm-d scorers work unchanged.
</td>
<td width="33%" valign="top">
<b>Expert-affinity routing</b><br>
A gateway sends each request to the worker whose resident experts best match it, the MoE analogue of prefix-cache-aware routing.
</td>
</tr>
</table>

Host↔device transfers on RNGD use MOVDIR64B posted writes and page-locked bounce buffers for low latency.

## 📊 Results

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
hot experts (≈ 240 µs) and the CPU's cold experts (≈ 237 µs, bound by DRAM bandwidth). Benchmarks are noisy on shared
hosts (±10–15 %). Measurement records are in [`experiments/`](experiments), [`PLAN.md`](PLAN.md) and
[`STATUS.md`](STATUS.md).

## 🏗 Architecture

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

<details>
<summary><b>Requirements</b></summary>
<br>

| Component | Needed for |
|---|---|
| Python ≥ 3.10 | control plane, worker, gateway (standard library only) |
| `torch`, `transformers` | tokenizer in the worker, profiling, planning from HF checkpoints (`pip install .[model]`) |
| FuriosaAI RNGD x2, driver 2026.4, `cargo-furiosa-opt` 0.9 | the reference Solar engine |
| x86-64 CPU with AVX-512 BF16, ≥ 256 GB RAM | cold experts on the host |
| Optional: a GPU | training the experimental EAGLE-3 draft head |
</details>

## 🚀 Quick start

Without accelerators, the control plane and everything above the engine run against a mock:

```bash
git clone https://github.com/ai-computing/npu-offload.git
cd npu-offload
pip install -e .            # control plane, CLI `moe-offload`
pip install -e .[model]     # + torch / transformers
moe-offload selftest        # placement, budget, reasoning split, classifier, gateway e2e (no hardware)
moe-offload worker --mock   # OpenAI-compatible worker on a mock engine
```

The gateway test in `selftest` needs `RNGD_LLMD`, the path to the RNGD llm-d gateway checkout (`rngd_llmd`, whose
`gateway/` package is imported). Set it in the environment or in `cluster.local.env` copied from
[`cluster.example.env`](cluster.example.env).

<details>
<summary><b>Build the reference engine and serve Solar-Open-100B on 2x RNGD</b></summary>
<br>

```bash
# reference engine (Solar-Open-100B on 2x RNGD)
cd track2_opt/solar_engine
cp ~/.cargo/registry/src/*/furiosa-opt-rt-0.9.0/prebuilt/* vendor/furiosa-opt-rt/prebuilt/
cargo furiosa-opt build --release --bin solar_engine
```

The engine also needs per-model inputs (RoPE tables, prompt template, placement); generate them with
`python3 solar_hybrid/export_engine_inputs.py` and `moe-offload plan`.

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
</details>

<details>
<summary><b>Python API</b></summary>
<br>

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
</details>

<details>
<summary><b>Serving endpoints and llm-d integration</b></summary>
<br>

| Endpoint (worker) | |
|---|---|
| `POST /v1/completions` | `prompt` or `prompt_token_ids`; `max_tokens` |
| `POST /v1/chat/completions` | chat template applied; `reasoning_content` split from `content` |
| `GET /metrics` | `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`, `moe_offload_expert_hit_rate`, `moe_offload_decode_ms_per_token` (deprecated → `airm_tpot_p50_ms`), per-class hit rate |
| `GET /health`, `/v1/models` | |

The gateway reuses the worker registry and endpoint picker of an llm-d style RNGD gateway (`$RNGD_LLMD`, see
`cluster.example.env`) and adds an **expert-affinity scorer**: each worker reports the hit rate of every request by
class (Korean / code / English), the gateway keeps an EWMA per (worker, class) seeded with the hit rates the worker's
placement predicts, and routes to the best `affinity − load` score. With workers carrying different placements,
requests go where their experts already live. An older Kubernetes sketch (Deployment + `InferencePool`) is in
[`moe_offload/deploy/`](moe_offload/deploy/k8s-moe-worker.yaml); the Helm chart is described under
[ai-rm integration](#-ai-rm-integration).
</details>

<details>
<summary><b>Engine configuration</b></summary>
<br>

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
</details>

## 🔌 ai-rm integration

| | |
|---|---|
| **Layer / kind** | `l2-memory` (memory tiers and data movement), `service` ([`ai-rm.yaml`](ai-rm.yaml)) |
| **Adapter** | `moe_offload.airm:serve`, entry point `airm.workloads` name `npu-offload` (`kind` `serve` or `decode`) |
| **Provides** | `MemoryTier` (per-accelerator HBM budget, expert slot counts), `Placement` (hot/cold expert placement and expected hit rates), `Metrics` |
| **Uses** | rm-core contracts (`Workload`, `MemoryTier`, `Measurement`). RNGD is driven directly through furiosa-opt; `AcceleratorProvider` is not used |
| **Metrics** | `airm_tok_s`, `airm_tpot_p50_ms`, `airm_tpot_p90_ms`, `airm_power_w{scope="card"}` |
| **KPIs** | Heterogeneous resource count (2 NPUs + host CPU in one model's decode); SLO (single-request decode latency: 28.7 ms/tok now, target ≤ 33 ms/tok) |
| **Deploy** | Helm chart [`deploy/helm/npu-offload/`](deploy/helm/npu-offload) (llm-d decode ModelService worker), not yet run on a cluster |

Expert offloading splits weights and activations across tiers rather than moving KV, so it does **not** implement
`KVTransport`.

```bash
python -m pip install -e .                  # tokenizer and profiling too: pip install -e .[model]
cp cluster.example.env cluster.local.env    # set RNGD_LLMD
python -m moe_offload.tests.run             # = moe-offload selftest; no accelerator, needs RNGD_LLMD
```

```python
from rm_core import registry, Workload
run = registry.load("airm.workloads", "npu-offload")     # no side effects (does not touch devices or the engine)
run(Workload(kind="serve", model="solar-open-100b"), engine="track2_opt/solar_engine/target/release/solar_engine",
    data="track2_opt/solar_engine/data", hf_dir=MODEL, host="0.0.0.0", port=8101, metrics_port=9400)
```

Command line: `moe-offload airm --kind serve --model solar-open-100b engine=... data=... hf_dir=... port=8101`
(values as `key=value`, JSON accepted).

<details>
<summary><b>Workload runner configuration keys</b></summary>
<br>

| Key | Default | Meaning |
|---|---|---|
| `engine`, `data`, `hf_dir` | required (not with `mock=True`) | engine binary, engine input directory, HF snapshot (tokenizer) |
| `role` | `worker` | `worker` or `gateway` (expert-affinity gateway, needs `RNGD_LLMD`) |
| `host`, `port` | `127.0.0.1`, `8101` | OpenAI API |
| `metrics_port` | none | extra port serving `/metrics` (the chart uses 9400) |
| `gateway` | `$LLMD_GATEWAY_URL` | gateway to register with; `""` disables registration |
| `wid`, `priors`, `env`, `log` | | worker id, expected hit rate per class (JSON), extra engine env (dict), engine log |
| `kv_transport` | `""` | must be empty (no KV path); any value raises `ContractError` |
| `prompt`, `max_new`, `repeat` | `"Hello"`, 32, 1 | `kind="decode"`: decode one prompt `repeat` times and return a list of `Measurement` |
| `mock`, `mock_ms`, `mock_hit` | `False` | mock engine without an NPU |
| `block` | `True` | `False` starts the server and returns immediately (for tests) |
</details>

<details>
<summary><b>Memory tiers, metrics and Helm chart</b></summary>
<br>

- **Memory tiers**: at start-up the runner logs each `rm_core.MemoryTier` (`[moe_offload.airm] memory tier {...}`) and
  exports it on `/metrics` as `moe_offload_memory_tier_capacity_bytes{tier,kind,accelerator}`. Capacities are real
  values: `device-hbm` per RNGD (`total_size` from `furiosa-smi status`, 47.5 GiB) and `host-dram` (physical memory).
  Accelerator ids are `<node>/furiosa:<i>`, where the node is `$NODE_NAME` (Kubernetes) or else the hostname.
- **Metrics** (`rm_core.metrics` names): `airm_tok_s`, `airm_tpot_p50_ms`, `airm_tpot_p90_ms` (from per-request decode
  ms/token over the last 256 requests; `tok_s` is generated tokens / decode time) and `airm_power_w{scope="card"}`
  (`furiosa-smi info`, cached for 5 s). A `kind="decode"` run prints the same names as
  `report.py --metric tok_s=... --metric tpot_p50_ms=...`. The expert hit rate is not a shared metric and stays
  `moe_offload_expert_hit_rate` (rm-core candidate: `expert_hit_pct`). `moe_offload_decode_ms_per_token` (EWMA)
  measures the same quantity as `airm_tpot_*` and is deprecated; under the PROTOCOL §6 transition rule it is kept
  only until protocol 0.2, so consumers should read `airm_tpot_p50_ms` first.
- **Helm chart**: `deploy/helm/npu-offload/` creates a decode-worker Deployment and Service labelled with the llm-d
  ModelService conventions (`llm-d.ai/inferenceServing`, `llm-d.ai/model`, `llm-d.ai/role: decode`) and selects nodes
  with the `airm.ai-computing.io/furiosa.RNGD` label. Site values such as the image repository, model volume and
  gateway address are not in `values`; they come from ai-rm `deploy/clusters/<name>/` overrides. llm-d creates the
  InferencePool by default (`llmd.inferencePool.create=true` creates it here). Without a container image the chart
  has not been run on a cluster.
</details>

<details>
<summary><b>Kubernetes + llm-d: what is still missing</b></summary>
<br>

The worker exports the vLLM-compatible gauges that the llm-d EPP collects (`vllm:num_requests_*`,
`vllm:kv_cache_usage_perc`) on `/metrics`. Still missing:

- a container image (engine binary + patched runtime + model mount)
- cluster validation of the chart (only `helm template` rendering has been checked)
- the expert-affinity scorer as an EPP plugin; today it is a separate gateway
- the gateway depends on the registry/EPP modules of `rngd_llmd` (`RNGD_LLMD`)
</details>

<details>
<summary><b>Site configuration: <code>cluster.local.env</code></b></summary>
<br>

See [`cluster.example.env`](cluster.example.env); variables already set in the environment take precedence.

| Variable | Required | Purpose |
|---|---|---|
| `RNGD_LLMD` | gateway and selftest | path to the `rngd_llmd` checkout (its `gateway/` package is imported) |
| `LLMD_GATEWAY_URL` | optional | default for `moe-offload worker --gateway` |
</details>

## 🧪 Experimental: speculative decoding

Solar-Open-100B has no MTP head and no compatible small draft model, so [`eagle3/`](eagle3) trains an EAGLE-3 draft
head from teacher features extracted by streaming the bf16 model through a GPU layer by layer (`extract.py` matches
the engine's tokens exactly). The latest draft (v3) reaches an acceptance length of τ = 1.92 (k = 6) on Solar's own
outputs, below the ≈ 2.5 needed before multi-token verification pays on this engine
([record](experiments/2026-10-16-eagle3-v3.md)). The engine does not use the draft yet.

## ⚠️ Limitations

- One decode stream per engine; requests queue in the worker. No prefix cache: prompts are processed token by token.
- The reference engine is specific to Solar-Open-100B: model dimensions and slot counts are compile-time constants of
  its kernels. The control plane accepts any routed-MoE config, but a new model needs its own data plane build.
- Accelerators are opened exclusively; other processes using the same RNGD cards block engine start-up.
- Expert-affinity routing is validated with mock workers; a single host with two cards runs one real worker.
- The Kubernetes manifests and the Helm chart have not been run on a cluster.

## 🛠 Development

```bash
moe-offload selftest                 # placement, budget, reasoning split, classifier, gateway e2e (no hardware)
cd track2_opt/solar_engine && cargo furiosa-opt build --release --bin solar_engine
REPEAT=5 ./target/release/solar_engine data_p2 32      # benchmark mode: prints tokens, ms/token, per-stage profile
```

Benchmarks are noisy on shared hosts (±10–15 %): use `REPEAT`, check the load average, and record runs in
`experiments/` using [`experiments/TEMPLATE.md`](experiments/TEMPLATE.md).

<details>
<summary><b>Repository layout</b></summary>
<br>

```
moe_offload/            control plane library (pip package, CLI `moe-offload`)
  spec.py profile.py placement.py budget.py engine.py worker.py cli.py
  airm.py               ai-rm adapter: workload runner `npu-offload`, memory tiers, airm_* metrics
  llmd/                 request classes, affinity scoring, gateway
  tests/                NPU-free tests (`moe-offload selftest`)
  deploy/               Kubernetes sketch
deploy/helm/npu-offload Helm chart (llm-d decode ModelService worker)
track2_opt/solar_engine reference data plane (Rust, furiosa-opt): engine, kernels, CPU kernels (C), patched runtime
track2_opt/solar_kernels  standalone kernel crates with CPU-emulation checks and NPU benchmarks
solar_hybrid/           Python reference model (HF-faithful) and input export
p0/                     microbenchmarks and the routing profile
eagle3/                 EAGLE-3 draft training pipeline (experimental)
experiments/            one record per measurement (setup, exact command, results)
PLAN.md, STATUS.md      design notes, decisions and running status
```
</details>

## 📚 Documentation

| Goal | Guides |
|---|---|
| 🧭 **Design and status** | [PLAN.md](PLAN.md) · [STATUS.md](STATUS.md) · [Solar engine spec](track2_opt/SOLAR_OPT_SPEC.md) |
| 📊 **Measurements** | [Experiment records](experiments) · [Vendor runtime patches](track2_opt/solar_engine/vendor/furiosa-opt-rt/VENDOR_PATCHES.md) |
| 🧩 **Kernel package** | [npu-kernels](npu-kernels) — RNGD expert FFN / GEMV generators, used by OptimusPrime's `rngd-opt` expert tier |
| 🔬 **Background analysis** | [furiosa.torch internals](analysis/furiosa_torch_internals.md) · [Vendor MoE runtime](analysis/vendor_moe_runtime.md) |
| 🔌 **ai-rm context** | [ai-rm](https://github.com/ai-computing/ai-rm) · [Optimus Prime consolidation plan](https://github.com/ai-computing/ai-rm/blob/master/docs/engine-integration.md) · [rm-core protocol](https://github.com/ai-computing/rm-core/blob/main/PROTOCOL.md) |

PLAN.md and STATUS.md are written in Korean.

## 📄 License

Not yet decided; until then this repository is for internal research use. `track2_opt/solar_engine/vendor/furiosa-opt-rt`
is a modified copy of FuriosaAI's `furiosa-opt-rt` 0.9.0 (Apache-2.0); see its
[`VENDOR_PATCHES.md`](track2_opt/solar_engine/vendor/furiosa-opt-rt/VENDOR_PATCHES.md). Model weights and datasets
are not included and keep their own licenses.

**Acknowledgements.** PolyStrata for the hot/cold expert offloading design, llm-d for the serving architecture, and FuriosaAI for the
furiosa-opt toolchain.

<div align="center"><sub>Part of <a href="https://github.com/ai-computing/ai-rm">ai-rm</a> · ai-computing team · ETRI</sub></div>
