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

The gateway reuses the worker registry and endpoint picker of an llm-d style RNGD gateway (`$RNGD_LLMD`, see `cluster.example.env`) and adds an
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

## ai-rm

**레이어**: `l2-memory` (메모리 계층·데이터 이동). **종류**: `service`.
MoE 모델의 hot expert는 NPU HBM에 두고 cold expert는 호스트 CPU에서 계산하며, PCIe로는 activation만 옮긴다.
ai-rm 안에서 NPU HBM ↔ 호스트 DRAM 사이의 expert 오프로딩(메모리 계층)을 맡는다.
구성 요소는 셋이다: 배치·예산 라이브러리(`moe_offload`), OpenAI 호환 워커(`moe-offload worker`), expert-affinity
게이트웨이(`moe-offload gateway`). 데이터 플레인은 Solar-Open-100B 전용 furiosa-opt 엔진(`track2_opt/solar_engine`)이다.

**k8s + llm-d 연동: chart는 있으나 클러스터 검증 전이다.**
- 워커는 llm-d EPP가 수집하는 vLLM 호환 게이지(`vllm:num_requests_*`, `vllm:kv_cache_usage_perc`)를 `/metrics`로 내보낸다.
  Helm chart는 `deploy/helm/npu-offload/`, 이전 초안은 `moe_offload/deploy/k8s-moe-worker.yaml`에 있다.
- 부족한 것은 다음과 같다.
  - 컨테이너 이미지가 없다(엔진 바이너리 + 패치한 런타임 + 모델 마운트).
  - chart를 클러스터에서 검증하지 않았다(`helm template` 렌더링만 확인).
  - expert-affinity 스코어러가 EPP 플러그인이 아니라 별도 게이트웨이다.
  - 게이트웨이는 비공개 `rngd_llmd`의 registry/EPP 모듈에 의존한다.

**설치·테스트** (가속기 불필요, 테스트는 `RNGD_LLMD` 필요):

```bash
python -m pip install -e .        # 토크나이저·프로파일링까지: pip install -e .[model]
cp cluster.example.env cluster.local.env   # RNGD_LLMD 경로 설정
python -m moe_offload.tests.run   # = moe-offload selftest
```

**인터페이스** (ai-rm `docs/interfaces.md`):
- 제공
  - `MemoryTier`: 가속기별 HBM 예산 산정, expert 슬롯 수 계산
  - `Placement`: 라우팅 프로파일 기반 hot/cold expert 배치와 예상 적중률
  - `Metrics`: `/metrics`의 vLLM 호환 게이지, `moe_offload_expert_hit_rate`, `moe_offload_decode_ms_per_token`
- 사용: `rm-core` 계약(`Workload`, `MemoryTier`, `Measurement`). RNGD는 furiosa-opt로 직접 다루며 `AcceleratorProvider`는 쓰지 않는다.

**rm-core 연동 (protocol v0.1)**: `moe_offload/airm.py`가 workload runner를 `airm.workloads`의 `npu-offload`로
등록한다. expert 오프로딩은 KV가 아니라 가중치·activation을 계층 사이로 나누므로 `KVTransport`는 구현하지 않는다.

```python
from rm_core import registry, Workload
run = registry.load("airm.workloads", "npu-offload")     # 부작용 없음 (장치·엔진을 건드리지 않는다)
run(Workload(kind="serve", model="solar-open-100b"), engine="track2_opt/solar_engine/target/release/solar_engine",
    data="track2_opt/solar_engine/data", hf_dir=MODEL, host="0.0.0.0", port=8101, metrics_port=9400)
```

명령줄: `moe-offload airm --kind serve --model solar-open-100b engine=... data=... hf_dir=... port=8101` (값은 `key=value`, JSON 허용).

| 설정 키 | 기본값 | 의미 |
|---|---|---|
| `engine`, `data`, `hf_dir` | 필수 (`mock=True`면 불필요) | 엔진 바이너리, 엔진 입력 디렉터리, HF 스냅샷(토크나이저) |
| `role` | `worker` | `worker` 또는 `gateway`(expert-affinity 게이트웨이, `RNGD_LLMD` 필요) |
| `host`, `port` | `127.0.0.1`, `8101` | OpenAI API |
| `metrics_port` | 없음 | `/metrics`를 추가로 내보낼 포트 (chart는 9400) |
| `gateway` | `$LLMD_GATEWAY_URL` | 등록할 게이트웨이, `""`이면 등록하지 않음 |
| `wid`, `priors`, `env`, `log` | | 워커 id, 클래스별 예상 적중률(JSON), 추가 엔진 env(dict), 엔진 로그 |
| `kv_transport` | `""` | 비어 있어야 한다(KV 경로 없음). 값이 있으면 `ContractError` |
| `prompt`, `max_new`, `repeat` | `"Hello"`, 32, 1 | `kind="decode"`: 한 프롬프트를 `repeat`번 decode하고 `Measurement` 목록을 반환 |
| `mock`, `mock_ms`, `mock_hit` | `False` | NPU 없는 모의 엔진 |
| `block` | `True` | `False`면 서버를 띄우고 바로 반환(테스트용) |

- **메모리 계층**: 시작 시 `rm_core.MemoryTier`를 로그에 남기고(`[moe_offload.airm] memory tier {...}`) `/metrics`에
  `moe_offload_memory_tier_capacity_bytes{tier,kind,accelerator}`로 낸다. 용량은 실제 값이다: RNGD마다 `device-hbm`
  (`furiosa-smi status`의 `total_size`, 47.5 GiB), `host-dram`(물리 메모리). accelerator id는 `<노드>/furiosa:<i>`이고
  노드는 `$NODE_NAME`(k8s), 없으면 호스트명이다.
- **지표** (`rm_core.metrics` 이름): `airm_tok_s`, `airm_tpot_p50_ms`, `airm_tpot_p90_ms`(최근 256개 요청의 요청별
  decode ms/token 기준, `tok_s`는 생성 토큰 / decode 시간), `airm_power_w{scope="card"}`(`furiosa-smi info`, 5초 캐시).
  `kind="decode"` 실행은 같은 이름을 `report.py --metric tok_s=... --metric tpot_p50_ms=...` 형태로 출력한다.
  expert 적중률은 공유 지표가 아니라 `moe_offload_expert_hit_rate`로 남는다(rm-core 후보: `expert_hit_pct`).
- **Helm chart**: `deploy/helm/npu-offload/`. llm-d ModelService 규약 레이블(`llm-d.ai/inferenceServing`,
  `llm-d.ai/model`, `llm-d.ai/role: decode`)을 단 decode 워커 Deployment와 Service를 만들고, 노드는
  `airm.ai-computing.io/furiosa.RNGD` 레이블로 고른다. 이미지 저장소, 모델 볼륨, 게이트웨이 주소 같은 사이트 값은
  values에 없고 ai-rm `deploy/clusters/<name>/` override로 들어온다. InferencePool은 기본적으로 llm-d가 만든다
  (`llmd.inferencePool.create=true`로 직접 만들 수 있다). 컨테이너 이미지가 아직 없어 클러스터에서 돌려 보지 않았다.

**지원 KPI**
- `이종자원수`: NPU 2장과 호스트 CPU를 한 모델 decode에 함께 쓴다.
- `SLO`: 단일 요청 decode 지연. 현재 28.7 ms/tok, 목표는 33 ms/tok 이하다.

**`cluster.local.env` 변수** (`cluster.example.env` 참고, 환경변수가 우선한다):

| 변수 | 필수 | 용도 |
|---|---|---|
| `RNGD_LLMD` | 게이트웨이·selftest | `rngd_llmd` 체크아웃 경로(`gateway/` 패키지를 import) |
| `LLMD_GATEWAY_URL` | 선택 | `moe-offload worker --gateway`의 기본값 |

## License

Not yet decided; until then this repository is for internal research use. `track2_opt/solar_engine/vendor/furiosa-opt-rt`
is a modified copy of FuriosaAI's `furiosa-opt-rt` 0.9.0 (Apache-2.0); see its
[`VENDOR_PATCHES.md`](track2_opt/solar_engine/vendor/furiosa-opt-rt/VENDOR_PATCHES.md). Model weights and datasets
are not included and keep their own licenses.

## Acknowledgements

PolyStrata for the hot/cold expert offloading design, llm-d for the serving architecture, and FuriosaAI for the
furiosa-opt toolchain.
