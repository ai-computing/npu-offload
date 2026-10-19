<div align="center">

<h1>npu-offload</h1>

<h3>가속기 메모리에 들어가지 않는 MoE 모델을 대화형 속도로 실행</h3>

<p>
  <a href="README.md">English</a> · <strong>한국어</strong>
</p>

<p>
  <a href="https://github.com/ai-computing/ai-rm"><img src="https://img.shields.io/badge/ai--rm-L2%20memory%20%26%20transport-22d3ee" alt="ai-rm 계층: L2 메모리·전송"></a>
  <a href="https://github.com/ai-computing/rm-core/blob/main/PROTOCOL.md"><img src="https://img.shields.io/badge/protocol-v0.1-7c5cff" alt="Protocol v0.1"></a>
  <a href="ai-rm.yaml"><img src="https://img.shields.io/badge/maturity-prototype-f59e0b" alt="성숙도: prototype"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&amp;logoColor=white" alt="Python 3.10+"></a>
  <a href="track2_opt/solar_engine"><img src="https://img.shields.io/badge/engine-Rust%20%C2%B7%20furiosa--opt-dea584?logo=rust&amp;logoColor=white" alt="엔진: Rust, furiosa-opt"></a>
  <a href="#-라이선스"><img src="https://img.shields.io/badge/license-undecided-64748b" alt="라이선스: 미정"></a>
</p>

<p>
  <a href="#-주요-기능"><b>주요 기능</b></a> ·
  <a href="#-결과"><b>결과</b></a> ·
  <a href="#-빠른-시작"><b>빠른 시작</b></a> ·
  <a href="#-ai-rm-연동"><b>ai-rm 연동</b></a> ·
  <a href="#-문서"><b>문서</b></a>
</p>

</div>

---

**npu-offload**는 routed-MoE 모델의 *hot* expert를 가속기 HBM에 두고 *cold* expert는 호스트 CPU에서 계산하며,
PCIe로는 activation만 옮긴다. 모델과 무관한 제어 플레인(`moe_offload`: 배치 계획, 메모리 예산, OpenAI 호환 서빙
워커, expert-affinity 게이트웨이)과 **FuriosaAI RNGD 2장(2 x 47.5 GiB HBM)에서 도는 Solar-Open-100B(bf16, 205 GB)**용
참조 데이터 플레인을 함께 제공한다. [ai-rm](https://github.com/ai-computing/ai-rm)에서는 NPU HBM과 호스트 DRAM
사이의 expert 오프로딩을 맡는 **L2 메모리·전송** 구성요소이며, workload runner `airm.workloads:npu-offload`로
등록된다.

또한 **Optimus Prime 엔진의 이전(migration) 소스**다. expert 오프로딩은 Optimus Prime 모듈로 통합하는 방향이며
([통합 계획](https://github.com/ai-computing/ai-rm/blob/master/docs/ko/engine-integration.md)), workload·메모리 계층 경계를
유지한다(`KVTransport`가 아니다). 엔진 통합이 승인되기 전까지는 이 저장소가 실행 경로다.

접근 방식은 [PolyStrata](https://huggingface.co/spaces/VecSzn/PolyStrata)(프로파일 기반 expert 배치, GPU hit / CPU
miss)를 따르고, 서빙은 [llm-d](https://github.com/llm-d/llm-d)처럼 cache affinity와 부하로 엔드포인트를 고르는
스케줄러 뒤에 워커를 두는 방식이다.

> [!NOTE]
> **프로토타입.** 참조 엔진은 Solar-Open-100B 전용이고, 수치는 decode만 다루며, Helm chart는 렌더링만 확인했고
> (컨테이너 이미지 없음), expert-affinity 라우팅은 mock 워커로 검증했다.

## ✨ 주요 기능

<table>
<tr>
<td width="33%" valign="top">
<b>정확한 decoding</b><br>
처음부터 끝까지 bf16. 확인한 프롬프트에서 출력이 호스트 레퍼런스와 일치한다(차이는 bf16 logit 동점에서만 났다).
</td>
<td width="33%" valign="top">
<b>단일 스트림 약 35 tok/s</b><br>
가중치가 전체 HBM의 2배인 100B 파라미터 MoE에서, hot expert는 RNGD 2장에, cold expert는 CPU에 둔다.
</td>
<td width="33%" valign="top">
<b>프로파일 기반 계층 배치</b><br>
라우팅 프로파일로 expert를 가속기에 나눠 배치하고, 플래너가 계층별 예상 적중률을 보고한다.
</td>
</tr>
<tr>
<td width="33%" valign="top">
<b>메모리 예산</b><br>
가속기별 예산(dense 레이어, KV cache, LM head, 여유분 → expert 슬롯)을 계산하고 rm-core 메모리 계층으로 선언한다.
</td>
<td width="33%" valign="top">
<b>OpenAI 호환 워커</b><br>
<code>/v1/completions</code>, reasoning을 분리하는 <code>/v1/chat/completions</code>, vLLM 호환 Prometheus 게이지를 제공해 기존 llm-d 스코어러가 그대로 동작한다.
</td>
<td width="33%" valign="top">
<b>Expert-affinity 라우팅</b><br>
요청마다 상주 expert가 가장 잘 맞는 워커로 보내는 게이트웨이. prefix-cache-aware 라우팅의 MoE 버전이다.
</td>
</tr>
</table>

RNGD의 호스트↔장치 전송은 MOVDIR64B posted write와 page-locked bounce buffer로 지연을 줄인다.

## 📊 결과

Solar-Open-100B, bf16, greedy decoding, 32토큰 생성, 5회 중앙값.
호스트: Xeon Silver 4410Y 2개(24코어, AMX/AVX-512), DDR5 1 TB, RNGD 2장(PCIe Gen5), FuriosaAI driver 2026.4.

| 구성 | Decode | 처리량 | 가속기 적중률 |
|---|---:|---:|---:|
| 모든 expert를 CPU에 (기준) | 435 ms/token | 2.3 tok/s | 0 |
| hot expert를 RNGD 2장에, 튜닝 (프롬프트 4개, 신규 3개) | 30.7 – 32.1 ms/token | 31.2 – 32.6 tok/s | 0.82 – 0.89 |
| + MOVDIR64B 업로드, pinned DMA 읽기 | **28.7 ms/token** | **34.9 tok/s** | 0.86 |
| 두 스트림 교차 실행 (합산) | 쌍 step당 64.9 ms | 30.8 tok/s | — |
| 근사: cold expert int8, cold이면서 순위 4 이상인 expert 생략 | 29.1 ms/token | 34.3 tok/s | top-1 일치율 92 % |

레이어당 시간: 가속기 A의 attention + shared expert 약 175 µs, 이어서 가속기 B의 hot expert(약 240 µs)와 CPU의
cold expert(약 237 µs, DRAM 대역폭 제약) 중 느린 쪽. 공유 호스트에서는 벤치마크가 ±10–15 % 흔들린다. 측정 기록은
[`experiments/`](experiments), [`PLAN.md`](PLAN.md), [`STATUS.md`](STATUS.md)에 있다.

## 🏗 구조

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

1. **Profile**: 대표 프롬프트에서 레이어마다 실제로 라우팅되는 expert를 프로파일링한다.
2. **Budget**: 가속기마다 HBM 예산을 잡고 레이어당 들어가는 expert 슬롯 수를 구한다.
3. **Place**: 가장 자주 쓰이는 expert를 그 슬롯에 배치한다(`top` 또는 `mix:<h>` 정책). Solar에서는 expert의 약
   44 %가 라우팅된 토큰의 82–89 %를 처리한다.
4. **Serve**: 엔진이 attention과 hot expert는 가속기에서, cold expert는 NUMA-local 호스트 메모리에서 AVX-512 BF16
   커널로 계산하며, 레이어마다 셋을 병렬로 실행한다.

<details>
<summary><b>요구 사항</b></summary>
<br>

| 구성 요소 | 용도 |
|---|---|
| Python ≥ 3.10 | 제어 플레인, 워커, 게이트웨이 (표준 라이브러리만 사용) |
| `torch`, `transformers` | 워커의 토크나이저, 프로파일링, HF 체크포인트 기반 계획 (`pip install .[model]`) |
| FuriosaAI RNGD x2, driver 2026.4, `cargo-furiosa-opt` 0.9 | 참조 Solar 엔진 |
| AVX-512 BF16 지원 x86-64 CPU, RAM ≥ 256 GB | 호스트의 cold expert |
| 선택: GPU | 실험용 EAGLE-3 draft head 학습 |
</details>

## 🚀 빠른 시작

가속기가 없으면 제어 플레인과 엔진 위의 모든 것이 mock으로 동작한다.

```bash
git clone https://github.com/ai-computing/npu-offload.git
cd npu-offload
pip install -e .            # 제어 플레인, CLI `moe-offload`
pip install -e .[model]     # + torch / transformers
moe-offload selftest        # 배치, 예산, reasoning 분리, 분류기, 게이트웨이 e2e (하드웨어 불필요)
moe-offload worker --mock   # mock 엔진 위의 OpenAI 호환 워커
```

`selftest`의 게이트웨이 테스트에는 `RNGD_LLMD`(RNGD llm-d 게이트웨이 체크아웃 `rngd_llmd`의 경로, 그 `gateway/`
패키지를 import한다)가 필요하다. 환경 변수로 주거나 [`cluster.example.env`](cluster.example.env)를 복사한
`cluster.local.env`에 설정한다.

<details>
<summary><b>참조 엔진 빌드와 RNGD 2장에서 Solar-Open-100B 서빙</b></summary>
<br>

```bash
# reference engine (Solar-Open-100B on 2x RNGD)
cd track2_opt/solar_engine
cp ~/.cargo/registry/src/*/furiosa-opt-rt-0.9.0/prebuilt/* vendor/furiosa-opt-rt/prebuilt/
cargo furiosa-opt build --release --bin solar_engine
```

엔진에는 모델별 입력(RoPE 표, 프롬프트 템플릿, 배치)도 필요하다. `python3 solar_hybrid/export_engine_inputs.py`와
`moe-offload plan`으로 만든다.

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
<summary><b>서빙 엔드포인트와 llm-d 연동</b></summary>
<br>

| 엔드포인트 (워커) | |
|---|---|
| `POST /v1/completions` | `prompt` 또는 `prompt_token_ids`, `max_tokens` |
| `POST /v1/chat/completions` | chat template 적용, `reasoning_content`를 `content`와 분리 |
| `GET /metrics` | `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`, `moe_offload_expert_hit_rate`, `moe_offload_decode_ms_per_token` (deprecated → `airm_tpot_p50_ms`), 클래스별 적중률 |
| `GET /health`, `/v1/models` | |

게이트웨이는 llm-d 방식 RNGD 게이트웨이(`$RNGD_LLMD`, `cluster.example.env` 참고)의 워커 registry와 endpoint
picker를 재사용하고 **expert-affinity 스코어러**를 더한다. 워커는 요청마다 클래스(한국어 / 코드 / 영어)별 적중률을
보고하고, 게이트웨이는 워커 배치가 예측한 적중률로 초기화한 (워커, 클래스)별 EWMA를 유지하며 `affinity − load`
점수가 가장 좋은 곳으로 보낸다. 워커마다 배치가 다르면 요청은 해당 expert가 이미 있는 곳으로 간다. 이전 Kubernetes
초안(Deployment + `InferencePool`)은 [`moe_offload/deploy/`](moe_offload/deploy/k8s-moe-worker.yaml)에 있고, Helm
chart는 [ai-rm 연동](#-ai-rm-연동)에서 설명한다.
</details>

<details>
<summary><b>엔진 설정</b></summary>
<br>

참조 엔진의 환경 변수 (워커는 `engine.SOLAR_ENV`의 튜닝 값을 설정한다):

| 변수 | 기본값 (튜닝) | 효과 |
|---|---|---|
| `URES` | 1 | 가속기 A가 다음 레이어 가중치를 온칩 메모리에 상주 |
| `LM_ON_B` | 1 | LM head를 가속기 B에 |
| `CHIP_CORES` / `CPU_THREADS` / `OMP_PLACES` | 7 / 17 / `{7}:17:1` | 가속기 호스트 스레드와 cold expert 스레드의 코어 분할 |
| `OMP_WAIT_POLICY` | passive | spin하는 OpenMP 스레드는 가속기 호스트 스레드를 느리게 한다 |
| `MMIO_VEC`, `MMIO_LIMIT` | 2, 32768 | 32 KiB 이하 업로드를 MOVDIR64B로 BAR 창을 통해 |
| `PIN_BOUNCE` | 1 | 작은 DMA 읽기를 page-locked 버퍼로 (`/dev/dma_heap/system` 접근 필요) |
| `SQPOLL_OFF`, `DMA_IDLE_US` | 1, 50 | DMA 제출 스레드 동작 |
| `COLD_INT8`, `TAIL_SKIP` | off | 근사 모드 (cold expert int8, 낮은 순위 cold expert 생략) |
| `NSTREAM` | 1 | 2 = decode 스트림 두 개 교차 실행 |
</details>

## 🔌 ai-rm 연동

| | |
|---|---|
| **계층 / 종류** | `l2-memory` (메모리 계층·데이터 이동), `service` ([`ai-rm.yaml`](ai-rm.yaml)) |
| **Adapter** | `moe_offload.airm:serve`, entry point `airm.workloads` 이름 `npu-offload` (`kind` `serve` 또는 `decode`) |
| **제공** | `MemoryTier`(가속기별 HBM 예산, expert 슬롯 수), `Placement`(hot/cold expert 배치와 예상 적중률), `Metrics` |
| **사용** | rm-core 계약(`Workload`, `MemoryTier`, `Measurement`). RNGD는 furiosa-opt로 직접 다루며 `AcceleratorProvider`는 쓰지 않는다 |
| **지표** | `airm_tok_s`, `airm_tpot_p50_ms`, `airm_tpot_p90_ms`, `airm_power_w{scope="card"}` |
| **KPI** | 이종자원수(NPU 2장과 호스트 CPU를 한 모델 decode에 함께 사용), SLO(단일 요청 decode 지연: 현재 28.7 ms/tok, 목표 33 ms/tok 이하) |
| **배포** | Helm chart [`deploy/helm/npu-offload/`](deploy/helm/npu-offload) (llm-d decode ModelService 워커), 클러스터 실행 전 |

expert 오프로딩은 KV가 아니라 가중치·activation을 계층 사이로 나누므로 `KVTransport`는 구현하지 **않는다**.

```bash
python -m pip install -e .                  # 토크나이저·프로파일링까지: pip install -e .[model]
cp cluster.example.env cluster.local.env    # RNGD_LLMD 경로 설정
python -m moe_offload.tests.run             # = moe-offload selftest. 가속기 불필요, RNGD_LLMD 필요
```

```python
from rm_core import registry, Workload
run = registry.load("airm.workloads", "npu-offload")     # 부작용 없음 (장치·엔진을 건드리지 않는다)
run(Workload(kind="serve", model="solar-open-100b"), engine="track2_opt/solar_engine/target/release/solar_engine",
    data="track2_opt/solar_engine/data", hf_dir=MODEL, host="0.0.0.0", port=8101, metrics_port=9400)
```

명령줄: `moe-offload airm --kind serve --model solar-open-100b engine=... data=... hf_dir=... port=8101`
(값은 `key=value`, JSON 허용).

<details>
<summary><b>Workload runner 설정 키</b></summary>
<br>

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
</details>

<details>
<summary><b>메모리 계층, 지표, Helm chart</b></summary>
<br>

- **메모리 계층**: 시작 시 `rm_core.MemoryTier`를 로그에 남기고(`[moe_offload.airm] memory tier {...}`) `/metrics`에
  `moe_offload_memory_tier_capacity_bytes{tier,kind,accelerator}`로 낸다. 용량은 실제 값이다: RNGD마다 `device-hbm`
  (`furiosa-smi status`의 `total_size`, 47.5 GiB), `host-dram`(물리 메모리). accelerator id는 `<노드>/furiosa:<i>`이고
  노드는 `$NODE_NAME`(k8s), 없으면 호스트명이다.
- **지표** (`rm_core.metrics` 이름): `airm_tok_s`, `airm_tpot_p50_ms`, `airm_tpot_p90_ms`(최근 256개 요청의 요청별
  decode ms/token 기준, `tok_s`는 생성 토큰 / decode 시간), `airm_power_w{scope="card"}`(`furiosa-smi info`, 5초 캐시).
  `kind="decode"` 실행은 같은 이름을 `report.py --metric tok_s=... --metric tpot_p50_ms=...` 형태로 출력한다.
  expert 적중률은 공유 지표가 아니라 `moe_offload_expert_hit_rate`로 남는다(rm-core 후보: `expert_hit_pct`).
  `moe_offload_decode_ms_per_token`(EWMA)은 `airm_tpot_*`와 같은 양이라 deprecated다. PROTOCOL §6 이행 규칙에 따라
  protocol 0.2까지만 유지하니 소비자는 `airm_tpot_p50_ms`를 먼저 읽는다.
- **Helm chart**: `deploy/helm/npu-offload/`. llm-d ModelService 규약 레이블(`llm-d.ai/inferenceServing`,
  `llm-d.ai/model`, `llm-d.ai/role: decode`)을 단 decode 워커 Deployment와 Service를 만들고, 노드는
  `airm.ai-computing.io/furiosa.RNGD` 레이블로 고른다. 이미지 저장소, 모델 볼륨, 게이트웨이 주소 같은 사이트 값은
  values에 없고 ai-rm `deploy/clusters/<name>/` override로 들어온다. InferencePool은 기본적으로 llm-d가 만든다
  (`llmd.inferencePool.create=true`로 직접 만들 수 있다). 컨테이너 이미지가 아직 없어 클러스터에서 돌려 보지 않았다.
</details>

<details>
<summary><b>Kubernetes + llm-d: 아직 부족한 것</b></summary>
<br>

워커는 llm-d EPP가 수집하는 vLLM 호환 게이지(`vllm:num_requests_*`, `vllm:kv_cache_usage_perc`)를 `/metrics`로
내보낸다. 부족한 것은 다음과 같다.

- 컨테이너 이미지가 없다(엔진 바이너리 + 패치한 런타임 + 모델 마운트).
- chart를 클러스터에서 검증하지 않았다(`helm template` 렌더링만 확인).
- expert-affinity 스코어러가 EPP 플러그인이 아니라 별도 게이트웨이다.
- 게이트웨이는 `rngd_llmd`(`RNGD_LLMD`)의 registry/EPP 모듈에 의존한다.
</details>

<details>
<summary><b>사이트 설정: <code>cluster.local.env</code></b></summary>
<br>

[`cluster.example.env`](cluster.example.env) 참고. 이미 설정된 환경 변수가 우선한다.

| 변수 | 필수 | 용도 |
|---|---|---|
| `RNGD_LLMD` | 게이트웨이·selftest | `rngd_llmd` 체크아웃 경로(`gateway/` 패키지를 import) |
| `LLMD_GATEWAY_URL` | 선택 | `moe-offload worker --gateway`의 기본값 |
</details>

## 🧪 실험: speculative decoding

Solar-Open-100B에는 MTP head도, 호환되는 작은 draft 모델도 없어서 [`eagle3/`](eagle3)에서 bf16 모델을 GPU로 레이어별
스트리밍해 추출한 teacher feature로 EAGLE-3 draft head를 학습한다(`extract.py`는 엔진 토큰과 정확히 일치한다).
최신 draft(v3)의 Solar 자체 출력에 대한 acceptance length는 τ = 1.92(k = 6)로, 이 엔진에서 multi-token 검증이
이득이 되는 약 2.5에 못 미친다([기록](experiments/2026-10-16-eagle3-v3.md)). 엔진은 아직 draft를 쓰지 않는다.

## ⚠️ 한계

- 엔진당 decode 스트림은 하나이며 요청은 워커에서 대기한다. prefix cache가 없어 프롬프트를 토큰 단위로 처리한다.
- 참조 엔진은 Solar-Open-100B 전용이다. 모델 차원과 슬롯 수가 커널의 컴파일 시간 상수다. 제어 플레인은 어떤
  routed-MoE 설정도 받지만, 새 모델에는 별도의 데이터 플레인 빌드가 필요하다.
- 가속기를 배타적으로 연다. 같은 RNGD 카드를 쓰는 다른 프로세스가 있으면 엔진이 시작하지 못한다.
- expert-affinity 라우팅은 mock 워커로 검증했다. 카드 2장인 호스트 하나에서는 실제 워커 하나만 돈다.
- Kubernetes 매니페스트와 Helm chart는 클러스터에서 실행해 보지 않았다.

## 🛠 개발

```bash
moe-offload selftest                 # placement, budget, reasoning split, classifier, gateway e2e (no hardware)
cd track2_opt/solar_engine && cargo furiosa-opt build --release --bin solar_engine
REPEAT=5 ./target/release/solar_engine data_p2 32      # benchmark mode: prints tokens, ms/token, per-stage profile
```

공유 호스트에서는 벤치마크가 ±10–15 % 흔들린다. `REPEAT`를 쓰고 load average를 확인하며,
[`experiments/TEMPLATE.md`](experiments/TEMPLATE.md)로 `experiments/`에 기록한다.

<details>
<summary><b>저장소 구성</b></summary>
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

## 📚 문서

| 목적 | 안내 |
|---|---|
| 🧭 **설계와 현황** | [PLAN.md](PLAN.md) · [STATUS.md](STATUS.md) · [Solar 엔진 spec](track2_opt/SOLAR_OPT_SPEC.md) |
| 📊 **측정** | [실험 기록](experiments) · [Vendor 런타임 패치](track2_opt/solar_engine/vendor/furiosa-opt-rt/VENDOR_PATCHES.md) |
| 🔬 **배경 분석** | [furiosa.torch 내부 구조](analysis/furiosa_torch_internals.md) · [Vendor MoE 런타임](analysis/vendor_moe_runtime.md) |
| 🔌 **ai-rm 맥락** | [ai-rm](https://github.com/ai-computing/ai-rm) · [Optimus Prime 통합 계획](https://github.com/ai-computing/ai-rm/blob/master/docs/ko/engine-integration.md) · [rm-core protocol](https://github.com/ai-computing/rm-core/blob/main/PROTOCOL.md) |

실험 기록, Solar 엔진 spec, 분석 문서는 영어로 작성되어 있다.

## 📄 라이선스

아직 정하지 않았으며, 그때까지 이 저장소는 내부 연구용이다. `track2_opt/solar_engine/vendor/furiosa-opt-rt`는
FuriosaAI `furiosa-opt-rt` 0.9.0(Apache-2.0)을 수정한 사본이다.
[`VENDOR_PATCHES.md`](track2_opt/solar_engine/vendor/furiosa-opt-rt/VENDOR_PATCHES.md)를 참고한다. 모델 가중치와
데이터셋은 포함하지 않으며 각자의 라이선스를 따른다.

**감사의 말.** hot/cold expert 오프로딩 설계의 PolyStrata, 서빙 구조의 llm-d, furiosa-opt 툴체인의 FuriosaAI에
감사한다.

<div align="center"><sub><a href="https://github.com/ai-computing/ai-rm">ai-rm</a>의 구성요소 · ai-computing 팀 · ETRI</sub></div>
