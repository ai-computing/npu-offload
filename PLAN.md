# npu_offload — PolyStrata 방식 MoE 오프로드를 Furiosa RNGD로

작성 2026-10-15. 참고 코드: `ref/PolyStrata-src` (HF Space VecSzn/PolyStrata, source 브랜치, 0.1.43.1-poly.2).

## 1. PolyStrata 핵심 (src/poly)
- 비-expert 가중치(attn, router, shared expert, norm, draft 블록)와 KV는 항상 GPU에 상주한다.
- Routed expert는 남은 VRAM에 expert 단위로 캐시한다. 선택 기준은 `(profile count)/bytes` 그리디 knapsack (`engine.cpp:926-1019`).
- Decode, 레이어마다:
  1. GPU가 attn과 router를 계산하고, router logits와 activation을 pinned host로 보낸다 (약 9KB).
  2. host가 top-k를 계산한다.
  3. hit expert는 GPU에서 8개씩 묶어 배치 GEMV로 계산한다. miss expert는 CPU(ggml vec_dot)가 동시에 계산한다.
  4. CPU 결과를 GPU로 axpy한다 (`engine.cpp:1786-1901`).
  - PCIe로는 activation만 오간다. 비용은 대역폭이 아니라 레이어당 1회 왕복의 지연이다.
  - 토큰 윈도우를 두 half로 나눠 GPU와 CPU를 파이프라인한다 (`run()` `:2316`).
- Prefill (32토큰 이상): 모든 expert를 GPU에서 계산한다.
  - miss expert의 가중치는 PCIe로 staging한다. routing 이전에 profile 순서로 미리 업로드한다.
  - 프롬프트는 layer-major로 읽는다. 레이어당 1회만 업로드하고 여러 청크가 재사용한다 (`read()` `:2515`).
- 부가 기법: decode 중 LFU-decay로 swap, MTP와 n-gram 투기 디코딩, tail-skip, NUMA 핀닝과 pre-fault.
- 서빙은 single-stream이다. 요청은 대기열에서 차례를 기다린다.

## 2. 현재 서버 (2026-10-15 실측)
| 항목 | 값 | 함의 |
|---|---|---|
| NPU | RNGD **2장** (npu0, npu1), 47.5GiB씩 = 95GB, 둘 다 idle | 이전 기록의 4장에서 줄었다 |
| PCIe | Gen5 x16 (이론 64GB/s). 실측 H2D는 없다 (과거 추정 3–5GB/s) | **P0에서 측정 필수** |
| CPU | Xeon Silver 4410Y ×2, 24코어. AMX-BF16/INT8, AVX512-BF16/VNNI/VBMI. SNC4라 8 NUMA × 2–4코어 | cold expert 계산은 메모리 대역폭에 묶인다 (~150–200GB/s/소켓) |
| RAM | 1TB | 200GB 이상 모델의 expert를 전부 담을 수 있다 |
| SW | furiosa-llm 2026.3, furiosa-torch 2026.2, transformers 5.1, gguf 0.18 | llama.cpp는 없다 |
| 벤더 MoE | Qwen3-MoE, GPT-OSS, EXAONE-MoE 지원. 가중치는 전부 HBM 상주, offload 없음, EP 없음(TP만) | `furiosa-llm serve`에는 끼어들 수 없다 |
| 재사용 가능 | `new_atten_kernel`: ctypes `Kernel.run(raw HBM addr)`, `CompileModule` weight-as-input, set_fusion/mesh, `generic_runtime*`, `generic_server`, scheduler | 커스텀 런타임 위에 구축한다 |
| 측정치 | graph launch 122–140µs, 소형 matmul 137–250µs | 레이어당 왕복 비용이 decode 속도를 결정한다 |

로컬 MoE 모델:
| 모델 | 크기 | 구조 |
|---|---|---|
| Qwen3-30B-A3B bf16 | 57GB | 48L, 128E/top8, ffn 768. 벤더 지원 |
| Qwen3.6-35B-A3B | 67GB | |
| gemma-4-26B-A4B | 49GB | |
| Solar-Open-100B NVFP4 | 63GB | 48L, 128E+1 shared/top8 |

로컬 모델은 모두 95GB HBM에 들어간다. 오프로드가 실제로 필요한 대상은 200GB 이상이다. 후보는 Qwen3-235B-A22B와 Solar-100B bf16.

## 3. 포팅의 핵심 난점
1. **AOT 정적 shape**: 토큰마다 다른 expert를 고르는 것을 그래프로 표현해야 한다.
   - 해법 A: host에서 top-k를 하고, raw-addr로 weight 포인터를 교체한다. 컴파일된 expert-FFN 커널 하나를 재사용하므로 gather가 필요 없다.
   - 해법 B: device에서 top-k를 하고, 상주 hot 테이블 `[E_hot,F,D]`에 remap id로 gather한다. 벤더 `TokenWiseMoEOp`의 `w1[expert_ids]`를 템플릿으로 쓴다.
2. **레이어당 host 왕복**: 48레이어 × (launch + D2H + H2D) ≈ 15–25ms/tok 하한. 레이어 경계마다 그래프 1개로 융합하고 ping-pong 파이프라인을 써야 한다.
3. **vendor 엔진이 닫혀 있다**: 가중치와 KV를 우리가 소유하는 커스텀 런타임이 필수다. PD-disagg 작업에서 이미 입증했다.
4. **기존 hbm_tensor 할당기 한계**: 약 12GB까지만 쓰고 free가 없다. 47GB 상주를 하려면 확장하거나 fusion/mesh 경로를 써야 한다.

## 4. 단계별 계획
**P0. 타당성 마이크로벤치 (Go/No-Go, 2–3일)**
- H2D/D2H 대역폭과 지연: 4KB–1GB, 카드별, NUMA 로컬/리모트.
- 작은 그래프 왕복 지연: 입력 쓰기 → launch → 출력 읽기.
- NPU expert-FFN 시간 (Qwen3-30B 기준 2048×768):
  - 1개 단독, 같은 launch 안의 k개 배치, raw-addr 포인터 교체 방식과 CompileModule 방식 비교.
- CPU expert-FFN 시간: torch/oneDNN AMX bf16·int8, batch 1–64, NUMA 핀닝.
- Expert 활성 분포 프로파일: Qwen3-30B(CPU에서 HF hook). 예산별 hit-rate 곡선을 얻는다.
- **Go 기준**: 레이어당 왕복이 0.3ms 이하이고, 예산 50%에서 hit-rate가 70% 이상.

**P1. 레퍼런스 오라클**
- generic_runtime에 Qwen3-MoE를 추가한다: router, top-k, norm_topk_prob, (shared expert).
- HF teacher-forced argmax와 일치하는지 검증한다.

**P2. 정적 하이브리드 decode (PolyStrata 충실 이식, 해법 A)**
- profile knapsack으로 hot expert를 HBM에 상주시킨다. 개발용으로 Qwen3-30B에 **인위적 HBM 예산**을 25/50/75/100%로 준다.
- 레이어 그래프는 attn, router, shared expert로 구성한다.
- host가 top-k를 하고, NPU의 hot expert 배치 launch와 CPU의 cold expert 계산을 병렬로 돌린다.
- 오라클과 정확도를 맞추고, 예산별 tok/s를 측정한다.
- 100% 예산 결과는 vendor furiosa-llm Qwen3-30B 대비 기준선이 된다.

**P3. NPU 친화 최적화 (해법 B, 차별점)**
- 레이어 경계를 그래프 1개로 융합한다: `[이전 레이어 CPU 결과 add → tail → 다음 레이어 head/router/top-k → hot gather FFN]`. launch는 레이어당 1회로 줄어든다.
- 두 micro-batch ping-pong: NPU가 A를 계산하는 동안 CPU가 B의 cold expert를 계산한다. 이는 poly halves 기법의 배치판이다.
- **동시 요청 배치**: RNGD는 c=64–128에서 효율이 정점이었다. 반면 poly는 single-stream이다.
  - 배치가 커지면 expert 합집합이 커져 CPU 부하가 늘어난다.
  - 따라서 배치 크기, hit-rate, CPU 대역폭 사이의 균형점을 찾는다.

**P4. Prefill**
- 비교 대상: (a) layer-major 가중치 스트리밍 (staging 버퍼 고정 크기, routing 전 선업로드) vs (b) CPU AMX로 cold expert 계산.
- P0의 실측 PCIe 대역폭으로 교차점을 결정한다.

**P5. 대형 모델 + 2카드 + 서빙**
- 타깃: 200GB 이상 MoE (Qwen3-235B-A22B 또는 Solar-100B bf16), 다운로드 필요, 디스크 여유 616GB.
- 2카드 layer-split(PP)로 hot 캐시 용량을 2배로 한다.
- generic_server(OpenAI)와 /metrics에 붙인다.
- 비교 대상: CPU-only, 정적 vs 동적 배치, 이론 상한.

**P6. (선택)**
- decode 중 LFU-decay expert swap: H2D 비동기, slot 주소 교체.
- MTP·n-gram 투기 디코딩, tail-skip.

## 5. 결정 필요
- 최종 타깃 모델 (P5): 235B FP8 vs Solar-100B bf16.
- 진행 순서: P2 우선(빠른 E2E) vs P3 우선(NPU 최적 구조부터).

## 6. NPU 구현 가능성 검토 (2026-10-15, 대상: Solar-Open-100B bf16)
실측 결과는 `p0/results/*.jsonl`에 있다.
- **상주 용량**: furiosa.torch(fusion8/mesh2)로 카드 1장에 expert 1450개(45.6GB)를 올렸다. 어떤 expert를 호출해도 지연은 0.35ms로 같다. 따라서 2장이면 약 2500개(41%)가 hot으로 상주할 수 있다.
- **expert 교체**: 그래프 입력으로 넘기는 가중치를 바꾸면 재컴파일 없이 동작한다. 오차는 0.5% 이하다.
- **device gather**: `index_select`와 `W[ids]` 모두 `UnsupportedOpError`가 난다. furiosa.torch 경로에서는 host가 top-k를 고르고 슬롯에 포인터를 넘기는 방식만 가능하다.
- **버킷 (T 토큰 × K 슬롯, masked dense, 왕복 포함)**: T=1/8/16에서 시간은 거의 같다. K=8/16/32에서 1.0/1.3/2.1ms다. 고정비 약 0.7ms에 expert당 약 0.04ms가 더해진다. → **T 버킷은 {1,8,16}, K 버킷은 {8,16,32}로 둔다.** 그래프는 48개 레이어가 공유하므로 총 9개, 컴파일은 수 분이면 된다.
- **dense 부분**: 레이어당 0.6ms다. 레이어당 왕복 하한은 0.5ms다.
- **SDK 2026.4 문서**: furiosa-torch API 문서가 없다(eager 모드는 2026 Q4 예정). offload는 KV에만 있고 weight offload나 EP는 없다. Solar 공식 지원은 NVFP4를 4장에 올리는 구성뿐이다.
  - 현재 버전 조합(driver 2026.4, llm 2026.3, torch 2026.2)은 지원 조합이 아니다. 업그레이드를 검토한다.
- **furiosa-opt 0.9 (2026-10-07, 로컬 cargo-furiosa-opt 0.9.0 설치됨)**: 다음 기능이 생겼다.
  - 영속 HBM 버퍼
  - byte offset이나 position으로 하는 런타임 gather (DM 256MB에 맞춰 타일링 필요)
  - `if e < active` 형태의 런타임 가드
  - 멀티칩 collective
  - 비동기 launch (목표치는 수 µs)

  한계: Python 바인딩이 없고, 같은 칩에서 furiosa.torch와 공존할 수 없으며, 모든 커널을 Rust DSL로 작성해야 한다.

### 두 트랙
- **트랙 1 (지금 가능)**: furiosa.torch로 구현한다.
  - 레이어당 그래프: H_i = [이전 expert 합 + residual → attn_i → router_i] 와 E_i = [hit 슬롯 K 버킷 FFN].
  - E_i를 비동기로 실행하는 동안 CPU가 miss expert를 계산해 겹친다.
  - 예상 성능은 batch 1에서 약 10–15 tok/s다. batch를 키우면 NPU 비용은 거의 그대로이므로 처리량이 늘어난다.
- **트랙 2 (상한 탐색 spike)**: furiosa-opt 0.9로 구현한다.
  - 상주 expert 테이블 + offset gather + FFN 커널 1개.
  - 측정 항목: launch 지연, GEMV 대역폭.
  - 측정 결과가 좋으면 expert 경로를 칩 단위로 분리해 이식한다. 예: chip1은 furiosa-opt expert 전용, chip0은 furiosa.torch dense.

## 7. furiosa-opt 엔진 (목표 30 tok/s, 2026-10-15)
`track2_opt/solar_engine` — Rust 단일 프로세스. 칩마다 OS 스레드 + 전용 tokio 런타임을 두고, CPU cold expert는 C 커널(AVX-512 BF16, OpenMP)로 처리한다.

**칩 배치**
- 칩 A
  - attn_block_cK (`solar_kernels/attn`): rmsnorm, qkv, RoPE, KV 기록, attention, o_proj, post-norm을 launch 1회로 처리. 160–190µs.
  - shared_router: 42–57µs.
  - combine: 24µs.
  - lm_head_am: 1.35ms.
  - expert 테이블: 24개 × 48레이어.
- 칩 B: expert 테이블 32개 × 48레이어 (ffn32). 결과는 장비에서 combine한 뒤 내려받는다.
- CPU: cold expert 계산 (ck_moe3). 가중치는 NUMA first-touch로 배치한다.

**decode 속도 변천 (bf16, 출력 토큰이 Python/HF 결과와 일치)**

| 구성 | decode | tok/s |
|---|---|---|
| 전부 host | 435ms | 2.3 |
| expert만 칩 | 220ms | 4.6 |
| dense를 칩으로 | 101.6ms | 9.8 |
| 스레드·코어 분리 | 56ms | 17.8 |
| CPU 커널 v2 + passive OMP | 51.6ms | 19.4 |
| NUMA 배치 | **48.5ms** | **20.6** |

**레이어당 시간 (최근 측정)**
- layerA 0.41ms: combine 업로드 53µs, attn 190µs, shared 57µs, logits·h2 다운로드 65µs.
- cold CPU 0.39–0.43ms.
- 칩 B 대기 0.1ms.

**진행 중**
- 칩 A 통합 커널: combine, attn, shared/router를 launch 1회로 묶고 입출력을 각각 1회 전송으로 줄인다.
- 칩 B 통합 커널과 슬롯 교체.

**옵션**
- COLD_INT8=1 (근사): activation까지 int8로 양자화. 1.6배 빠르고 오차 2.2%.
- 가중치만 int8로 하는 방식: 오차 0.7%이지만 속도 이득이 없다.

**시도했지만 효과 없음**
- 실시간 expert 교체: 47토큰 trace 시뮬레이션에서 적중률 0.845 → 0.846.
- OMP active 대기: 칩 host 쪽 작업이 느려진다.
- 칩 코어를 4개 이하로 축소: 크게 느려진다.

**측정 주의**
- 이 서버의 다른 Kubernetes 파드(vLLM, npu_lora 학습)가 CPU 경합을 일으켜 수치가 ±15% 흔들린다.

### 7.1 목표 달성 (30 tok/s 이상)
측정 조건은 REPEAT=5 중앙값, 출력 32토큰, 프롬프트 82토큰이다.

| 모드 | ms/tok | tok/s | teacher-forced 일치율 (정확 모드 대비, 64토큰) |
|---|---|---|---|
| 정확 bf16 | 40–45 | 22–25 | 100% |
| COLD_INT8=1 | 35.3 | 28.3 | 95.3% |
| COLD_INT8=1 TAIL_SKIP=5 | 31.2 | **32.1** | 90.6% |
| COLD_INT8=1 TAIL_SKIP=4 | 29.1 | **34.3** | 92.2% |

참고로, 반올림 경로만 다른 정확 모드 변형끼리의 일치율은 96.9%다.

**마지막 단계의 핵심**
- 칩 A의 attention 커널과 expert 커널을 한 프로그램으로 합쳤다. 실행 중 모드 값으로 분기한다 (`unified_a_cK`).
- 프로그램 전환 비용이 사라졌다.

**실행 방법**
```
CHIP_CORES=10 CPU_THREADS=14 OMP_PLACES="{10}:14:1" OMP_WAIT_POLICY=passive OMP_PROC_BIND=close FURIOSA_VISIBLE_CHIPS=0,1 COLD_INT8=1 TAIL_SKIP=4 ./target/release/solar_engine data 32
```

### 7.2 정확 모드 (bf16)

| 구성 | 결과 |
|---|---|
| 단일 요청 | 40.1ms/tok (24.9 tok/s) |
| 동시 2요청 (NSTREAM=2, 레이어 단위 교차) | 합산 **30.8 tok/s** (요청당 15.4 tok/s). 출력이 단일 요청과 같음 |

**설정:** `OMP_WAIT_POLICY=passive GOMP_SPINCOUNT=100000 CHIP_CORES=10 CPU_THREADS=14 OMP_PLACES="{10}:14:1"`

**단일 요청 정확 모드의 한계 분석**
- 레이어당 시간은 칩 A 단계 약 0.31ms와 CPU cold 약 0.43ms가 직렬로 이어진다. cold는 레이어당 약 2개 미적중 expert를 bf16 63MB로 계산한다.
- 전체를 메모리 대역폭으로 나눈 이론 하한은 약 35ms이다.
- CPU 대역폭은 칩 런타임의 polling 스레드(코어 8–10개)와 나눠 쓴다.

**효과가 없었던 시도**
- CPU 캐시 프리패치: pre-gating 예측 정밀도는 84%였지만, 스레드당 몫(약 4.5MB)이 L2 캐시(2MB)를 넘어 효과가 없었다.

### 7.3 정확 모드 단일 요청 30 tok/s 달성 (2026-10-16)

**구성:** URES=1 (칩 A unified_res: DM 상주 가중치 + 다음 레이어 prefetch를 같은 프로그램에서), LM_ON_B=1 (lm_head를 칩 B로), 배치 mix4 (칩 A = 랭크 0–3, 36–55 / 칩 B = 랭크 4–35), 패치한 런타임 `vendor/furiosa-opt-rt` (SQPOLL_OFF=1로 io_uring sqpoll 끔, DMA_IDLE_US=50으로 worker idle spin 조정).

```
SQPOLL_OFF=1 DMA_IDLE_US=50 URES=1 LM_ON_B=1 OMP_WAIT_POLICY=passive GOMP_SPINCOUNT=100000 \
CHIP_CORES=7 CPU_THREADS=17 OMP_PLACES="{7}:17:1" OMP_PROC_BIND=close REPEAT=5 \
FURIOSA_VISIBLE_CHIPS=0,1 ./target/release/solar_engine <data_dir> 32
```

| 프롬프트 (32토큰 실제 생성, 5라운드 중앙값) | decode | tok/s | 적중률 |
|---|---|---|---|
| data (원래, 프로파일에 포함) | 32.1ms | 31.2 | 0.815 |
| data_p1 (한국어 신규) | 30.9ms | 32.3 | 0.889 |
| data_p2 (영어 신규, 이야기) | 30.7ms | 32.6 | 0.856 |
| data_p3 (영어 신규, TCP vs UDP) | 31.2ms | 32.1 | 0.872 |

- 고정 토큰열(FORCE, 원래 프롬프트)에서는 34.2ms (29.2 tok/s)로 경계선이다. 생성 내용에 따라 적중률이 달라진다.
- 정확성: p2는 호스트 레퍼런스(CHIPS=0)와 8토큰 모두 일치. p1은 5번째 토큰에서 갈라지지만 Python 레퍼런스에서 후보 4121과 4380의 로짓이 bf16로 30.375 동점이다(합산 순서 차이). 원래 프롬프트의 13번째 토큰 분기도 같은 종류다.
- 남은 여유 후보: lm_head 두 칩 분할(~0.65ms 추정), Token 명령 업로드 최적화, pinned 메모리(`/dev/dma_heap/system` 권한 필요).

### 7.4 PCIe 전송 지연 줄이기 (2026-10-16)
- **채택: MOVDIR64B 업로드** (`MMIO_VEC=2 MMIO_LIMIT=32768`, vendor 런타임 패치). 런타임은 2KB 이하 쓰기만 BAR4 창에 8바이트 store로 쓰고 나머지는 DMA(고정 약 25µs)를 썼다. 64바이트 MOVDIR64B로 바꾸고 32KB까지 창으로 보내니 칩 A 업로드 39→21µs, 칩 B 77→50µs, decode 31.6→약 30.0ms (p2, 약 33 tok/s). 토큰 동일. AVX-512 일반 store는 오히려 느림(UC 메모리에서 분할).
- **효과 없음: BAR 창 직접 읽기** (`MMIO_RT=17 MMIO_RLIMIT=16384`). UC 64B 로드는 코어당 약 0.7µs씩 직렬이지만 여러 코어로 나누면 겹친다(8KB: 1스레드 88µs, 8스레드 11µs). 칩 스레드에서 별도 OMP 팀을 띄우면 cold 계산과 코어를 다퉈 10배 이상 느려졌다. 메인 스레드가 기존 OMP 팀으로 대신 읽게 하니 정확하고 L.dl 54→약 40µs였지만 종단 시간은 차이 없음(교차 A/B 측정 30.3/30.5 vs 30.1/30.2ms). 기본값 off.
- 결론: 레이어당 임계 경로 = 칩 A 레이어(장치 약 175µs) + max(칩 B 약 240µs, CPU cold 약 237µs, 메모리 대역폭 한계). 남은 후보는 lm_head 두 칩 분할(토큰당 약 0.5ms 추정) 정도다.
