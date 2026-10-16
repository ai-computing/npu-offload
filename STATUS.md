# STATUS — NPU Offload

<!-- research-hub:header:begin (maintained by .research/report.py; keep the "- **Key**: value" format) -->
- **State**: active
- **Updated**: 2026-10-16 13:42+09:00
- **Agent**: claude@etri-ESC8000-E11
- **Task**: Speculative decoding for Solar
- **Last result**: EAGLE-3 v1 (2.78M chat tokens): tau 1.42 on Solar's own generations (too low); v2 online training on 20.3M tokens with reasoning traces running on GPU1
- **Next**: eval v2 checkpoints; if tau >= 2.5 build T>1 verify kernels
- **Blockers**: none
<!-- research-hub:header:end -->

## Goal

PolyStrata 방식 MoE 오프로드를 Furiosa RNGD에 이식한다. hot expert는 NPU HBM에 상주시키고 cold expert는 호스트 CPU에서 계산하며, PCIe로는 activation만 오간다. 대상은 HBM(RNGD 2장, 95GB)에 다 들어가지 않는 200GB 이상 MoE 모델이며, 현재는 Solar-Open-100B bf16(205GB)이다. 목표는 2× RNGD에서 decode 30 tok/s 이상(33 ms/tok 이하)이다 (`PLAN.md` §1–2, §6, `track2_opt/SOLAR_OPT_SPEC.md`).
- 참고: 이 디렉터리는 git 저장소가 아니다(`git log` 실패). 아래 근거는 모두 파일 경로로 단다.

## Findings

<!-- Durable conclusions with numbers and a pointer to the experiment record. Newest first. -->
- **PCIe 업로드 지연 단축 (2026-10-16)**: vendor 런타임을 패치해 MOVDIR64B로 BAR4 창에 쓴다(`MMIO_VEC=2 MMIO_LIMIT=32768`). 칩 A 업로드 39→21µs, 칩 B 77→50µs, decode 31.6→약 30.0ms(p2, 약 33 tok/s)이며 토큰은 동일하다. BAR 창 직접 읽기(`MMIO_RT`)는 정확했지만 종단 시간 차이는 없었다(30.3/30.5 vs 30.1/30.2ms). 레이어당 임계 경로는 칩 A 약 175µs + max(칩 B 약 240µs, CPU cold 약 237µs)다 (`PLAN.md` §7.4).
- **정확 bf16 단일 요청으로 30 tok/s 달성 (2026-10-16)**: `URES=1 LM_ON_B=1` + mix4 배치 + 패치한 `vendor/furiosa-opt-rt`(SQPOLL_OFF=1, DMA_IDLE_US=50) 구성이다. 32토큰 생성, REPEAT=5 중앙값 기준 프롬프트 4개에서 30.7–32.1 ms/tok(31.2–32.6 tok/s), 적중률 0.815–0.889다. 신규 프롬프트 3개 포함. 고정 토큰열(FORCE)에서는 34.2ms(29.2 tok/s)로 경계선이다. 호스트 레퍼런스와의 토큰 분기는 bf16 로짓 동점(30.375)으로 확인했다 (`PLAN.md` §7.3, `track2_opt/solar_engine/data*`).
- **정확 모드 동시 2요청 (2026-10-16)**: NSTREAM=2로 합산 30.8 tok/s(요청당 15.4 tok/s)이며 출력은 단일 요청과 같다. 당시 단일 요청 정확 모드는 40.1ms(24.9 tok/s)였다 (`PLAN.md` §7.2).
- **근사 모드 (2026-10-15)**: `COLD_INT8=1 TAIL_SKIP=4`에서 29.1 ms/tok(34.3 tok/s)이며 teacher-forced 일치율은 92.2%다. `TAIL_SKIP=5`는 32.1 tok/s(90.6%), `COLD_INT8=1`만 쓰면 28.3 tok/s(95.3%)다. 칩 A 커널을 한 프로그램(`unified_a_cK`)으로 합쳐 프로그램 전환 비용을 없앴다 (`PLAN.md` §7.1, `track2_opt/solar_kernels/unified_a`).
- **furiosa-opt 엔진 진척 (2026-10-15)**: 단계별 decode 시간은 다음과 같다. 출력은 Python/HF와 일치한다 (`PLAN.md` §7, `track2_opt/solar_engine`).
  - 전부 host: 435ms(2.3 tok/s)
  - expert만 칩: 220ms
  - dense를 칩으로: 101.6ms
  - 스레드·코어 분리: 56ms
  - CPU 커널 v2: 51.6ms
  - NUMA 배치: 48.5ms(20.6 tok/s)
- **효과 없었던 시도**: 실시간 expert 교체(47토큰 trace 적중률 0.845→0.846), OMP active 대기, 칩 코어 4개 이하, CPU 캐시 프리패치(pre-gating 정밀도 84%이나 스레드당 약 4.5MB > L2 2MB) (`PLAN.md` §7, §7.2).
- **트랙 1 (furiosa.torch 하이브리드) 상한이 낮다 (2026-10-15)**: 48L 전체 decode는 4132 ms/tok(적중률 0.812)이다. expert_srv와 프로파일링 버전은 381 ms/tok(적중률 0.877, 15토큰)이다. 반복 실행 중 `DRAM_SHARED` OOM이 났다. 이후 트랙 2로 전환했다 (`logs/full_gather_swap2.log`, `logs/prof_srv.log`, `logs/hybrid_full_v2.log`, `logs/prof7.log`).
- **furiosa-opt 0.9 spike (2026-10-15)**: launch 13µs(furiosa.torch 약 250µs 대비)다. K=8 expert FFN은 on-device gather 포함 p50 247.4µs(1017 GB/s), K=4 guarded는 141.2µs(891 GB/s)다 (`track2_opt/k8_bench_v3.log`).
- **P0 마이크로벤치 (2026-10-15, furiosa.torch)**
  - 카드 1장에 expert 1450개(45.6GB)를 상주시켜도 호출 지연은 0.35ms로 같다.
  - 버킷 K=8/16/32의 시간은 1.0/1.3/2.1ms(왕복 포함)다. T=1/8/16은 시간이 거의 같다.
  - dense 레이어는 0.6ms, 왕복 하한은 0.5ms다.
  - device gather(`index_select`, `W[ids]`)는 `UnsupportedOpError`다.

  (`PLAN.md` §6, `p0/results/npu_microbench.jsonl`, `p0/results/npu_moe_forms.jsonl`)
- **Expert 활성 분포 (Solar, 12 프롬프트, decode 564토큰)**: prefill+decode 프로파일 기준 예산 30/39/50/75%에서 적중률 0.725/0.801/0.868/0.967이다. 레이어 8개 전부 적중 비율은 예산 50%에서 0.406이다 (`p0/results/routes_analysis.json`).

## Open questions

- **Prefill(P4)이 없다**: layer-major 가중치 스트리밍과 CPU AMX cold 계산 중 무엇이 나은지 정하지 않았다. 현재 수치는 decode만 다룬다 (`PLAN.md` §4 P4).
- **배치·서빙(P3/P5)**: NSTREAM=2를 넘어서는 배치를 하면 expert 합집합이 커져 CPU 부하가 는다. 배치 8/16/64에서 레이어당 합집합 expert는 평균 35.1/51.9/85.6개다(`p0/results/routes_analysis.json` union). 배치 크기, 적중률, CPU 대역폭 사이의 균형점을 찾아야 한다. generic_server(OpenAI API)와의 연동도 아직 없다.
- **30 tok/s 여유가 작다**: FORCE 34.2ms가 경계선이다. 남은 후보는 lm_head 두 칩 분할(토큰당 약 0.5–0.65ms 추정), Token 명령 업로드 최적화, pinned 메모리(`/dev/dma_heap/system` 권한 필요)다 (`PLAN.md` §7.3–7.4).
- **최종 타깃 모델**: Qwen3-235B FP8과 Solar-100B bf16 중 무엇으로 할지 정하지 않았다 (`PLAN.md` §5).
- **SDK 버전 조합**: driver 2026.4, llm 2026.3, torch 2026.2는 지원 조합이 아니다. 업그레이드 여부를 정하지 않았다 (`PLAN.md` §6).
- **패치한 vendor 런타임 유지 방안**: `track2_opt/solar_engine/vendor/furiosa-opt-rt`를 어떻게 유지할지 정하지 않았다.
- **측정 잡음**: 같은 서버의 k8s 파드(vLLM, npu_lora)가 CPU를 경합해 수치가 ±15% 흔들린다. 그래서 정식 측정 절차(loadavg 게이트, A/B 교차)를 실험 기록으로 정착시켜야 한다. `experiments/`에는 아직 기록이 없다 (`PLAN.md` §7).

## Log

<!-- research-hub:log (report.py inserts new entries below this line, newest first) -->

### 2026-10-16 13:42+09:00 — Speculative decoding for Solar

- Result: EAGLE-3 v1 (2.78M chat tokens): tau 1.42 on Solar's own generations (too low); v2 online training on 20.3M tokens with reasoning traces running on GPU1
- Next: eval v2 checkpoints; if tau >= 2.5 build T>1 verify kernels
- Metrics: tau=1.42
- Experiments: experiments/2026-10-16-eagle3-v1.md
- By: claude@etri-ESC8000-E11 · main@bb532fd-dirty

### 2026-10-16 13:42+09:00 — Solar decode latency

- Result: PIN_BOUNCE pinned DMA reads + MOVDIR64B uploads: exact bf16 single-stream 28.7 ms/tok (34.9 tok/s, p2); lm_head split slower (48.8), BAR reads neutral
- Next: repeat pinned A/B on a quiet box; diagnose LM_SPLIT layer slowdown
- Metrics: tok_s=34.9, ms_per_tok=28.7
- Experiments: experiments/2026-10-16-movdir64b-upload.md, experiments/2026-10-16-pinned-bounce.md, experiments/2026-10-16-bar-reads.md, experiments/2026-10-16-lm-head-split.md
- By: claude@etri-ESC8000-E11 · main@bb532fd-dirty

### 2026-10-16 10:39+09:00 — NPUOFF-1 ✅

- Result: STATUS.md 부트스트랩 완료: Solar-100B bf16 2xRNGD 정확 모드 단일 요청 30.7-32.1 ms/tok (31-33 tok/s), MOVDIR64B 업로드로 약 30.0ms
- Next: prefill(P4) 설계와 배치 서빙(P3/P5) 착수, 측정 결과를 experiments/에 기록
- By: claude@etri-ESC8000-E11
