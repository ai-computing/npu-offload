# STATUS — NPU Offload

<!-- research-hub:header:begin (maintained by .research/report.py; keep the "- **Key**: value" format) -->
- **State**: active
- **Updated**: 2026-10-21 19:58+00:00
- **Agent**: claude@npo-dev-wjpk6
- **Task**: RNGD layer engine: shape templates for attention / norm / router (#3)
- **Last result**: Qwen3-30B-A3B layer engine generated from the geometry: tokens = HF (3 prompts x 32 exact; 128-token runs exact except one 1-ulp swap), 25-30 tok/s on one RNGD (88 hot experts/layer); Solar geometry passes the same kernel checks
- **Next**: pack x2/h2/logits into one download, device-side expert sum, server mode for OptimusPrime
- **Blockers**: none
<!-- research-hub:header:end -->

## Goal

PolyStrata 방식 MoE 오프로드를 Furiosa RNGD에 이식한다. hot expert는 NPU HBM에 상주시키고 cold expert는 호스트 CPU에서 계산하며, PCIe로는 activation만 오간다. 대상은 HBM(RNGD 2장, 95GB)에 다 들어가지 않는 200GB 이상 MoE 모델이며, 현재는 Solar-Open-100B bf16(205GB)이다. 목표는 2× RNGD에서 decode 30 tok/s 이상(33 ms/tok 이하)이다 (`PLAN.md` §1–2, §6, `track2_opt/SOLAR_OPT_SPEC.md`).
- 참고: 이 디렉터리는 git 저장소가 아니다(`git log` 실패). 아래 근거는 모두 파일 경로로 단다.

## Findings

<!-- Durable conclusions with numbers and a pointer to the experiment record. Newest first. -->
- **레이어 엔진을 모델 geometry에서 생성 (2026-10-22, #3)**: `npu_kernels.rngd.layer`(`plan_layer`, `generate_layer`)가 attention 블록(q/k norm, RoPE, KV, attention, o_proj, post-norm, router), expert, LM head 커널과 decode 엔진을 만든다. Qwen3-30B-A3B를 RNGD 1장에서 HF와 같은 토큰으로 decode한다(프롬프트 3개 × 32토큰 전부 일치, 128토큰 실행은 1-ulp 교차 1회 외 일치). hot expert 88개/레이어(적중률 0.86-0.94)에서 25-30 tok/s다. Solar geometry도 같은 커널 검사를 통과하며, 손으로 쓴 Solar 엔진은 바꾸지 않았다 (`experiments/2026-10-22-rngd-layer-templates-qwen3.md`).
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

### 2026-10-21 19:58+00:00 — RNGD layer engine: shape templates for attention / norm / router (#3)

- Result: Qwen3-30B-A3B layer engine generated from the geometry: tokens = HF (3 prompts x 32 exact; 128-token runs exact except one 1-ulp swap), 25-30 tok/s on one RNGD (88 hot experts/layer); Solar geometry passes the same kernel checks
- Next: pack x2/h2/logits into one download, device-side expert sum, server mode for OptimusPrime
- Metrics: tok_s=28.4
- Params: model=Qwen3-30B-A3B, e_t=88, device=rngd-x1
- Experiments: experiments/2026-10-22-rngd-layer-templates-qwen3.md
- By: claude@npo-dev-wjpk6 · feat@a93ef3d

### 2026-10-20 10:47+09:00 — NPUOFF-32

- Result: Solar-Open-100B LoRA with expert offload on 1 A100 + CPU: loss 3.53->0.60 / 10 steps (708 s); fixed host-memory duplication and per-pair backward OOM
- Next: RNGD MoE LoRA (npu-train#6) on silicon; PD split per aicomp#69
- Params: model=Solar-Open-100B, offload_gpu_gb=40, steps=10
- Experiments: experiments/2026-10-19-optimus-training-real-models.md
- By: claude@xeon-max · docs/optimus-offload-records@5cf81d4-dirty

### 2026-10-19 22:41+09:00 — NPUOFF-32

- Result: RNGD Llama-3.2-1B full FT via train()/npu-lora: loss 3.75->0.34 / 20 steps, ~1.6 s/step; HF reload generates trained answer
- Next: PR creation and library repo location pending user decision; RNGD MoE training needs host-side expert backward
- Params: device=rngd, method=full
- Experiments: experiments/2026-10-19-optimus-training-real-models.md
- By: claude@npu-node-2 · main@2250c63-dirty

### 2026-10-19 22:34+09:00 — NPUOFF-32

- Result: RNGD Qwen2.5-0.5B LoRA via train()/npu-lora: loss 2.41->0.49 / 30 steps, ~1.7 s/step
- Next: RNGD Llama-1B full FT result
- Params: device=rngd, seq_len=128
- Experiments: experiments/2026-10-19-optimus-training-real-models.md
- By: claude@npu-node-2 · main@25ad364-dirty

### 2026-10-19 21:08+09:00 — NPUOFF-32

- Result: Real-model LoRA with expert offload: Qwen3-30B-A3B 3.21->0.38, gemma-4-26B 6.42->0.72 (1 A100 20GB + CPU); RNGD Llama-1B LoRA 3.75->0.59
- Next: RNGD Qwen2.5 LoRA / Llama full FT results; PR creation pending user OK
- Params: offload_gpu_gb=20, seq_len=128, steps=20
- Experiments: experiments/2026-10-19-optimus-training-real-models.md
- By: claude@npu-node-2 · main@27cbae8-dirty

### 2026-10-19 17:18+09:00 — OptimusPrime MoE offload (generalized tiers)

- Result: ATOM expert tier works (per-expert RBLN graphs, ~210us/expert, PASS); engine selection runs Solar on the PoC engine at 29.4 tok/s decode; ModelSpec/providers/engine registry pushed (aicomp feat/optimus-moe-offload)
- Next: template PoC dense kernels for more geometries (step 3); ATOM custom kernels pending SDK support
- Metrics: tok_s=29.4
- Params: model=Solar-Open-100B, engine=rngd-layer
- Experiments: experiments/2026-10-19-optimus-moe-offload-tiers.md
- By: claude@npu-node-2 · main@e6807c3-dirty

### 2026-10-19 16:49+09:00 — OptimusPrime MoE offload (generalized tiers)

- Result: Solar-Open-100B on OptimusPrime exact: rngd-opt x2+cpu 8.56 tok/s (64/64), +gpu45 10.18 (bf16 tie at tok 2); Qwen3-30B rngd-opt x2(64)+cpu 15.42, compiled 18.42 tok/s; native dispatcher +20-40%; ModelSpec step 1 done
- Next: block providers + RNGD in-process segment executor (generalized solar_engine) to reach PoC 31+ tok/s through the planner
- Metrics: tok_s=8.56
- Params: model=Solar-Open-100B, tiers=rngd-opt_x2+cpu
- Experiments: experiments/2026-10-19-optimus-moe-offload-tiers.md
- By: claude@npu-node-2 · main@3be2bc0-dirty

### 2026-10-19 09:55+09:00 — NPUOFF-31 ✅

- Result: PR #1 updated (airm/v0.1 e9d7289): rm-core v0.1.1; selftest reports test_airm_runner as SKIP when the entry point is not registered (was KeyError): uninstalled 5/6 passed + 1 skipped exit 0, installed 6/6; rm_core.conformance exit 0 (sim/loopback/greedy PASS); moe_offload_decode_ms_per_token marked deprecated per PROTOCOL §6; check_sensitive clean; review comment posted
- Next: build worker image and run the chart on llm-d (NPUOFF-29)
- By: claude@npu-node-2 · main@3be2bc0-dirty

### 2026-10-19 09:10+09:00 — NPUOFF-30 ✅

- Result: PR https://github.com/ai-computing/npu-offload/pull/1 (airm/v0.1): runner airm.workloads:npu-offload loads via registry.load; rm_core.conformance exit 0 (sim/loopback/greedy PASS; v0.1 has no airm.workloads check); selftest 6/6; helm lint/template OK; check_sensitive clean
- Next: build worker image and run the chart on llm-d (NPUOFF-29); rm-core issue: conformance check for airm.workloads, candidate metric expert_hit_pct
- By: claude@npu-node-2 · main@3be2bc0-dirty

### 2026-10-19 08:50+09:00 — Speculative decoding for Solar

- Result: EAGLE-3 v3 (87.5M reasoning-style tokens, 2-GPU queue training, 6.2h): tau 1.92 (k=6) on Solar's own outputs, up from 1.42 (v1); per prompt 1.59-2.56
- Next: self-distillation on Solar generations; 2-layer draft; then decide on T>1 verify kernels
- Metrics: tau=1.92
- Experiments: experiments/2026-10-16-eagle3-v3.md
- By: claude@npu-node-2 · main@25bb785-dirty

### 2026-10-16 15:01+09:00 — NPUOFF-28 ✅

- Result: checker clean (no IPs/credentials found). Parametrized RNGD_LLMD (now required, no ~/ fallback) and worker --gateway default ($LLMD_GATEWAY_URL) via env + untracked cluster.local.env (moe_offload/siteenv.py, cluster.example.env); added ai-rm.yaml (l2-memory, service) and README ## ai-rm. selftest 5/5. Pushed bf2fe53
- Next: eagle3/prep*.py and evalgen/*.json, logs/ contain /home/<user> paths (not secrets); container image + EPP plugin needed for llm-d
- New todos: Build npu-offload worker container image and validate k8s manifest on llm-d cluster
- By: claude@npu-node-2 · main@bf2fe53-dirty

### 2026-10-16 13:42+09:00 — Speculative decoding for Solar

- Result: EAGLE-3 v1 (2.78M chat tokens): tau 1.42 on Solar's own generations (too low); v2 online training on 20.3M tokens with reasoning traces running on GPU1
- Next: eval v2 checkpoints; if tau >= 2.5 build T>1 verify kernels
- Metrics: tau=1.42
- Experiments: experiments/2026-10-16-eagle3-v1.md
- By: claude@npu-node-2 · main@bb532fd-dirty

### 2026-10-16 13:42+09:00 — Solar decode latency

- Result: PIN_BOUNCE pinned DMA reads + MOVDIR64B uploads: exact bf16 single-stream 28.7 ms/tok (34.9 tok/s, p2); lm_head split slower (48.8), BAR reads neutral
- Next: repeat pinned A/B on a quiet box; diagnose LM_SPLIT layer slowdown
- Metrics: tok_s=34.9, ms_per_tok=28.7
- Experiments: experiments/2026-10-16-movdir64b-upload.md, experiments/2026-10-16-pinned-bounce.md, experiments/2026-10-16-bar-reads.md, experiments/2026-10-16-lm-head-split.md
- By: claude@npu-node-2 · main@bb532fd-dirty

### 2026-10-16 10:39+09:00 — NPUOFF-1 ✅

- Result: STATUS.md 부트스트랩 완료: Solar-100B bf16 2xRNGD 정확 모드 단일 요청 30.7-32.1 ms/tok (31-33 tok/s), MOVDIR64B 업로드로 약 30.0ms
- Next: prefill(P4) 설계와 배치 서빙(P3/P5) 착수, 측정 결과를 experiments/에 기록
- By: claude@npu-node-2
