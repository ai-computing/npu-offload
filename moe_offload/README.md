# moe_offload — tiered MoE expert offloading, served llm-d style

Control plane for running a routed-MoE model whose experts do not fit in accelerator memory: hot experts stay in
accelerator HBM, cold experts are computed on the host CPU, and only activations cross PCIe. The data plane is a
model-specific engine binary (today `track2_opt/solar_engine`, Solar-Open-100B bf16 on 2x Furiosa RNGD); everything
here is model-agnostic and talks to it through a small stdin/stdout protocol.

```
            ┌───────────────────── gateway (llmd/gateway.py) ─────────────────────┐
 client ──▶ │ classify request (ko/code/en) → score workers:                       │
  OpenAI    │   expert-affinity EWMA (seeded by placement priors) − load           │
            │ registry / KV index / EndpointPicker reused from rngd_llmd           │
            └──────────────┬───────────────────────────────┬──────────────────────┘
                           │ HTTP (OpenAI)                  │ /register /heartbeat /moe_event
            ┌──────────────▼──────────────┐   ┌────────────▼────────────┐
            │ worker.py  (one per engine) │   │ worker.py (other        │
            │ /v1/(chat/)completions      │   │ placement, other node)  │
            │ /metrics (vLLM-compatible)  │   └─────────────────────────┘
            └──────────────┬──────────────┘
                           │ JSON lines (engine.py)
            ┌──────────────▼──────────────────────────────────────────────┐
            │ engine binary `<data_dir> serve`: chip A attention+shared+  │
            │ hot experts, chip B hot experts + lm_head, CPU cold experts │
            └─────────────────────────────────────────────────────────────┘
```

## Modules

| module | what it does |
|---|---|
| `spec.py` | `ModelSpec` from an HF `config.json`; bytes per expert / dense layer / head / KV |
| `profile.py` | expert activation counts `[layer][expert]` from a routing profile; per-class profiles |
| `placement.py` | tiered placement `[layer][tier] -> experts` (`top`, `mix:<h>`), tier hit fractions, class priors; CLI |
| `budget.py` | per-tier HBM accounting (dense, KV, head, reserve) → max expert slots per layer; checks compiled slots fit |
| `engine.py` | `EngineProcess` (serve-mode engine, streaming), `MockEngine` (NPU-free), the tuned Solar env (`SOLAR_ENV`) |
| `worker.py` | OpenAI-compatible HTTP worker, reasoning split, vLLM-compatible `/metrics`, gateway registration |
| `llmd/affinity.py` | request classifier, per-(worker, class) hit-rate EWMA, scoring |
| `llmd/gateway.py` | MoE-aware gateway on rngd_llmd's `WorkerRegistry`/`KVIndex`/`EndpointPicker`, HTTP proxy dispatch |

## How the llm-d ideas map

| llm-d | here |
|---|---|
| KV-cache tiering / offload (HBM → CPU → storage) | expert tiering (accelerator HBM → host DRAM), planned from a routing profile |
| resource declaration per variant | `budget.Tier` (HBM, what else the chip hosts) → expert slots; placement policy per tier set |
| EPP prefix-cache-aware scorer | expert-affinity scorer: workers report per-class hit rate after each request; placement priors seed it |
| EPP load scorer / queue metrics | `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc` exported by the worker |
| model server behind InferencePool | `worker.py` (OpenAI API); `deploy/` has an untested Kubernetes sketch |

## Use

```bash
# NPU-free regression (placement, budget, reasoning split, classifier, gateway affinity e2e over HTTP)
python3 -m moe_offload.tests.run

# plan a placement from a profile
python3 -m moe_offload.placement --model <hf dir> --profile p0/results/routes.pt --slots 24,32 --policy mix:4 --out placement.json

# gateway + one real worker (2x RNGD)
python3 -m moe_offload.llmd.gateway --port 8080
python3 -m moe_offload.worker --engine track2_opt/solar_engine/target/release/solar_engine \
    --data track2_opt/solar_engine/data --model <hf dir> --port 8101 --gateway http://127.0.0.1:8080 \
    --priors '{"ko":0.8683,"en":0.8623,"code":0.8964}'
curl -s localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'
```

The gateway imports rngd_llmd from `$RNGD_LLMD` (required; set it in the environment or in an untracked
`cluster.local.env`, see `cluster.example.env`). `--gateway` of the worker defaults to `$LLMD_GATEWAY_URL`.

## Status and limits

- Engine: exact bf16 greedy decode, one stream per engine (requests queue in the worker). No prefix cache yet:
  each request re-runs its prompt token by token (prefill = T=1 steps).
- Model dims are compile-time in the engine's kernels; a different MoE model needs its own data plane build.
  `spec`/`budget`/`placement` already take any routed-MoE config.
- Affinity routing is verified with mock workers (tests); a single real worker cannot show it. Multiple real
  workers need more accelerators or one engine per placement on separate hosts.
- `deploy/` is a sketch: no Kubernetes on the development host, so it is not exercised.
