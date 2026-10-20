# Optimus train(): real-model LoRA with expert offload (GPU+CPU) and on RNGD

- **Date**: 2026-10-19
- **Task**: NPUOFF-32 OptimusPrime MoE offload (generalized tiers)
- **Host / device**: RNGD+GPU host: 2× A100 80GB PCIe (CUDA 12.8, torch 2.10.0), 2× RNGD (furiosa-torch 2026.2, npu-lora local 96b36ed)
- **Commit**: aicomp feat/optimus-moe-offload 0391d19 (fixes bdb4adc, 1d6f19c)

## Question

Does `opt_prime.train.train(TrainJob)` train real checkpoints for the MoE families with experts offloaded (GPU budget
+ frozen host experts), and LoRA on RNGD through npu-lora?

## Setup

Data: 64 synthetic prompt/response examples (`train_auto.py --make-data`), one sequence per step, seq_len 128.
MoE runs: `--offload-gpu-gb 20` -> planner puts hot experts on the GPU within 20 GB, the rest stay on CPU (frozen,
backward via host weights). LoRA r=16, alpha=32, q/v.

Two bugs found and fixed on the way:
1. Gemma 4 real config: router + experts are decoder-layer children next to a dense MLP -> ModelSpec saw 0 MoE layers,
   offload was skipped, OOM. Fixed in `model/hf.py` (bdb4adc), test added.
2. `import torch` autoloads furiosa-torch (device-backend autoload); any CUDA backward then fails with
   `opt_ready_stream && opt_parent_stream INTERNAL ASSERT FAILED (engine.cpp:1085)`, even `(x*2).sum().backward()`.
   Fixed by `TORCH_DEVICE_BACKEND_AUTOLOAD=0` for non-RNGD jobs (1d6f19c).

## Command

```bash
cd aicomp/opt_prime/examples
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_auto.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --data train.jsonl --device cuda:0 --steps 20 --seq-len 128 --lr 2e-4 --offload-gpu-gb 20 --out qwen3moe_lora
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 ~/venvs/tf58/bin/python -m torch.distributed.run --nproc_per_node=1 train_auto.py \
  --model google/gemma-4-26B-A4B-it --data train.jsonl --device cuda:0 --steps 20 --seq-len 128 --lr 2e-4 --offload-gpu-gb 20 --out gemma4_lora
HF_HUB_OFFLINE=1 python3 train_auto.py --model unsloth/Llama-3.2-1B-Instruct --data train.jsonl --device rngd:0 --steps 30 \
  --seq-len 128 --lr 5e-4 --out adapter_rngd
```

## Results

| model | device | experts GPU / CPU | steps | loss first -> last | wall |
|---|---|---|---:|---|---:|
| Qwen3-30B-A3B-Instruct-2507 (bf16) | 1× A100 + CPU | 2275 / 3869 | 20 | 3.2067 -> 0.3811 | 184 s |
| gemma-4-26B-A4B-it (bf16, tf 5.8.1) | 1× A100 + CPU | 1805 / 2035 | 20 | 6.4158 -> 0.7231 | 202 s |
| Solar-Open-100B (bf16; 2026-10-20, budget 40 GB, commit 6a2e35f) | 1× A100 + CPU | 1365 / 4779 | 10 | 3.5282 -> 0.6020 | 708 s |
| Llama-3.2-1B-Instruct (LoRA, npu-lora) | RNGD chip 0 | - | 30 | 3.7525 -> 0.5886 | 8102 s (compile 8065 s, then ~1.2 s/step) |
| Qwen2.5-0.5B-Instruct (LoRA, npu-lora qwen2 block) | RNGD chip 1 | - | 30 | 2.4085 -> 0.4879 | 2399 s (compile 2349 s resumed from part cache after a 10000 s timeout; ~1.7 s/step) |
| Llama-3.2-1B-Instruct (full FT, npu-lora) | RNGD chip 0 | - | 20 | 3.7530 -> 0.3436 | 7803 s (compile 7771 s, ~1.6 s/step) |

Failed attempts (same day): Gemma-4 OOM (78.9 GiB, spec bug), Qwen3/Gemma-4 backward assert (furiosa autoload).
Tiny probes after the fix: qwen3_moe / gemma4_text LoRA offload-hot 0.5 bf16 first-step loss = HF (5.07301 vs 5.07291,
4.84954 vs 4.84943), base unchanged -> PASS.

## Addendum 2026-10-20: Solar-Open-100B

Two failures before it ran: (1) pp=2 put ~450 GB of host memory in each rank (the profiler kept the original
experts alive and the host tier copied them once more); stopped by hand at 900 GB used of 1 TB; fixed in aicomp
(profiler holds modules weakly, host tier shares tensors; Qwen3-30B anon RSS 128.6 -> 74.6 GiB). (2) pp=1 ran
out of GPU memory in the backward: the offloaded experts' backward gathered a weight copy per routed pair (21 MB x
1024 pairs per layer); fixed by computing it one expert at a time (6a2e35f). Then 10 steps 3.53 -> 0.60 in 708 s
(includes 5.6 min weight loading).

## Conclusion

LoRA training with experts offloaded works on real 26–30B MoE checkpoints on a single GPU. Loss drop is on a tiny
synthetic set (memorization), so it shows the gradient path works, not model quality. RNGD LoRA works for Llama
dense; first compile is 2.2 h (cached afterwards). Qwen2 block (q/k/v bias) verified on silicon. Full FT on RNGD: saved weights load in HF and greedy-generate the trained answer ('Project 0 is code-named Optimus-0.'), base model does not, RNGD MoE
training (needs host-side expert backward).
