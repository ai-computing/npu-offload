# Vendor MoE runtime mechanisms: what our Solar hybrid runtime can reuse

Date: 2026-10-15. This is a read-only analysis. Nothing was run on an NPU. The only Python executed was light imports and docstring/`dir()` introspection.

Installed versions (from `pip list`): furiosa-llm 2026.3.0, furiosa-llm-native 2026.3.0, furiosa-native-llm-common 2026.3.0, furiosa-kernels 2026.3.0, furiosa-tcc 2026.3.0, furiosa-torch-ext 2026.3.0, **furiosa-torch 2026.2.0**, furiosa-models 2026.2.0, **furiosa-native-runtime 2026.2.0**, furiosa-native-compiler 2026.1.0.

Path prefix used below: `SP = ~/.local/lib/python3.10/site-packages`.

Each claim is tagged as one of:
- **[V]** verified in source, stubs, or introspection.
- **[I]** inferred from strings or naming. Not proven.
- **[X]** must be tested on hardware.

---

## TL;DR

1. **The vendor has no special gather op.** The PyTorch MoE ops emit plain `aten.index.Tensor` (`w1[expert_ids]`). The furiosa-llm tracer enforces two conditions:
   - a **single index tensor**;
   - an index of dtype **int32**.

   The torch compiler reports `aten::index.Tensor`, `aten::index_select` and `aten::gather` as supported. `furiosa.torch.compiler.compile` wraps every failure as `UnsupportedOpError`, so the message does not tell you which of these happened:
   - the op is genuinely unsupported;
   - the index is int64;
   - the multi-chip sharding conflicts with the gather;
   - a tactic failed.

   **First, read `e.__cause__`. Then retry with int32 ids, the resident table as an `nn.Parameter` or buffer, and the vendor `TokenWiseMoEOp` graph shape, on a single chip first.** [V]/[X]

2. The **real device-side gather primitive** is TCL `tcl.gather(table, idx_i32, Axis)`. It optionally takes `valid_length` for a sparse gather. furiosa-kernels ships a **Solar-Open activation-stationary MoE kernel** that does exactly our computation: `gather → dequant → einsum(silu) → einsum → weighted sum`. It is NVFP4 only. [V]
   - The documented Python path (`furiosa.torch.TclModule`) is **missing from the installed furiosa-torch 2026.2**. The 2026.3 examples import it.
   - `furiosa-tcc` (TC → EDF CLI) is installed. `ir.Edf.deserialize` and `EdfModule` exist. So *TCL → EDF → run with our own tensors* is plausible but unverified (version skew).

3. Per-call overhead fix that needs **no new compiler features**: `CompileModule.load_and_build_execute_program()` returns an `ir.LoadedTpProgram` bound to fixed device I/O buffers. The vendor's own `ExecutionPool` reuses it: copy inputs into the pooled buffers, then `await prog.run()`.
   - This skips per-call output allocation, param deserialization and program building in `torch.ops.furiosa.edf`.
   - It only works when **all inputs are fixed buffers**: x, ids, gates, and a resident `[E_hot,…]` table. Our current design passes a different weight tensor per slot, so it cannot use this. A device gather on a resident table unlocks it.
   - `ir.sprint.seq/par` plus `memory.copy` return composable `TpProgram`s. In principle that allows a single submit of H2D + exec + D2H, or of several EDFs. [V API] / [X behavior]

4. Vendor decode runs inside the closed Rust `NativeLLMEngine`.
   - Python can only call `generate` and `stream_generate`.
   - There is **no RawModel API in the installed 2026.3.0 wheels**. A grep of every .py and .so found no `RawModel`/`raw_model` string.
   - The artifact is a pipeline of **stages**: `[emb+L0 qkv] [L0 attn] [L0 oproj+MoE+L1 qkv] [L1 attn] …`. Each stage has bucketed EDF tasks.
   - The runtime **composes** these per iteration ("ComposedTask", FirstFitComposer, `npu_queue_limit` = 1). [V for structure, I for composition]
   - You cannot load your own graphs into it through any supported API.

5. **Partial expert residency or host expert compute: none in the vendor paths.**
   - `ExpertParallelMoEOp` is a single-device weight-stationary scatter/gather, not cross-device EP.
   - mppp accepts `cpu` devices, but the next-gen builder compiles every task to EDF.

---

## 1. How the vendor lowers expert-weight gather

### 1.1 PyTorch-level MoE ops (furiosa-models, used by furiosa-llm tracing)

`SP/furiosa/models/core/operators/moe.py`

| Op | Gather mechanism | Lines |
|---|---|---|
| `TokenWiseMoEOp.compute` (activation-stationary, decode) | `w1_e = w1[expert_ids]`, `w2[...]`, `w3[...]`, i.e. **`aten.index.Tensor`** → `(T,k,F,D)`, then flatten to `(T·k)` and `torch.bmm` | 279-329 (gather 316-318); `expert_mlp` 331-390; `_mlp_core_bmm3d` 225-253 |
| `ExpertParallelMoEOp` (weight-stationary) | builds a one-hot, a `cumsum` of positions, **`index_put_`** of activations into `(E·L, D)` with L = T, a dense bmm over **all E**, then **`index_select`** back | class 440; `_scatter` 591-629; `_gather` 631-640; `_one_hot` 643-648 |
| `BlockwiseMoEOp` (prefill, T > E/K) | the scatter index is computed in pure aten (mask/cumsum/div), the weight gather index comes from a **TacticKernel DFG** (`tk_graphs/moe_blockwise_compute_wg_idx.yaml`), then `w1[gather_idx]` (`aten.index.Tensor`) and bmm over G blocks | class 651; TK load 694-707; `_compute_indices` 860-958 (TK call 954-956); `_gather_weights` 998-1024; `_gather` 1078-1104 |

Op selection: `MoELayer` uses TokenWise when T ≤ E/K, otherwise Blockwise (`SP/furiosa/models/core/layers/moe/moe.py:74-75, 159-161`). For Solar, E/K = 16. Routing returns **int32** ids (`routing.py:99`, `topk_indices.to(torch.int32)`).

### 1.2 What makes the compiler accept it

- furiosa-llm tracer passes (`SP/furiosa_llm/parallelize/trace.py`) [V]:
  - `_remove_unnecessary_larger_typecast_before_index` (204-221) strips an int64 cast placed in front of an index.
  - `_check_all_index_ops_i32_index` (224-238) **raises unless every `aten.index.Tensor` / `index_put_` has exactly one index tensor of dtype int32**.
  - Both are called from `_preprocess_gm_for_model_rewrite` (352-365).
- In the vendor pipeline the expert weights are ordinary **graph inputs**. Constant tensors become pipeline `ParamInfo` inputs (`parallelize/new_pipeline_builder.py` `_add_tensor_info`), so "gather from an input tensor" is the normal vendor case.
- The vendor compiles with **`furiosa.native_common.compiler.compile`**, the npu-tools LLM compiler from native-llm-common 2026.3 (`parallelize/pipeline/builder/converter.py:853-922`). It passes an LLM compiler config: tactic hints, layer range, `ForLlmModelIOBound`, and so on. **furiosa.torch uses a different binding**: `furiosa.native_torch.compiler`, from furiosa-torch 2026.2 / native-compiler 2026.1. The two are not guaranteed to have the same lowering coverage. [V]
- `furiosa.native_torch.compiler.is_supported_aten(...)` (introspected) [V]:
  - `aten::index.Tensor`, `aten::index_select`, `aten::gather`, `aten::index_put`, `aten::scatter.src`, `aten::cumsum`, `aten::bmm`, `aten::mm` → **supported=True**.
  - `aten::embedding`, `aten::index_put_`, `aten::_unsafe_index.Tensor` → **False**.
  - `furiosa::sparse_select` and `furiosa::gather_i32` → supported=False, importable=True.
- `furiosa.torch.compiler.compile` (`SP/furiosa/torch/compiler/__init__.py:51-69`) **catches every exception and re-raises `UnsupportedOpError("failed to compile the graph") from e`**. Our "UnsupportedOpError for index_select / W[ids]" (PLAN.md §6) therefore does not prove the op is unsupported. The real cause is in `e.__cause__` and the compiler log (`furiosa.native_torch.enable_logging`). [V]

### 1.3 Custom ops registered via `torch.library` (callable inside our graph)

`SP/furiosa/torch/custom_ops/furiosa_internals.py` [V]:

```
furiosa::sparse_select(Tensor x, Tensor[] indices, int dummy_index, int index_group_size) -> Tensor        # :13, fake :53  (== aten.index(x, indices))
furiosa::sparse_select_v1_5(Tensor x, Tensor[] indices, int dummy_index, int index_group_size) -> Tensor   # :110, fake :124
furiosa::gather_i32(Tensor x, int dim, Tensor index, bool sparse_grad) -> Tensor                           # :63, fake :98 (== torch.gather)
furiosa::dequantize(Tensor x, Tensor scale, str quantization_method, ScalarType output_type) -> Tensor     # :134
```

- The *eager* implementations wrap the index as a **module buffer**, so it is a compile-time constant, and compile a fresh `CompileModule` on every call. That is useless at runtime.
- Inside a traced graph the op node is kept and lowered natively. The vendor uses `sparse_select_v1_5` for the **paged-KV gather by runtime block ids**:
  - `SP/furiosa/models/core/attention/ops/paged_attention.py:162-254`;
  - constraint: 1-D indices, table shape `(N, 1, …)` (block_size = 1), `dummy_index` rows are skipped.
- `furiosa::gather_i32` is registered a second time in `SP/furiosa_llm/parallelize/model_rewriter/ops/custom_ops.py:5`. Importing both modules may warn or conflict.

Other custom ops in furiosa.torch:
- `furiosa::edf` (`custom_ops/edf.py:27`): executes an EDF.
- `furiosa::dfg` (`custom_ops/dfg.py:22`): embeds a DFG or TacticKernel graph in a torch graph. This is how `BlockwiseMoEOp` inlines its TK yaml.
- `furiosa::compilable` (`custom_ops/compilable.py:24`).

### 1.4 Minimal recipe for `out[t] = Σ_k g[t,k]·FFN(x[t], W_table[ids[t,k]])`

This is a direct transliteration of `TokenWiseMoEOp`, so the graph matches what the vendor compiler sees. Cold experts map to any valid slot (for example 0) with g = 0. Each candidate below is [X].

```python
class HotMoE(nn.Module):                       # one per layer (or one shared graph + per-layer tables as inputs)
    def __init__(self, E_hot, H=4096, F=1280):
        super().__init__()
        bf = torch.bfloat16
        self.w1 = nn.Parameter(torch.zeros(E_hot, F, H, dtype=bf), requires_grad=False)  # gate
        self.w2 = nn.Parameter(torch.zeros(E_hot, F, H, dtype=bf), requires_grad=False)  # up
        self.w3 = nn.Parameter(torch.zeros(E_hot, H, F, dtype=bf), requires_grad=False)  # down
    def forward(self, x, ids, g):              # x:(T,H) bf16, ids:(T,K) **int32** slot ids, g:(T,K) bf16
        T, K = ids.shape; H = x.shape[1]
        w1 = self.w1[ids].flatten(0, 1)         # aten.index.Tensor, single i32 index -> (TK,F,H)
        w2 = self.w2[ids].flatten(0, 1)
        w3 = self.w3[ids].flatten(0, 1)         # (TK,H,F)
        xk = x[:, None, :].expand(T, K, H).reshape(T * K, H, 1).contiguous()
        h = torch.nn.functional.silu(torch.bmm(w1, xk)) * torch.bmm(w2, xk)   # (TK,F,1)
        y = torch.bmm(w3, h).view(T, K, H)
        return torch.einsum("TkD,TkD->TD", y, g[..., None].expand(T, K, H))     # vendor post-gating form (moe.py layer :351)
m = CompileModule.from_module(HotMoE(E_hot), (x0, ids0.to(torch.int32), g0)); m.to(dev)
```

Fallbacks, in order:
- (a) The same graph with `torch.ops.furiosa.sparse_select_v1_5.default(W.view(E_hot, 1, -1), [ids.flatten()], dummy, K)` in place of `W[ids]`. This mirrors the paged-KV usage.
- (b) `torch.gather` (`aten::gather` is listed as supported).
- (c) A TCL kernel (§2).

Diagnose first:
- Print `e.__cause__`.
- Compile once **without** `set_mesh_kind(Mesh(2))` and fusion. Gathering along E on a tensor whose inter-chip axes shard E or F/H is a plausible cause of the failure (`RngdTensor._with_inter_chip_axes`, `edf.py:62-66`).
- Try E_hot small (2–4) first.

Note the size: one Solar expert (gate + up + down) in bf16 is about 31.5 MB. A `[E_hot,…]` table must be a **single contiguous allocation**, which in practice means one table per layer of about 1 GB for 30 slots. The earlier "1450 experts / 45.6 GB" was many small tensors. Large single allocations under fusion8/Mesh(2) are untested [X].

---

## 2. furiosa-kernels / TCL

### What exists [V]

`SP/furiosa/kernels/`:
- `qwen3_moe/{naive,optimized,common}`:
  - activation-stationary FP8: `common/activation_stationary_moe_fp8.py:21-51`, using `tcl.gather(w, ids_i32, E)` on weight and scale;
  - blockwise FP8 (`common/blockwise_moe_fp8.py`, `optimized/qwen3_blockwise_moe_fp8.py`);
  - fused `first_tokenwise` / `mid_tokenwise_*` / `last_tokenwise_*_with_lm_head` and `_small_t` variants.
- **`solar_open/solar_pro3_100b_w4fa16kv16/`**, a full Solar-Open kernel set for NVFP4 weights and bf16 activations (config JSON `solar_open/common/*.json` has 128E/top-8/1 shared/F = 1280/H = 4096/48L):
  - `activation_stationary_moe.py`:
    - `route_tokens_to_experts` (sigmoid + bias + topk + normalize, :32-51);
    - **`gather_and_dequantize_nvfp4_weights`** (`tcl.gather(weight_f4, gather_index_i32, E)`, :54-72);
    - `activation_stationary_moe_nvfp4_w4a16` (:115-188);
    - top-level `activation_stationary_moe`, which fuses router + routed + shared + `tcl.all_reduce` over F-chips (:191-245).
  - `blockwise_moe.py`, `mid_tokenwise.py` (`mid_tokenwise_as` :49, `mid_tokenwise_bw` :161, `mid_tokenwise_all_reduce` :280), `qkv.py`, `oproj.py`, `lm_head.py`.
  - Entry functions `solar_open/entry_function.py` build specialized `Kernel`s:
    - `_FXB_NUM_CHIPS = 4` (:35);
    - AS vs BW choice is `T <= E//K` (`solar_open/common/entry_function.py:11-22`);
    - the compiler config is in `solar_pro3_100b_w4fa16kv16/config.py`.
- `common/blockwise_moe_index.py`, `common/topk.py`, `common/contract.py` (einsum_dpe / einsum_dpe_silu).
- Generic TCL example of exactly our op: `SP/furiosa/tcl/examples/segment_gemm.py:57-74`, `segment_gemm_as_kernel` = `typecast(i16→i32)` + `tcl.gather(w, ids, G)` + contract. `tcl.gather` signature (`SP/furiosa/tcl/_tcl.py:769-809`):

  ```
  tcl.gather(table, indices(i32), axis, *, batch_axis=None, valid_length=None,
             table_dram_layout=None, compiler_hints=None)
  ```

  `valid_length` gives a **sparse gather that skips invalid entries**. That is a natural way to say "only the first n slots are hot".

### How they are invoked [V]

`Kernel.from_edsl(tcl.kernel(bindings=…, config=…)(fn._func))` produces TC DSL text (`SP/furiosa/tcl/kernel.py:88-142`).

The examples run it through `from furiosa.torch import TclModule` and then `CompileModule.from_exported(torch.export.export(TclModule(k), args))` (`segment_gemm.py:240-266`). **`TclModule` does not exist in the installed furiosa-torch 2026.2.0.** Its `__all__` (`SP/furiosa/torch/__init__.py`) has `TacticKernelModule` (a TK *yaml* DFG, `custom_ops/dfg.py:140-143`) but no TclModule. So the documented path needs furiosa-torch ≥ 2026.3.

Other routes:
- `~/.local/bin/furiosa-tcc [--target-npu renegade-8pe[-Nchip]] --output x.edf file.tc` compiles TC → EDF.
- `furiosa.native_torch.ir.Edf.deserialize(bytes)` exists (introspected).
- `EdfModule(edf)` (`custom_ops/edf.py:125`) can run an `ir.Edf` with our tensors.
- **TCL → furiosa-tcc → `EdfModule(ir.Edf.deserialize(open(f,'rb').read())).to(dev)([x, ids, gates, W…])`** is therefore a plausible standalone path. The input order follows the kernel signature, via `edf.input_shapes`.

  Risks [X]:
  - EDF format/version skew between tcc 2026.3 and the native_torch 2026.2 runtime;
  - multi-chip `inter_chip_axes` on I/O;
  - no Python TCL interpreter on CPU in this torch version, so validation has to happen on the NPU.

### Dependencies [V]

- `furiosa.kernels` imports only `furiosa.tcl` and its own modules. There is **no dependency on furiosa_llm internals**.
- furiosa_llm Python does not import `furiosa.kernels`. The kernels are consumed by the native FXB builder (`furiosa.llm_native.build.build_fxb`, `flm build`).
- The Solar kernels assume NVFP4 weights (`u8` packed + `f8_e4` scales + `f32` global) and a 4-chip F-sharding (`Fo` = num_chips, `tcl.all_reduce`). For **bf16 2-card**, write a small custom kernel modeled on `segment_gemm_as_kernel` plus `shared_expert_mlp`, or use bindings with `Fo` = 1.

---

## 3. How the vendor engine runs a decode step

- `LLM.__init__` → `furiosa.llm_native.llm.NativeLLMEngine(model_path|fxb_path, devices, scheduler_config_json, …)` (`SP/furiosa_llm/api.py:428-450, 505-520`).
- The only methods are `generate`, `stream_generate[_mm]`, `encode`, `abort_request`, `shutdown`, `reset_prefix_cache` (`SP/furiosa/llm_native/llm.pyi`).
- Introspection of `furiosa.llm_native`: `build`, `fxb`, `llm`.
- `furiosa.native_runtime.llm`: the same engine classes only.
- `furiosa.native_llm_common`: `NextGenArtifact`, `compiler`, `compute_limits`, `select_target_tasks`, `wire_tasks` (no docstrings).
- **No per-step or "RawModel" API is present.** [V]

Artifact structure (`SP/furiosa_llm/parallelize/pipeline/next_gen.py:123-238`) [V]:
- `Pipeline{tensors (incl. ParamInfo→param_files), stages{…}, blobs}`.
- `Stage.kind ∈ {attention, tokenwise, composed}`. Each stage holds `tasks[(symbol values)] → Task{kind ∈ fx|dfg|kernel|edf|precommandgen+edf, data_blob}`.
- Stages are cut by `LayerRange` (`parallelize/layer_range.py`):
  - attention = `TransformerBlock(i, Attention)`;
  - tokenwise = `OutputProjection(i) … QkvProjection(i+1)`.
  - So **one decode step is ~2 kernels per layer** (≈ 97 for 48 L), not one program. The fused Solar `mid_tokenwise` kernel carries oproj + residual + norm + MoE + next-layer QKV.
- Bucketed tasks are chosen per step by symbol values: `ComposableKernelPipelineBuildConfigGenerator`, `parallelize/pipeline_build_configurer.py:50-115`.
- The `llm_native` .so contains `furiosa-generator/src/next_gen/scheduler/composer/first_fit.rs`, "Composed task", `composable_blobs` and `precommandgen`. This suggests the runtime **stitches the per-stage pre-command-generated kernels into one composed submission per iteration**. [I]
- `npu_queue_limit` (`metadata/config_types.py:99,123`) is the hardware task-queue depth. Strings say "npu_queue_limit must be 1 for now" and "Overlap scheduling requires npu_queue_limit == 1". `scheduling_loop_type='overlap'` (:105-113) pre-schedules batch n+1 on the CPU while batch n runs. [V strings]

Loading our own graphs into the engine: only by building a full NextGen artifact through the furiosa-llm builder with a custom model class (the mppp hook) and running it as a whole model. **Not feasible for a CPU/NPU hybrid MoE**: the scheduler and KV manager are closed, and there are no host callbacks between stages.

The usable low-level runtime is **furiosa.torch's** `ir` layer (§4).

---

## 4. Weight residency and per-call overhead

### Vendor engine [V, I]

Weights are pipeline `ParamInfo` tensors loaded once from param files (safetensors) into device DRAM. Graphs take them as inputs bound at load time (`next_gen.py:113-120`).

`embed_all_constants_into_graph` (`artifact/builder.py:285`) is an option to bake constants into the EDF instead. That corresponds to `NpuNode.weight` in furiosa.torch.

### furiosa.torch (our path) [V]

- `CompileModule.forward` (`custom_ops/edf.py:529-538`) calls `module_weight.flatten_inputs`. **Parameters/buffers are just the first EDF inputs** and become resident after `.to(dev)` (`export/exported_program.py:93-173`). Passing weights as explicit inputs and registering them as Parameters are therefore the same mechanism. Constants are baked into `npu_node.weight` and uploaded once (`edf.py:290-317`).
- Every call to `torch.ops.furiosa.edf` (`edf.py:27-85`) does the following:
  - deserializes params;
  - **allocates outputs** (`ir.allocate_edf_outputs`);
  - wraps `RngdTensor`;
  - builds and submits (`ir.Edf._execute_edf`);
  - checks aliases.

  This is most of our ~0.25–0.5 ms fixed cost, plus the separate `.to(dev)` / `.cpu()` copies.
- **Prebuilt program reuse:** `CompileModule.load_and_build_execute_program(inputs, device=)` (`edf.py:333-362, 511-519`) → `ir.LoadedTpProgram` (`native_torch/ir/__init__.pyi:129-148`).
  - Properties: `inputs`, `outputs`, `operands`, `execute_tp_program`, `wait_tp_program`; method `run()`.
  - Vendor usage: `SP/furiosa/torch/_ext/execution_pool.py:15-73` preallocates input buffers, builds the program once, and per call does `i_npu.copy_(i_cpu)` then `await loaded.run()`.
  - **This requires every input to be the same device tensor each call.** That means our K weight-slot inputs must become a resident table plus device gather (§1.4).
- **Program composition:**
  - `ir.sprint.seq(p0,p1)` / `par(p0,p1)` / `lift_hal` (`native_torch/ir/sprint.pyi:10-32`);
  - `TpProgram.write/read/execute/run/run_blocking`;
  - `memory.copy(src,dst)` returns a TpProgram (`custom_ops/copy_from.py:46-48`);
  - `hal.seq/copy/copy_dtod/execute` (`ir/hal/__init__.pyi`);
  - `Device.execute(program)` (`native_torch/device.pyi`).
  - In principle one submit can be `seq(write x/ids/g, execute EDF_moe_l, read out)`, or several layers' EDFs chained. [X] `hal.seq` documents `ProgramTakenError`, so programs may be single-use and need rebuilding or caching per call.
- DRAM↔DRAM copy (`hal.copy_dtod`, used by rngd `cat/stack` in `custom_ops/cat_stack.py:53-75`) works without compilation. It is an option for **hot-swapping experts into table rows**: `table[slot].copy_(new_expert_cpu)` goes through `aten_copy_from` → `memory.copy` (`copy_from.py:28-49`). Slice-view writes into a large table are [X].

---

## 5. Partial residency and host-side expert compute in vendor code

- `ExpertParallelMoEOp` (`moe.py:440-648`): despite the name it is **single-device weight-stationary**. It scatters tokens into an `(E, L=T, D)` buffer with `index_put_`, runs a dense bmm over all E, and gathers back with `index_select`. There is no cross-device or host component. It only shows that `index_put` / `index_select` with int64 `flat_index` appear in compiled vendor graphs (int64 here, after `.to(torch.long)`).
- mppp devices: `cpu[:N]` is a valid kind (`parallelize/mppp/config.py:84-90`), and `UnitDevice.to_torch_device` maps it. The legacy pipeline spec allows `SuperTaskKind` FX tasks on `cpu` (`parallelize/pipeline/pipeline.md`).
- The **next-gen builder compiles every task to EDF** and raises on non-FX input kinds (`parallelize/new_pipeline_builder.py:1289-1293`). The engine has no CPU-task execution path that we could find.
- There is no expert offload, partial residency or EP. This is consistent with the SDK doc note in PLAN.md §6, which says offload is KV-only.
- The only "sparse residency" primitives are the gather masks:
  - `sparse_select` `dummy_index` (skip rows);
  - TCL `gather(valid_length=…)` (skip a suffix).

  Both fit "hot slots + cold → dummy".

---

## 6. Recommendations for our runtime, by priority

| # | Action | Expected gain | Risk |
|---|---|---|---|
| **P1** | **Re-diagnose the device gather.** Print `e.__cause__`. Use the §1.4 graph: int32 ids, a single index, the table as a Parameter, the TokenWise bmm form. Compile on a single chip without Mesh/fusion first, then fusion8, then Mesh(2). If `aten.index` fails, try `torch.ops.furiosa.sparse_select_v1_5` (paged-KV form), then `torch.gather`. | Unlocks plan B (resident table). Removes host slot marshalling and the K-bucketing of weight inputs. Enables P2. | Low code risk. The compiler binding is furiosa-torch 2026.2, not the vendor's LLM compiler, so coverage may differ. |
| **P2** | **Prebuilt program per (T,K) bucket**, `ExecutionPool`-style: `prog = cm.load_and_build_execute_program(fixed_inputs)`. Per layer: `x_dev.copy_(x)`, `ids_dev.copy_(ids)`, `g_dev.copy_(g)`, `await prog.run()`. Needs fixed inputs, so it depends on P1. Per-layer tables need a per-layer program, or one program per layer that is bound once at init (48 × buckets). | Removes per-call alloc, deserialize and build. Likely a large share of the 0.25–0.5 ms fixed cost (estimate) [X]. | Medium. `_ext/execution_pool.py` is internal (underscore module) and `run()` is async. Watch output aliasing: outputs are reused buffers, so copy them out before the next run. |
| **P3** | **Single submit per layer** with `ir.sprint.seq(memory.copy(x→dev), …, loaded.execute_tp_program, memory.copy(out→host))`, or chain the next layer's dense + MoE EDFs when no CPU result is needed in between. | Removes several host round trips per layer. Moves toward the vendor's "composed task". | High. These are undocumented pyo3 APIs, and programs may be single-use (`ProgramTakenError`). Prototype on a toy graph first. |
| **P4** | **Fuse per-layer graphs** like vendor `mid_tokenwise`: `[prev MoE combine + residual → oproj/norm → router/top-k (device, int32) → hot gather-FFN + shared]` as one EDF. The host only gets ids, for its cold set, plus partial outputs. | Matches the vendor stage cut (2 kernels per layer). Halves launches. | Medium. Top-k (`aten::topk`) is supported. Remapping global id → hot slot needs a resident `[E]→slot` int32 lookup table, which is itself a gather. |
| **P5** | **TCL route** for the expert FFN. Write a bf16 variant of `segment_gemm_as_kernel` / Solar `activation_stationary_moe_nvfp4_w4a16` using `tcl.gather(..., valid_length=n_hot)`. Compile with `furiosa-tcc` and run via `EdfModule(ir.Edf.deserialize(...))`, or upgrade furiosa-torch to 2026.3 for `TclModule`. | Vendor-grade AS MoE kernel, sparse gather, explicit `config` tactic hints (`solar_pro3…/config.py:_apply_as_tokenwise`: `lowering_mode='Optimal'` for T < 8, `ForLlmModelIOBound`, `dma_preference 0.8`, `enable_vrf_half_mode`). | Medium to high. Version skew (torch 2026.2 vs tcc 2026.3). PLAN already flags the combination as unsupported. Upgrading the whole SDK to one release is the cleanest fix. |
| **P6** | **Consider NVFP4 hot experts.** The vendor Solar kernels exist for W4A16 and fit about 4× more experts per GB. | The hit rate rises sharply at the same HBM. | Accuracy is fine (it is the vendor's shipping format). It needs P5 and a CPU-side NVFP4 or bf16 copy for cold experts. |
| **P7** | **Expert hot-swap** via in-place row writes into the resident table (`aten_copy_from` → `memory.copy`). No recompile is needed because the table is an input. | Enables P6 (LFU-decay) in PLAN. | Low to medium. Concurrent write vs a running program needs ordering; use `sprint.seq` or a blocking copy. |

### Explicit non-recommendations

- Do not try to drive `NativeLLMEngine` or the FXB runtime for the hybrid. It has no per-step hooks, no RawModel in the installed wheels, and the scheduler and KV are closed.
- Do not use eager `torch.ops.furiosa.sparse_select` / `gather_i32` on device tensors. They recompile on every call because the indices become buffers.
- `ExpertParallelMoEOp` is not an EP or offload mechanism.

### Stability notes

- The `furiosa.native_torch.ir.*` / `sprint` / `hal` stubs are pyo3-generated and undocumented. `_ext/execution_pool.py`, `ir.Edf._execute_edf` and `RngdTensor._with_inter_chip_axes` are private.
- The current wheel set mixes 2026.1, 2026.2 and 2026.3 components. EDFs from `native_common.compiler` (2026.3) or `furiosa-tcc` (2026.3) are not guaranteed to load in `native_torch.ir.Edf.deserialize` (2026.2).
