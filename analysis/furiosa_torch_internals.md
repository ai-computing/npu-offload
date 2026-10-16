# furiosa.torch internals: compile caching, the call path, chaining, ops, native symbols

Scope: read-only analysis of `~/.local/lib/python3.10/site-packages/furiosa/torch` (furiosa-torch 2026.2.0),
`furiosa/native_torch*.so` plus its `.pyi` stubs, and `furiosa_torch_ext` 2026.3.0. Nothing ran on the NPUs.
Python introspection imported only `torch` + `furiosa.native_torch` (no `get_device`).

Abbreviations: `FT = ~/.local/lib/python3.10/site-packages/furiosa/torch`,
`NT = ~/.local/lib/python3.10/site-packages/furiosa/native_torch` (stubs), `NAK = ~/workspace/new_atten_kernel`.

> Environment pitfall: `python3 -I` (or `-s`) skips user site-packages, so it loads the **old system
> install** `/usr/local/lib/python3.10/dist-packages/furiosa` (furiosa-torch 2026.1.0, a different
> `native_torch.so` without `LoadedTpProgram`, `Edf.build_execute_tp_program`, ...). Always run
> without `-I`/`-s`, and avoid mixing the two installs.

---

## TL;DR (prioritized recommendations)

| # | Change | Fixes | Effort | Confidence |
|---|---|---|---|---|
| 1 | **Persist compiled graphs to disk**: `m.edf.serialize()` + `torch.save(m.module_weight)`, then reload with `ir.Edf.deserialize` → `CompileModule(EdfModule(edf), weight)`. Already shown in `NAK/rngd_llmd/serving/artifact_cache.py` (fresh-process reload, identical output). | (a) Relaunch goes from about 25 min to a few seconds | 30 lines | **Verified in prior project** |
| 2 | **Compile each unique graph once, then load it on every card.** `runtime.py` compiles the `exp` graphs separately for each `Npu` (`runtime.py:145-147`, a per-instance `self.g`). The EDF does not depend on the device (compile runs on CPU tensors). Wrap the same `ir.Edf` in one `EdfModule` per device. | (a) Cold compile: 24 to 15 graphs | 10 lines | High |
| 3 | Optionally enable the **native compile cache**: `furiosa.torch.config.set_cache_config(CacheConfig(root_dir=...))` (default `root_dir=None` means off). | (a) | 1 line | Medium. The cache code is in the binary but no Python caller uses it. Test it |
| 4 | Compile cold graphs **in parallel subprocesses**. Compile is CPU-only; send back `edf.serialize()` bytes. | (a) Cold-build wall time | 40 lines | Medium (must not open a device in workers) |
| 5 | **Bypass the per-call Python/custom-op path**: build an `ir.LoadedTpProgram` once per (graph, fixed input set) with `ir.Edf.build_execute_tp_program(...)`, using preallocated device input buffers (`dev_buf.copy_(cpu)`) and preallocated outputs. This is what the vendor's `furiosa/torch/_ext/execution_pool.py` does. | (b) Removes pytree flatten, `torch.library` dispatch, param deserialize, output alloc, stack alloc and binary reload on every call | 60 lines | Medium-high: the API is used by vendor code, but nobody has timed it |
| 6 | **Use `await lp.run()` (asyncio) instead of threads.** `LoadedTpProgram.run()` returns an awaitable (`execution_pool.py:61-73`). Use `asyncio.gather` over the two cards and `run_in_executor` for CPU experts. | (d) True async submit/wait | Medium | Medium |
| 7 | **Keep activations on the device.** `h2` comes out of `G_post` on rngd:0, but `runtime.py` calls `.cpu()` on it (`runtime.py:235`) and then re-uploads it with `npu.put(h2)` (`runtime.py:184`). Feed the device tensor straight into `G_exp` on device 0. Preallocate host outputs and use `out_cpu.copy_(dev)` instead of `.cpu()`. | (b) Fewer D2H/H2D per layer | Small | High |
| 8 | Experimental: **chain several programs into one submit** with `ir.sprint.seq(p0, p1)` / `par`, using `TpProgram`s from `memory.copy(...)` (H2D/D2H) and `LoadedTpProgram.execute_tp_program` / `.wait_tp_program`. | (b) One submit for H2D→G→D2H, or G1→G2 | Medium-high | Low-medium (undocumented) |
| 9 | Device gather: the compiler lists `aten::index_select`, `aten::index.Tensor` and `aten::gather` as supported. Read `e.__cause__` to get the real error, then try int32 indices on a stacked `(E,F,D)` weight with `W[ids]`, as vendor `TokenWiseMoEOp` does. Fallback: one-hot matmul. | (c) | Experiment | Low-medium |

---

## 1. Compile pipeline and how to persist compiled graphs

### 1.1 Pipeline (file:line)

1. `CompileModule.from_module(m, args)`: `FT/custom_ops/edf.py:483-494`
   - `maybe_wrap_function(m)`, then **`torch.export.export(module, args, kwargs, strict=False)`** (`edf.py:493`)
2. `CompileModule.from_exported(ep)`: `edf.py:464-481`
   - applies `PASSES` (`FT/export/passes.py:458-462`):
     - `decompose` (`passes.py:182-252`) loops until the op set stops changing. Each pass does post-dispatch **re-export** (`torch.export._trace._export(..., pre_dispatch=False)`, `passes.py:72`) and then `ep.run_decompositions(STD_DECOMPOSITIONS)` (`passes.py:211`). This runs several full re-traces, so it is a noticeable Python-side cost for large graphs.
     - `remove_tensor_metadata_assertions`, `fill_tensor_meta_for_aggregates`
   - `compiler.compile(exported, compiler_config)`: `FT/compiler/__init__.py:60-77`
     - `_extract_dram_shape_guides(ep)` (`__init__.py:34-57`) gets layout hints only from RngdTensor example inputs. Your inputs are CPU zeros, so it returns nothing and the result does not depend on the device.
     - `furiosa.native_torch.compiler.compile(ep, compiler_config=..., dram_shape_guides=...)` (native, `NT/compiler.pyi:75`). It returns `ir.Edf`.
     - **Every exception is re-raised as `UnsupportedOpError("failed to compile the graph")`** (`__init__.py:75-77`). The real reason is in `e.__cause__`.
   - `EdfModule(edf)` (`edf.py:131-258`) pulls binaries, weight scatter, stack sizes and io block maps out of `edf.npu_node`.
   - `ExportedProgramWeight(exported)` (`FT/export/exported_program.py:93-109`) holds the parameters and buffers plus the pytree in/out spec. It is pure torch and picklable.
3. `m.to("rngd:0")` (`edf.py:521-532`) runs `EdfModule.to` (`edf.py:260-298`), which calls `ir.Edf.load_weight_and_binaries(cpu_weight, weight_scatter, cpu_binaries, device.index)` and uploads the binaries to device DRAM. Notes:
   - `EdfModule.to` **raises if the module is already loaded on another device** (`edf.py:282-284`). Use one `EdfModule` per device.
   - `CompileModule.to` **returns None** (no `return self`, `edf.py:521-532`), so `CompileModule.from_module(...).to(dev)` gives `None`.

The native stage names come from `NAK/docs/integration/wire-chain-mechanism.md`: dfg → primitive → kernelized → prelower → postlower → precommandgen → commandgraph → tiledgraph → ldfg → cdfg → lir → resourcelir → edf. Most of the time goes to postlower and tactic search (`NAK/docs/book/14-performance.md` §4).

### 1.2 What exists for serialization and caching

| Facility | Where | Status |
|---|---|---|
| `ir.Edf.serialize() -> bytes`, `ir.Edf.deserialize(bytes) -> Edf`, `Edf.__getstate__/__setstate__` | `NT/ir/__init__.pyi:75-81, 93-94`; present in the 2026.2.0 `.so` (checked with `dir(ir.Edf)`) | **Works.** Prior project round-tripped it in a fresh process |
| `ir.Dfg.serialize_to_str / deserialize_from_str` | `NT/ir/__init__.pyi:21-23` | For the lowered DFG (not executable) |
| `CompileModule` pickling | `torch.save(module)` fails because native `ir.Scatter` is not picklable (`NAK/rngd_llmd/serving/artifact_cache.py:3`) | Save the two parts separately |
| **Native compile cache** `furiosa.native_torch.config.CacheConfig(root_dir, summary)`, `set_cache_config` / `get_cache_config` | `NT/config.pyi:8-21,53`; wrappers in `FT/config.py:100-149` (`cache_context`, `with_context(cache_dir=...)` at `config.py:227-236`) | Default is `CacheConfig { root_dir: None }` (introspected), so it is off. The binary contains `furiosa-torch/src/compiler/cache.rs` with `run_cached::{compute_cache_key, read_cache, write_cache, validate_cache_collision}` and files `key.bin`/`output.bin` plus a cache lock, all next to `furiosa_torch::compiler::compile_to_wrapped`. **Nothing in furiosa.torch Python sets it.** This looks like a real cross-process compile cache keyed on the serialized compile input. Not tested; the key probably comes after `torch.export`+`decompose`, so export cost stays |
| Env vars | Only `FURIOSA_TORCH_LOG` in furiosa.torch. Compiler-internal strings: `NPU_COMPILER_CONFIG_PATH`, `NPU_GLOBAL_CONFIG_PATH`, `TACTIC_HINT_PATH`, `SELECTED_SERIALIZED_TACTIC_PATH`, `TACTIC_CONTEXT_PATH`, `SCHEDULE_CONTEXT_PATH`, `ESTIMATION_CACHE`, `HAND_TACTIC_PATH`, `NPU_IR_SERDE_VERIFICATION` | No `FURIOSA_*CACHE*` env var. `SELECTED_SERIALIZED_TACTIC_PATH` / `TACTIC_CONTEXT_PATH` / `ESTIMATION_CACHE` hint at reusing tactic-search results, but their format is unknown |
| furiosa-llm cache (`~/.cache/furiosa/llm/...`, `nlc.compiler.compile(cache_dir=, cache_id=)`) | `NAK/docs/book/10-hidden-apis.md` §10-10 | Separate pipeline, not used by `CompileModule` |

### 1.3 Recipe: persistent artifact cache (do this first)

Adapted from `NAK/rngd_llmd/serving/artifact_cache.py:28-65`, which `probe_artifact_cache.py` verified: fresh-process load about 0 s, identical output checksum.

```python
# graph_cache.py
import os, hashlib, torch
from furiosa.native_torch import ir
from furiosa.torch.custom_ops import CompileModule
from furiosa.torch.custom_ops.edf import EdfModule

CACHE = os.path.expanduser("~/.cache/solar_hybrid_edf")

def key(kind, T, K, cfg, fusion=8, mesh=2, extra=""):
    import furiosa.native_torch.compiler as C
    s = repr((kind, T, K, cfg.H, cfg.F, cfg.nh, cfg.nkv, cfg.hd, cfg.eps, fusion, mesh,
              C.version(), torch.__version__, extra))          # include compiler version!
    return hashlib.blake2b(s.encode(), digest_size=12).hexdigest()

def save(m: CompileModule, path):
    with open(path + ".edf.tmp", "wb") as f: f.write(m.edf.serialize())
    os.replace(path + ".edf.tmp", path + ".edf")
    torch.save(m.module_weight, path + ".w.tmp"); os.replace(path + ".w.tmp", path + ".w")

def load_edf(path):
    with open(path + ".edf", "rb") as f:
        return ir.Edf.deserialize(f.read()), torch.load(path + ".w", weights_only=False)

def instantiate(edf, weight, dev):
    # one EdfModule per device (EdfModule.to refuses a second device, edf.py:282-284)
    m = CompileModule(EdfModule(edf), weight)
    m.to(dev)                       # uploads binaries; returns None
    return m

def get(kind, T, K, cfg, build_module_and_args, devs):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, key(kind, T, K, cfg))
    if not os.path.exists(p + ".edf"):
        mod, args = build_module_and_args()
        m = CompileModule.from_module(mod, args)       # CPU-only compile
        save(m, p)
    edf, w = load_edf(p)
    # each device gets its own EdfModule; deserialize again per device to be safe
    return {d: instantiate(ir.Edf.deserialize(edf.serialize()) if i else edf, w, d)
            for i, d in enumerate(devs)}
```

In `runtime.py`, move the graph dict out of `Npu` into a shared compile cache keyed by `(kind,T,K)` and instantiate per device. That alone removes 9 duplicate `exp` compiles (24 → 15). With the disk cache, relaunch costs only `deserialize` plus binary upload.

Also include in the key: `nd.get_fusion()`, `nd.get_mesh_kind()`, the compiler `Config` (`repr(compiler.Config(...))`) and the module source (e.g. a hash of `inspect.getsource(GExp)`). Weights are runtime inputs, so they don't belong in the key. Graph constants like `eps` and `F` do.

### 1.4 Optional: native cache (one line, unverified)

```python
from furiosa.torch.config import set_cache_config, CacheConfig
set_cache_config(CacheConfig(root_dir=os.path.expanduser("~/.cache/furiosa_torch_compile"), summary=True))
```
Set it before any `from_module`. If `key.bin`/`output.bin` files appear under the dir and a second process compiles faster, it works. Item 1.3 is still more robust because it also skips `torch.export` and the decompose passes.

### 1.5 Parallel cold compile

`compiler.compile` runs on CPU tensors (`ensure_exported_program_cpu_tensor`, `FT/compiler/__init__.py:67`), and the result does not depend on the device (§1.1). A `ProcessPoolExecutor` (spawn) worker can:

1. `import torch, furiosa.torch; nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))`
2. `m = CompileModule.from_module(...)` (do not call `.to(...)` and do not call `nd.get_device`)
3. return `m.edf.serialize()` and `torch.save`-bytes of `m.module_weight`

`_COMPILE_LOCK` in `runtime.py:62` exists only because `torch.export` is not thread-safe; processes avoid that. Caveat: I couldn't verify that `set_fusion`/`set_mesh_kind` never touch `/dev/rngd*`. Test the worker while the cards are free and check that the device isn't opened (e.g. `lsof /dev/rngd*`). The native compiler may already be multithreaded, so the speedup could be under N×.

### 1.6 Compile-time knobs (untested effect on time)

`furiosa.torch.compiler.Config(...)` (`NT/compiler.pyi:9-58`), passed via `from_module(..., compiler_config=cfg)`. The defaults (introspected) are `enable_tactic_pruning=True, tactic_sorting_policy=Random, tactic_hint=Default`. Options to try:
- `tactic_hint=TacticHintConfig.ForLlmModelDecode` for T∈{1,8,16} (the prior project used it for decode graphs, `NAK/rngd_llmd/runtime/engine_wired_generic.py:117`)
- `tactic_sorting_policy=TacticSortingPolicy.ByEstimation`

`GExp` is unrolled over K slots (K=32 means 64 matmuls and 66 inputs), and compile time grows with K. Fewer K buckets, e.g. {8, 32}, means fewer and cheaper graphs.

---

## 2. Execution path of a `CompileModule` call, and where ~0.25 ms goes

### 2.1 Call chain (each step runs on every call)

| Step | Code | Per-call work |
|---|---|---|
| 1 | `CompileModule.forward` `edf.py:534-543` | `module_weight.flatten_inputs`: `fx_pytree.tree_flatten_spec` + `_check_input_value` for every arg + filter (`exported_program.py:35-57`, `179-188`). Cost grows with the number of inputs (GExp K=32 has 66) |
| 2 | `EdfModule.forward` → `_execute` `edf.py:412-445` | `utils.ensure_same_device(inputs)` loops over all inputs. List building, static/passthrough handling, `copies`, `output_index_map` remap |
| 3 | `torch.ops.furiosa.edf(...)` `edf.py:372-391` | **`torch.library.custom_op` dispatch with a Python implementation** and 18 arguments, several of them long `list[int]`. Schema parsing and boxing cost tens of µs |
| 4 | inside the op `edf.py:51-66` | `utils.ensure_rngd_device`; **`ir.deserialize_edf_params(...)`** re-parses shapes, strides, scatter and dtypes **every call** (`edf.py:55`); **`ir.allocate_edf_outputs(...)` allocates fresh device output tensors every call** (`edf.py:58`); `deserialize_inter_chip_axes` + `RngdTensor._with_inter_chip_axes` per output (`edf.py:62-66`) |
| 5 | `ir.Edf._execute_edf(...)` `edf.py:68-83` (native) | Builds operands and commits (cf. `Edf.build_operands_and_commits`, `NT/ir/__init__.pyi:84`), **allocates stack DRAM** (`stack_sizes`; separate stacks are an input to `build_operands_and_commits`, so they are created per call), submits the TP program, **waits until done (blocking)**, then does in-place copy-back (`inplace_inputs_copy_back`, `pyi:86`) |
| 6 | `ir._ensure_no_alias` `edf.py:85` | Alias check |
| 7 | `unflatten_outputs` | pytree unflatten |
| 8 | your `.cpu()` | **New pageable host tensor every call** + `memory.copy(src,dst)` → `TpProgram.run_blocking()` (`FT/custom_ops/copy_from.py:49-50`) |
| 9 | your `npu.put(x)` = `.to(dev)` | **New device allocation every call** + `memory.copy` + `run_blocking` |

Without a profiler run this is inference, but the 0.25 ms device-only number is about what you'd expect from about 5 Python layers, one `torch.library` custom op dispatch, two native deserialize/allocate calls, per-call DRAM allocations for outputs and stacks, and one blocking submit+wait. The NPU kernel time for a tiny graph is a small fraction of that. Host→device→run→host at about 0.5 ms adds one allocation plus one blocking DMA program on each side (about 85 µs per small H2D).

To measure the split, use the vendor profiler (`FT/profiler`, `RNGDRecordFunction` already wraps `furiosa::compile` / `copy`) or `torch.profiler` with the `rngd` device. Also time `ir.Edf._execute_edf` directly and compare with `m(...)`.

### 2.2 Lower-level APIs that already exist (Python-bound)

| Need | API | Ref |
|---|---|---|
| **Prebuilt, reusable program with fixed input/output buffers** | `ir.Edf.build_execute_tp_program(device_index, inputs, binaries(nested), weight, stack_sizes, stack_is_dram_shared, outputs, profiles, io_block_sizes, io_block_is_dram_shared, input_io_indice(flat), output_io_indice(flat)) -> LoadedTpProgram` | `NT/ir/__init__.pyi:90`; used by `EdfModule.load_and_build_execute_program` `edf.py:333-360` and `ExecutionPool` `FT/_ext/execution_pool.py:15-47` |
| Run a prebuilt program **asynchronously** | `await loaded_tp_program.run()` returns the output list (the preallocated `outputs`). The binary contains pyo3 asyncio glue (`create_future`, `call_soon_threadsafe`, `get_running_loop`) | `execution_pool.py:61-73`; `NT/ir/__init__.pyi:129-148` (the stub's `-> list` return type is wrong; it is awaited) |
| Access sub-programs | `LoadedTpProgram.execute_tp_program`, `.wait_tp_program` (TpProgram), `.inputs`, `.outputs`, `.operands` | `pyi:129-148` |
| **Write into an existing device tensor** | `dev_buf.copy_(cpu_t)` → `aten_copy_from` → `memory.copy(src,dst).run_blocking()`, with no allocation | `copy_from.py:28-51` |
| Async/chainable copies | `furiosa.native_torch.memory.copy(src, dst) -> TpProgram` (not in stubs; seen in `dir()`); `TpProgram.read/write/copy_intra_tp` | `copy_from.py:49`; `NT/ir/sprint.pyi:10-26` |
| TpProgram run modes | `TpProgram.run()` (async), `TpProgram.run_blocking()` | `sprint.pyi:23-24` |
| Raw HAL programs | `ir.hal.copy`, `copy_dtod`, `execute(pe_stmt)`, `seq([...])`; `Device.execute(program)` (awaitable, see docstring example `await native_torch.get_device(i).execute(program)`), `Device.execute_blocking` | `NT/ir/hal/__init__.pyi:18-71`, `NT/device.pyi:22-43`, `NT/ir/__init__.pyi:255-278` |
| Output into a caller-chosen tensor | The `outputs=` argument of `build_execute_tp_program` (e.g. a persistent device tensor) | `pyi:90` |
| In-place update of an input | Write the graph so it mutates an input (`buf.copy_(...)`, `index_put_`, slice assign). Export adds copy nodes (`_insert_copy_for_mutations`, `passes.py:77,236`) and the EDF lists `inplace_update_inputs`. `_execute_edf` copies back. The LoadedTpProgram path does **not** copy back | `edf.py:44,185`, `pyi:86` |
| Dram-shared alloc (multi-chip mesh) | `furiosa.torch.utils.empty_strided_dram_shared(...)`, `with_dram_shared(bool)` | `FT/utils/__init__.py:259-283`; `execution_pool.py:28-34` uses it when `num_chips > 1` |
| Pinned host memory / zero-copy | **No Python API.** `Allocator` only does `inst_alloc` / `dram_alloc` (`NT/memory.pyi`). The binary contains `HugePageCpuProxyDma` and `Non-optimal DMA unit size of`. My inference is that host DMA goes through a hugepage proxy/bounce buffer, which would explain the 3.2 GB/s drop above 16 MB. You cannot register your own pinned buffer from Python. Practical fix: keep large tensors on the device, or split D2H into ≤16 MB chunks and overlap them |

### 2.3 Recipe: fast path with prebuilt `LoadedTpProgram`s

Use this for graphs whose input set is fixed per (layer, T): `G_qkv` and `G_post`, with per-layer weights passed as inputs.

```python
import asyncio, torch
from furiosa.native_torch import ir
from furiosa.torch import native_device as nd, utils

class FastEdf:
    """Per (CompileModule, device): binaries uploaded once, programs prebuilt per fixed input set."""
    def __init__(self, cm, dev):
        em = cm.edf_module; e = em.edf
        # guard: only the simple case (all inputs go to the NPU node; no static/passthrough/copies/inplace)
        n_in = len(e.input_shapes)
        assert list(em.input_indice_of_edf) == list(range(n_in)), em.input_indice_of_edf
        assert not em.copies and not em.passthrough_pair and not em.static_outputs
        assert not em.inplace_update_inputs
        assert em.output_index_map == list(range(em.num_outputs))
        assert list(em.output_indice_of_edf) == list(range(em.num_outputs))
        self.em, self.dev = em, torch.device(dev)
        # upload ONCE and keep the NESTED binaries (load_and_build_execute_program re-uploads every time!)
        self.weight, self.bins = ir.Edf.load_weight_and_binaries(
            em.cpu_weight, em.weight_scatter, em.cpu_binaries, self.dev.index)
        self.multi_chip = nd.get_device(self.dev.index).num_chips > 1

    def empty_input(self, i):
        s = self.em.edf.input_shapes[i]
        if self.multi_chip:
            return utils.empty_strided_dram_shared(size=s.size, stride=s.stride, dtype=s.dtype, device=self.dev)
        return torch.empty_strided(s.size, s.stride, dtype=s.dtype, device=self.dev)

    def build(self, inputs, outputs=None):
        em = self.em
        outputs = outputs if outputs is not None else em._create_outputs(self.dev)   # edf.py:310-331
        return ir.Edf.build_execute_tp_program(
            self.dev.index, list(inputs), self.bins, self.weight,
            em.stack_sizes, em.stack_is_dram_shared, outputs, None,
            em.io_block_sizes, em.io_block_is_dram_shared,
            em.input_io_indice, em.output_io_indice)

# setup (once): per layer l and bucket T
#   fq = FastEdf(qkv_cm[T], "rngd:0"); xbuf = fq.empty_input(0)
#   lp_qkv[l][T] = fq.build([xbuf, dd[l]["in_norm"], dd[l]["wqkv"]])
#   host_out = torch.empty(T, nqkv, dtype=BF)     # preallocated host result
# per call:
#   xbuf.copy_(x_cpu)                      # H2D into existing buffer, no alloc
#   (out,) = await lp_qkv[l][T].run()      # no flatten / custom-op / alloc / deserialize
#   host_out.copy_(out)                    # D2H into existing host tensor, no alloc
```

Notes and caveats:
- `outputs` are reused, so copy them out or consume them before the next `run()` of the same program.
- The weights you pass (`dd[l][...]`, resident experts) are the same device tensors the normal path gets, so layout matches. Check bit-exactness against `m(...)` once.
- For **expert graphs** the inputs change on every call (a different expert set). Choices:
  (i) Build a program per call with the actual expert tensors. `build_execute_tp_program` with pre-uploaded binaries should be cheap since it only allocates stacks and an operand table, but time it.
  (ii) Cache programs per `(l, tuple(experts))` with an LRU.
  (iii) Rework into one program per resident expert. Then chain the selected ones (§3) and accumulate in place.
- If `run()` must be called from a non-async thread, use one dedicated asyncio loop thread and `asyncio.run_coroutine_threadsafe(lp.run(), loop).result()`.
- This removes steps 1-4 and 7-9 of §2.1, plus per-call stack allocation and binary reload. Expect the remaining per-call cost to be the native submit+wait.

### 2.4 Overlap without threads

```python
async def moe_step(...):
    cpu_fut = loop.run_in_executor(cpu_pool, cpu_experts, ...)     # torch ops release the GIL
    r0, r1 = await asyncio.gather(lp_dev0.run(), lp_dev1.run())    # both cards in flight
    cpu = await cpu_fut
```
Right now each `m(...)` blocks inside native code. I couldn't confirm that `_execute_edf` releases the GIL. If it doesn't, the current Python threads serialize host-side work.

---

## 3. Chaining several graphs into one submit

### 3.1 What "wired dispatch" in the prior project is

It is **not** a runtime API. "Wired EDF" is what `CompileModule.from_module` produces for a whole multi-op graph: the compiler chains all of its tasks inside one EDF (`task_num=N`). One host submit runs all N tasks (`NAK/docs/integration/wire-chain-mechanism.md` TL;DR; `NAK/docs/book/06-l5-compile-module.md`).

`NAK/rngd_llmd/runtime/engine_wired_generic.py` and `llama_engine_resident.py` work like this:
- One `CompileModule` covers **all layers** of a dense model: `WireModel` loops over `WireLayer`s (`engine_wired_generic.py:59-68`). Per-layer weights are stacked `[nl, ...]` device tensors passed as **runtime inputs** (`:91-104`). Compile config: `allow_external_operators=True, allow_unlowered_operators=True, tactic_hint=ForLlmModelDecode` (`:117`).
- Custom `TacticKernelModule` DSL kernels (rmsnorm, softmax, zip_add, swiglu) are inlined (`:38-43`).
- **The KV cache stays on the device.** The graph returns the full updated `Kf, Vf`, and the next token passes those device tensors back in (`:53-55, 124-131`). KV insert at position `pos` uses a **one-hot multiply-add** (`Kp + k ⊗ oh`), not scatter (`:52-53`).
- Each token makes one `m(...)` call, and only the hidden state comes back to host for lm_head/argmax.
- Results: Qwen2.5-0.5B **6.4 ms/token**, Llama-3.1-8B **16.3 ms/token** resident-KV (`NAK/rngd_llmd/README.md:16,60-61`). The single-layer microbench was **2.42 ms** with `set_fusion(8)+Mesh(2)` (`NAK/docs/book/14-performance.md` §1). The whole-model 8B graph took about 1800 s to compile (`NAK/docs/integration/llmd-integration-plan.md:160`), which is why `artifact_cache.py` was written.

For Solar's hybrid, the host attention and routing between graphs prevents one whole-layer graph. You can still cut round trips:
- Fold `G_qkv(l+1)` into the same submit as the tail of layer l. Not possible as long as the MoE result is combined on host.
- Keep `x` and `h2` on device 0 between `G_post` and `G_exp` (recommendation 7).

### 3.2 Runtime-level chaining primitives (bound in Python, undocumented)

- `furiosa.native_torch.ir.sprint.seq(p0, p1) -> TpProgram` and `par(p0, p1)` (`NT/ir/sprint.pyi:30-32`), plus `lift_hal(device_index, hal_program)` (`:28`)
- `TpProgram.execute(device_index, binaries, operands, profiles=None, indirect=None)`, `.write/.read/.copy_intra_tp`, `.wait_for_each_chip(device_index, commits)` (`sprint.pyi:10-22`)
- `LoadedTpProgram.execute_tp_program` / `.wait_tp_program` (`NT/ir/__init__.pyi:145-147`)
- `furiosa.native_torch.memory.copy(src, dst) -> TpProgram` (H2D/D2H/D2D)
- HAL level: `ir.hal.seq([...])`, `ir.hal.pert.{seq, par, launch, wait, notify, indirect_execute, load_inst, let_in}` (`NT/ir/hal/__init__.pyi:59`, `NT/ir/hal/pert/__init__.pyi`)

Untested sketch of one submit for H2D, graph A, graph B and D2H, where B's inputs are A's preallocated outputs:
```python
from furiosa.native_torch import memory
from furiosa.native_torch.ir import sprint
lpA = fa.build([xbuf, w1]);  lpB = fb.build([lpA.outputs[0], w2])
def S(*ps):
    p = ps[0]
    for q in ps[1:]: p = sprint.seq(p, q)
    return p
prog = S(memory.copy(x_cpu, xbuf),
         lpA.execute_tp_program, lpA.wait_tp_program,
         lpB.execute_tp_program, lpB.wait_tp_program,
         memory.copy(lpB.outputs[0], y_cpu))
await prog.run()     # or prog.run_blocking()
```
Unknowns: whether `execute_tp_program`/`wait_tp_program` can be reused after a run (`ProgramTakenError` exists for HAL programs, so a TpProgram may be consumed and need rebuilding per call). Also whether `seq` needs the explicit wait program between dependent executes. Try it on tiny graphs and compare outputs against `m(...)` first.

---

## 4. Ops support, gather, and custom-op hooks

### 4.1 Op tables
- Natively supported aten ops: docstring of `furiosa.native_torch.compiler.is_supported_aten` (`NT/compiler.pyi:266-371`). It includes **`aten::index.Tensor` (296), `aten::gather` (337), `aten::index_select` (361)**, `index_put`, `scatter.src`, `slice_scatter`, `topk`, `argmax`, `cumsum`, `where`, `mm/bmm`.
- Importable ops: `is_importable` (`NT/compiler.pyi:90-263`). This also lists `aten::embedding` paths, `scatter_add`, `sort`, `nonzero`, and the furiosa ops `furiosa::gather_i32`, `furiosa::sparse_select(_v1_5)`, `furiosa::dequantize`, `furiosa::dfg`, `furiosa::edf`.
- Python decomposition DB: `FT/db/decomp_rules_db.py` (5953 lines) maps e.g. `aten.embedding.default → aten::index.Tensor` (`:1294`) and `aten.index_select.default → aten::index.Tensor` (`:2023`). Local decompositions are in `FT/db/local_decomp.py` (`embedding_bag` at `:361-507`), with `STD_DECOMPOSITIONS` in `FT/db/std_decompositions.py`. You can check this at runtime with `furiosa.torch.compiler.is_supported_aten("aten::index_select")`.

### 4.2 Why your gather fails, and what to try
- `UnsupportedOpError` is a **catch-all wrapper** (`FT/compiler/__init__.py:75-77`), so print `traceback.print_exception(e.__cause__)`. Compiler strings show specific gather failure modes: `Gather.indices has invalid element type (`, `Failed to populate gather tactic:`, `not implemented: Gather: input/output type`, `is expected as gather operation but the data source(`, plus passes `gather_index_downcast.rs`, `gather_reifier.rs`, `gather_scatter_kernelizer.rs`, and `DmaCommandGather` / `_indirect_gather_output`.
- Vendor MoE code **does gather weights on device**: `w1[expert_ids]` on stacked `(E,F,D)` weights (`furiosa/models/core/operators/moe.py:316-318`). `BlockwiseMoEOp._gather_weights` is `w[gather_idx]` (`moe.py:998-1024`), and ids are cast to **int32** (`moe.py:601, 899`). That code is compiled by the furiosa-llm pipeline with LLM compiler configs, so it is evidence that runtime-index weight gather lowers in some configurations.
- Things to try, cheapest first:
  1. `W[ids.to(torch.int32)]` with `W` stacked as `(E, F, D)` (dim-0 gather of whole rows), `Config(allow_unlowered_operators=True, allow_external_operators=True, tactic_hint=ForLlmModelDecode)`
  2. `torch.gather` / `index_select` on a 2-D view
  3. **One-hot matmul**: `onehot(ids)[T,E] @ W.view(E, -1)` or the per-slot gate trick you already use. This always lowers, but costs E×(row) bandwidth, so use it only for small tables.
  4. Vendor helper ops `torch.ops.furiosa.gather_i32` / `sparse_select` (`FT/custom_ops/furiosa_internals.py:13-107`). **Their indices are registered as buffers, i.e. compile-time constants** (`:31-34, :81`). They don't help for dynamic routing.
- A device-side expert gather would put all E experts of a layer in device memory and let one graph pick from them. That conflicts with the hot/cold design, where residents are a subset. Host-side slot selection with weights-as-inputs (what `runtime.py` does now) is reasonable. The cost is in per-call overhead (§2), not in the gather.

### 4.3 Custom kernel hooks inside a CompileModule graph
- **`TacticKernelModule(dsl_str)`** (`FT/custom_ops/dfg.py:140-143`) is a `DfgModule` (`dfg.py:102-137`) that emits `torch.ops.furiosa.dfg` (`dfg.py:22-43`). When the module is traced inside `CompileModule.from_module`, `furiosa::dfg` stays in the exported graph as an op the native compiler can import (`NT/compiler.pyi:255`), so the kernel is compiled into the same EDF. (`passes.py:379` `_inline_dfg_tensors_from_user_inputs` is used only by the torch.compile backend, `backend/torch_compile.py:76,125`.) This is the supported way to inject a hand-written kernel (tactic-kernel DSL) into a graph. Working examples: `NAK/rngd_llmd/runtime/dsl_gen.py` and the `WireLayer` modules. A vendor example of a MoE index kernel is `furiosa/models/core/operators/tk_graphs/moe_blockwise_compute_wg_idx.yaml`, loaded via `TacticKernelModule` (`moe.py:691-705`).
- `DfgModule(ir.Dfg.parse(yaml))` takes raw DFG `#naive_yaml` (schema partly reverse-engineered in `wire-chain-mechanism.md`).
- `torch.library` ops: furiosa registers its own ops this way (`furiosa::edf`, `::dfg`, `::gather_i32`, ...). A user `torch.library.custom_op` would be an *unknown* op to the compiler unless it lowers to `furiosa::dfg`, so wrap a `TacticKernelModule` rather than writing a new torch op.
- An EDF (furiosa-opt/visa) kernel cannot be called inside a CompileModule graph. The prior project instead substituted `task_binaries_map` into vendor EDFs (`NAK/docs/book/17-l4-in-furiosa-llm.md`). As a separate step it could be run as its own `EdfModule(ir.Edf.deserialize(...))`, but only if the EDF has the pt-track schema. visa `.edf` fails with "missing field data" (`NAK/docs/integration/visa-edf-compat.md`).

---

## 5. Native symbols (`nm -D --defined-only`)

`native_torch.cpython-310-x86_64-linux-gnu.so` exports a C ABI. The C++ side of the torch PrivateUse1 backend uses it. There are no Python bindings for these names, but most have Python equivalents:

| Symbols | Meaning | Python equivalent |
|---|---|---|
| `device_execute`, `device_execution_wait`, `device_execution_is_completed`, `device_execution_is_success`, `device_execution_destroy` | **Async execute handle (submit → poll/wait)** | `Device.execute` (awaitable) / `TpProgram.run()` / `LoadedTpProgram.run()` |
| `hal_program_copy`, `hal_program_execute`, `hal_program_read_at`, `hal_program_write_at`, `hal_program_seq` | HAL program builders | `ir.hal.copy/execute/seq` |
| `pe_program_launch`, `pe_program_load_inst`, `pe_program_copy_local`, `pe_program_notify`, `pe_program_wait`, `pe_program_waits`, `pe_program_seq` | PE-level statements | `ir.hal.pert.*` |
| `allocator_dram_allocate`, `allocator_dram_shared_allocate`, `allocator_sram_allocate`, `allocator_auto_free`, `_mm_alloc_raw`, `_mm_free_raw`, `_mm_shrink_dram_shared_to_fit` | Device allocators | `Allocator.dram_alloc` (`NT/memory.pyi`), `memory._set_sram_mode`, `_set_dram_shared_mode` |
| `process_group_{new,send,recv,broadcast,allgather,barrier,get_rank,get_world_size}`, `work_{wait,synchronize,is_completed}` | c10d collectives backend (`furiosa-torch/src/cpp/collective_communication`) | Probably reachable via `torch.distributed` with the rngd backend. This could enable **card-to-card send/recv** without a host bounce (not explored) |
| `tch_read_stream_*`, `tch_write_stream_*` | tch-rs serialization streams | none needed |

`native_runtime.so` exports `PyInit_pyo3_async_runtimes` plus llguidance (`llg_*`), which serves the LLM engine and isn't relevant here. No exported symbol mentions a cache or serialization beyond what §1.2 lists. The compile cache is internal Rust (`run_cached`), reachable only through `CacheConfig`.

You could call `device_execute`/`device_execution_wait` through ctypes, but the signatures and the `Program` handle layout are undocumented. The Python `run()` awaitables already provide the same async semantics, so ctypes isn't worth it.

---

## 6. Concrete plan for `solar_hybrid/runtime.py`

1. **Compile cache (today):** move `self.g` from `Npu` to a module-level cache keyed by `(kind,T,K)` plus the §1.3 key, saved and loaded on disk. Instantiate per device with `CompileModule(EdfModule(edf), weight).to(dev)`. Expect a first run of 15 compiles instead of 24, and seconds on later launches.
2. **Avoid round trips (today):** on device 0 pass the device `h2` from `G_post` straight to `G_exp`, and D2H only what the host needs (`h2` for routing, `y`). Upload `x` once per layer and reuse it for qkv and post. Preallocate the `G` gate tensor on each device and `copy_` into it. Use preallocated host output tensors with `.copy_()` instead of `.cpu()`.
3. **Fast path (next):** `FastEdf` + prebuilt `LoadedTpProgram` for `qkv[l][T]` and `post[l][T]` (48×3×2 programs). Time `build_execute_tp_program` to decide between per-call builds and an LRU for expert graphs.
4. **Async (next):** a single asyncio loop that drives `lp.run()` on both cards, with CPU experts in a thread pool. This replaces `threading.Thread` per call.
5. **Experiments (when the cards are free):** native `CacheConfig`; parallel compile in subprocesses; `sprint.seq` chaining; gather with int32 on stacked `(E,F,D)` while reading `e.__cause__`; compiler `tactic_hint=ForLlmModelDecode` and its effect on compile time.

Micro-benchmark to run later on a free card (not executed here):
```python
import time, asyncio, torch, furiosa.torch
from furiosa.torch import native_device as nd
nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))
from furiosa.torch.custom_ops import CompileModule
from furiosa.native_torch import ir
class G(torch.nn.Module):
    def forward(self, x, w): return x @ w.t()
x = torch.zeros(8, 4096, dtype=torch.bfloat16); w = torch.zeros(4096, 4096, dtype=torch.bfloat16)
m = CompileModule.from_module(G(), (x, w)); m.to("rngd:0")
xd, wd = x.to("rngd:0"), w.to("rngd:0")
def bench(f, n=200):
    f(); t = time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter() - t) / n * 1e6
print("module call us", bench(lambda: m(xd, wd)))
em = m.edf_module
wt, bins = ir.Edf.load_weight_and_binaries(em.cpu_weight, em.weight_scatter, em.cpu_binaries, 0)
def build(): return ir.Edf.build_execute_tp_program(0, [xd, wd], bins, wt, em.stack_sizes,
    em.stack_is_dram_shared, em._create_outputs(torch.device("rngd:0")), None,
    em.io_block_sizes, em.io_block_is_dram_shared, em.input_io_indice, em.output_io_indice)
print("build us", bench(build, 50))
lp = build()
loop = asyncio.new_event_loop()
print("prebuilt run us", bench(lambda: loop.run_until_complete(lp.run())))
blob = m.edf.serialize(); t = time.perf_counter(); ir.Edf.deserialize(blob); print("deserialize s", time.perf_counter() - t, len(blob))
```
