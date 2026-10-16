import torch, sys
import furiosa.torch
from furiosa.torch import native_device as nd
nd.set_fusion(8)
from furiosa.native_torch import memory as M
print([a for a in dir(M) if not a.startswith('__')])
DEV="rngd:0"
for n in [8, 64, 1024, 4096, 65536, 1<<20, 16<<20]:
    t = torch.zeros(n, dtype=torch.bfloat16).to(DEV)
    try: alloc = M.get_allocator(0) if hasattr(M,'get_allocator') else None
    except Exception: alloc=None
    sh = None
    for fn in ("is_dram_shared",):
        try: sh = alloc.is_dram_shared(t) if alloc else None
        except Exception as e: sh = f"err {e}"
    print(f"bytes={n*2:>10} dram_shared={sh}")
# fill: keep small tensors until OOM
keep=[]; total=0
try:
    for i in range(200000):
        keep.append(torch.zeros(4096, dtype=torch.bfloat16).to(DEV)); total+=8192
except Exception as e:
    print("OOM after", len(keep), "x 8KB =", total/1e6, "MB:", str(e)[:120])
else:
    print("no OOM up to", total/1e6, "MB")
