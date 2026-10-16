import torch, time, statistics
import furiosa.torch
from furiosa.torch import native_device as nd
nd.set_fusion(8)
from furiosa.native_torch import memory as M
DEV="rngd:0"
def bench(name, fn, n=50):
    for _ in range(5): fn()
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    print(f"{name:40s} {statistics.median(ts)*1e3:7.3f} ms", flush=True)
for shape in [(8,), (1,8), (1,4096), (8,4096)]:
    src = torch.randn(*shape).to(torch.bfloat16); dst = src.to(DEV)
    bench(f"to(dev) {shape}", lambda: src.to(DEV))
    bench(f"dst.copy_(src) {shape}", lambda: dst.copy_(src))
    bench(f"dst.copy_(src,nb) {shape}", lambda: dst.copy_(src, non_blocking=True))
    try:
        bench(f"memory.copy {shape}", lambda: M.copy(src, dst))
    except Exception as e: print("memory.copy err", str(e)[:150])
    bench(f"dst.cpu() {shape}", lambda: dst.cpu())
print(M.copy.__doc__)
