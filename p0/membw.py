import torch, time, sys
torch.set_num_threads(int(sys.argv[1]))
x = torch.ones(2*1024**3, dtype=torch.float32)  # 8 GB
for _ in range(2): x.sum()
ts=[]
for _ in range(5):
    t=time.perf_counter(); x.sum(); ts.append(time.perf_counter()-t)
print(f"threads={sys.argv[1]} read BW = {x.numel()*4/min(ts)/1e9:.1f} GB/s")
