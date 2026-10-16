"""How many Solar experts (31.5MB bf16 each) can be resident on one card via furiosa.torch?"""
import sys, time, torch
import furiosa.torch
if "--fusion" in sys.argv:
    from furiosa.torch import native_device as nd
    nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))
BF=torch.bfloat16; H,F=4096,1280; DEV=sys.argv[1]
wgu=(torch.randn(2*F,H)*.02).to(BF); wd=(torch.randn(H,F)*.02).to(BF)
res=[]; t0=time.time()
try:
    for i in range(1600):
        res.append((wgu.to(DEV), wd.to(DEV)))
        if (i+1)%100==0: print(f"{i+1} experts = {(i+1)*31.46/1000:.1f} GB  {time.time()-t0:.0f}s", flush=True)
except Exception as e:
    print(f"FAILED at {len(res)} experts = {len(res)*31.46/1000:.1f} GB: {type(e).__name__}: {str(e)[:200]}")
print("max resident", len(res))
