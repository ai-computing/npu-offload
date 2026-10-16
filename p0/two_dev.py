import torch, time
import furiosa.torch
from furiosa.torch import native_device as nd
nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))
from furiosa.torch.custom_ops import CompileModule
BF=torch.bfloat16
class M(torch.nn.Module):
    def forward(s,x,w): return x@w.t()
x=torch.randn(1,4096).to(BF); w=(torch.randn(2560,4096)*.02).to(BF)
ref=(x.float()@w.float().t())
for dev in ["rngd:0","rngd:1"]:
    try:
        m=CompileModule.from_module(M(),(x,w)); m.to(dev)
        y=m(x.to(dev),w.to(dev)).cpu().float()
        print(dev,"ok rel",((y-ref).norm()/ref.norm()).item())
    except Exception as e: print(dev,"FAIL",type(e).__name__,str(e)[:200])
