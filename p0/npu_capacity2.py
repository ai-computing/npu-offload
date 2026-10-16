import sys, time, torch, subprocess
import furiosa.torch
from furiosa.torch import native_device as nd
nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))
from furiosa.torch.custom_ops import CompileModule
BF=torch.bfloat16; H,F=4096,1280; DEV="rngd:0"; N=int(sys.argv[1])
class FFN(torch.nn.Module):
    def forward(self,x,wgu,wd):
        gu=x@wgu.t(); return (torch.nn.functional.silu(gu[:,:F])*gu[:,F:])@wd.t()
x=torch.randn(1,H).to(BF)
pool=[((torch.randn(2*F,H)*.02).to(BF),(torch.randn(H,F)*.02).to(BF)) for _ in range(4)]
m=CompileModule.from_module(FFN(),(x,*pool[0])); m.to(DEV)
res=[(pool[i%4][0].to(DEV),pool[i%4][1].to(DEV)) for i in range(N)]
print(subprocess.run(["furiosa-smi","status"],capture_output=True,text=True).stdout[-600:])
xd=x.to(DEV); bad=0
for j in [0,N//2,N-1]:
    ref=FFN()(x.float(),pool[j%4][0].float(),pool[j%4][1].float())
    got=m(xd,*res[j]).cpu().float(); rel=((got-ref).norm()/ref.norm()).item(); print("expert",j,"rel",round(rel,4))
import statistics
for label, idxs in [("same4", [i%4 for i in range(200)]), ("rotate_all", [(i*7)%N for i in range(200)])]:
    ts=[]
    for j in idxs:
        t=time.perf_counter(); m(xd,*res[j]); ts.append(time.perf_counter()-t)
    print(label, "median ms", round(statistics.median(ts[20:])*1e3,3), "max", round(max(ts[20:])*1e3,3))
