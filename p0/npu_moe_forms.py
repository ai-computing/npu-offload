"""Compare NPU MoE execution forms for static-shape AOT:
 A) slots   : K expert weight tensors passed as inputs (host picks; pointer swap), x [T,H], G [T,K] gate matrix (0 = token not routed to slot)
 B) gather  : one resident table Wgu[E,2F,H], Wd[E,H,F]; ids [K] int input -> index_select on device
All T tokens computed against all K slots, masked by G (dense-masked; NPU decode is weight-bound so T is ~free).
"""
import sys, time, statistics, json, torch
import furiosa.torch
from furiosa.torch import native_device as nd
nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))
from furiosa.torch.custom_ops import CompileModule
BF=torch.bfloat16; H,F=4096,1280; DEV="rngd:0"; E=64
import os; IDX=os.environ.get("IDX","sel"); form=sys.argv[1]; Ts=[int(t) for t in sys.argv[2].split(",")]; Ks=[int(k) for k in sys.argv[3].split(",")]
def emit(r):
    print(json.dumps(r),flush=True); open("results/npu_moe_forms.jsonl","a").write(json.dumps(r)+"\n")
def ffn_masked(x,G,wgu,wd,K):
    out=None
    for i in range(K):
        gu=x@wgu[i].t() if wgu.dim()==3 else None
        a=torch.nn.functional.silu(gu[:,:F])*gu[:,F:]
        y=(a@wd[i].t())*G[:,i:i+1]
        out=y if out is None else out+y
    return out
class Gather(torch.nn.Module):
    def __init__(s,K): super().__init__(); s.K=K
    def forward(s,x,G,ids,Wgu,Wd):
        wgu=Wgu[ids.long()] if IDX=="adv" else torch.index_select(Wgu,0,ids); wd=Wd[ids.long()] if IDX=="adv" else torch.index_select(Wd,0,ids)
        return ffn_masked(x,G,wgu,wd,s.K)
class Slots(torch.nn.Module):
    def __init__(s,K): super().__init__(); s.K=K
    def forward(s,x,G,*ws):
        out=None
        for i in range(s.K):
            gu=x@ws[2*i].t(); a=torch.nn.functional.silu(gu[:,:F])*gu[:,F:]
            y=(a@ws[2*i+1].t())*G[:,i:i+1]; out=y if out is None else out+y
        return out
torch.manual_seed(0)
Wgu=(torch.randn(E,2*F,H)*.02).to(BF); Wd=(torch.randn(E,H,F)*.02).to(BF)
if form=="gather": Wgu_d=Wgu.to(DEV); Wd_d=Wd.to(DEV)
else: slots_d=[(Wgu[e].contiguous().to(DEV),Wd[e].contiguous().to(DEV)) for e in range(E)]
for T in Ts:
  for K in Ks:
    x=torch.randn(T,H).to(BF); G=(torch.rand(T,K)<0.5).to(BF)*0.1
    ids=torch.randperm(E)[:K].to(torch.int32)
    t0=time.time()
    try:
        if form=="gather":
            m=CompileModule.from_module(Gather(K),(x,G,ids,Wgu,Wd)); m.to(DEV)
            call=lambda ids_: m(x.to(DEV),G.to(DEV),ids_.to(DEV),Wgu_d,Wd_d)
        else:
            ex=[w for e in ids.tolist() for w in (Wgu[e],Wd[e])]
            m=CompileModule.from_module(Slots(K),(x,G,*ex)); m.to(DEV)
            call=lambda ids_: m(x.to(DEV),G.to(DEV),*[w for e in ids_.tolist() for w in slots_d[e]])
    except Exception as e:
        emit(dict(form=form,T=T,K=K,error=f"{type(e).__name__}: {str(e)[:300]}")); continue
    comp=time.time()-t0
    ref=Slots(K)(x.float(),G.float(),*[w.float() for e in ids.tolist() for w in (Wgu[e],Wd[e])])
    got=call(ids).cpu().float(); rel=((got-ref).norm()/(ref.norm()+1e-9)).item()
    idsets=[torch.randperm(E)[:K].to(torch.int32) for _ in range(8)]
    for i in range(3): call(idsets[i%8])
    ts=[]
    for i in range(20):
        t=time.perf_counter(); call(idsets[i%8]).cpu(); ts.append(time.perf_counter()-t)
    med=statistics.median(ts)
    emit(dict(form=form,T=T,K=K,compile_s=round(comp,1),rel_err=round(rel,4),rt_ms=round(med*1e3,3),eff_GBps=round(K*3*H*F*2/med/1e9,1)))
