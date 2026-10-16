"""Solar dense-part per layer on NPU (fusion=8): qkv, o, shared expert, router matmuls at T tokens."""
import sys, time, statistics, json, torch
import furiosa.torch
from furiosa.torch import native_device as nd
nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))
from furiosa.torch.custom_ops import CompileModule
BF=torch.bfloat16; H=4096; QKV=8192+2*1024; F=1280; E=128; DEV="rngd:0"
class Dense(torch.nn.Module):
    def forward(self, x, a, wqkv, wo, wgu, wd, wr):
        qkv = x @ wqkv.t()
        o = a @ wo.t()                      # attention output stand-in
        h = x + o
        gu = h @ wgu.t(); s = (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ wd.t()
        r = torch.sigmoid((h @ wr.t()).float())
        return qkv, h + s, r
for T in [int(t) for t in sys.argv[1:]] or [1]:
    ws=[(torch.randn(QKV,H)*.02).to(BF),(torch.randn(H,8192)*.02).to(BF),(torch.randn(2*F,H)*.02).to(BF),(torch.randn(H,F)*.02).to(BF),(torch.randn(E,H)*.02).to(BF)]
    x=torch.randn(T,H).to(BF); at=torch.randn(T,8192).to(BF)
    t0=time.time(); m=CompileModule.from_module(Dense(),(x,at,*ws)); m.to(DEV); comp=time.time()-t0
    wd_=[w.to(DEV) for w in ws]; xd=x.to(DEV); ad=at.to(DEV)
    for _ in range(3): m(xd,ad,*wd_)
    ts=[]
    for _ in range(20):
        t=time.perf_counter(); m(xd,ad,*wd_); ts.append(time.perf_counter()-t)
    med=statistics.median(ts); wb=sum(w.numel() for w in ws)*2
    print(json.dumps(dict(sec="dense_layer",T=T,compile_s=comp,ms=med*1e3,eff_GBps=wb/med/1e9,MB=wb/1e6)),flush=True)
