import torch, sys
sys.path.insert(0,'.')
from model import *
from runtime import Npu, bucket
W=Weights(); c=W.cfg; d=W.layer_dense(0)
npu=Npu(c,"rngd:0"); P=npu.put
rel=lambda a,b: ((a.float()-b.float()).norm()/b.float().norm()).item()
torch.manual_seed(0)
for Tb in (1,8):
    x=(W.get("model.embed_tokens.weight")[torch.randint(0,1000,(Tb,))]).contiguous()
    ref=rmsnorm(x,d["in_norm"],c.eps)@d["wqkv"].t()
    got=npu.graph("qkv",Tb)(P(x),P(d["in_norm"]),P(d["wqkv"])).cpu()
    print("qkv",Tb,rel(got,ref))
    a=(torch.randn(Tb,c.nh*c.hd)*0.05).to(BF)
    x2=x+a@d["wo"].t(); h2=rmsnorm(x2,d["post_norm"],c.eps); y=x2+swiglu_ffn(h2,d["sh_gu"],d["sh_d"])
    gy,gh=npu.graph("post",Tb)(P(x),P(a),P(d["wo"]),P(d["post_norm"]),P(d["sh_gu"]),P(d["sh_d"]))
    print("post y",rel(gy.cpu(),y),"h2",rel(gh.cpu(),h2))
    ex=[W.expert(0,e) for e in range(8)]; G=torch.rand(Tb,8).to(BF)
    ref=sum(swiglu_ffn(h2,*ex[i]).float()*G[:,i:i+1].float() for i in range(8))
    got=npu.graph("exp",Tb,8)(P(h2),P(G),*[P(w) for e in ex for w in e]).cpu()
    print("exp",rel(got,ref))
