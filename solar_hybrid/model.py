"""Solar-Open-100B weights + host-side reference ops (HF-faithful, bf16).

Weights are read lazily from the HF safetensors shards. Host math mirrors
transformers.models.solar_open (RMSNorm in fp32, rotary from SolarOpenRotaryEmbedding,
sigmoid router + e_score_correction_bias + group top-k, norm_topk_prob).
"""
import glob, json, os
import torch
from safetensors import safe_open

MODEL = "upstage/Solar-Open-100B"
BF = torch.bfloat16


def snapshot_dir(model=MODEL):
    base = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots")
    return sorted(glob.glob(os.path.join(base, "*")))[-1]


class Config:
    def __init__(self, d):
        self.raw = d
        self.H = d["hidden_size"]; self.L = d["num_hidden_layers"]
        self.nh = d["num_attention_heads"]; self.nkv = d["num_key_value_heads"]; self.hd = d["head_dim"]
        self.E = d["n_routed_experts"]; self.topk = d["num_experts_per_tok"]; self.F = d["moe_intermediate_size"]
        self.n_group = d.get("n_group", 1); self.topk_group = d.get("topk_group", 1)
        self.norm_topk = d["norm_topk_prob"]; self.scale = d["routed_scaling_factor"]
        self.eps = d["rms_norm_eps"]; self.vocab = d["vocab_size"]


class Weights:
    """Lazy tensor access by HF name."""
    def __init__(self, model=MODEL):
        self.dir = snapshot_dir(model)
        self.map = json.load(open(os.path.join(self.dir, "model.safetensors.index.json")))["weight_map"]
        self.cfg = Config(json.load(open(os.path.join(self.dir, "config.json"))))
        self._h = {}

    def get(self, name):
        f = self.map[name]
        if f not in self._h:
            self._h[f] = safe_open(os.path.join(self.dir, f), framework="pt")
        return self._h[f].get_tensor(name)

    def layer_dense(self, l):
        p = f"model.layers.{l}."
        g = lambda n: self.get(p + n)
        return dict(
            in_norm=g("input_layernorm.weight"), post_norm=g("post_attention_layernorm.weight"),
            wqkv=torch.cat([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"),
                            g("self_attn.v_proj.weight")], 0).contiguous(),
            wo=g("self_attn.o_proj.weight"),
            sh_gu=torch.cat([g("mlp.shared_experts.gate_proj.weight"),
                             g("mlp.shared_experts.up_proj.weight")], 0).contiguous(),
            sh_d=g("mlp.shared_experts.down_proj.weight"),
            router=g("mlp.gate.weight").float(), bias=g("mlp.gate.e_score_correction_bias").float())

    def expert(self, l, e):
        p = f"model.layers.{l}.mlp.experts.{e}."
        gu = torch.cat([self.get(p + "gate_proj.weight"), self.get(p + "up_proj.weight")], 0).contiguous()
        return gu, self.get(p + "down_proj.weight").contiguous()


# ---------------- host reference ops ----------------
def rmsnorm(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def swiglu_ffn(x, gu, d):
    F = d.shape[1]
    y = x @ gu.t()
    return (torch.nn.functional.silu(y[..., :F]) * y[..., F:]) @ d.t()


def route(cfg, h, router_w, bias):
    """h [T,H] bf16 -> (idx [T,k] long, w [T,k] fp32). Mirrors SolarOpenMoE.route_tokens_to_experts."""
    logits = torch.nn.functional.linear(h.float(), router_w)
    s = logits.sigmoid()
    sc = s + bias
    T = h.shape[0]
    if cfg.n_group > 1:
        gs = sc.view(T, cfg.n_group, -1).topk(2, dim=-1)[0].sum(-1)
        gi = torch.topk(gs, k=cfg.topk_group, dim=-1, sorted=False)[1]
        gm = torch.zeros_like(gs).scatter_(1, gi, 1)
        sm = gm.unsqueeze(-1).expand(-1, cfg.n_group, cfg.E // cfg.n_group).reshape(T, -1)
        sc = sc.masked_fill(~sm.bool(), 0.0)
    idx = torch.topk(sc, k=cfg.topk, dim=-1, sorted=False)[1]
    w = s.gather(1, idx)
    if cfg.norm_topk:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return idx, w * cfg.scale


class Rotary:
    def __init__(self, cfg_raw):
        from transformers import AutoConfig
        from transformers.models.solar_open.modeling_solar_open import SolarOpenRotaryEmbedding
        c = AutoConfig.from_pretrained(snapshot_dir())
        self.rot = SolarOpenRotaryEmbedding(c)

    def __call__(self, pos):  # pos LongTensor [T] -> cos, sin [T, hd] (bf16)
        dummy = torch.zeros(1, dtype=BF)
        cos, sin = self.rot(dummy, pos[None])
        return cos[0].to(BF), sin[0].to(BF)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), -1)


def attention(cfg, q, k_cache, v_cache, n_past, T):
    """q [T,nh,hd] (roped), k/v_cache [S,nkv,hd] valid up to n_past+T. causal. returns [T, nh*hd] bf16."""
    S = n_past + T
    k = k_cache[:S].transpose(0, 1)  # [nkv,S,hd]
    v = v_cache[:S].transpose(0, 1)
    rep = cfg.nh // cfg.nkv
    k = k.repeat_interleave(rep, 0); v = v.repeat_interleave(rep, 0)
    qh = q.transpose(0, 1)  # [nh,T,hd]
    sc = (qh @ k.transpose(1, 2)) * (cfg.hd ** -0.5)  # [nh,T,S]
    qpos = torch.arange(n_past, S)[:, None]; kpos = torch.arange(S)[None]
    sc = sc.masked_fill(kpos > qpos, float("-inf"))
    p = torch.softmax(sc, dim=-1, dtype=torch.float32).to(q.dtype)
    return (p @ v).transpose(0, 1).reshape(T, -1)
