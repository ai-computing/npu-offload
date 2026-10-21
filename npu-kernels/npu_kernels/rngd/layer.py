#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Decoder-layer engine for RNGD generated from a model geometry (furiosa-opt 0.9, one chip = 2 clusters x 256 slices).

`LayerGeometry` holds the shape of one MoE decoder layer (what a ModelSpec / HF config says); `plan_layer` derives every
tiling constant of the attention block, the router and the LM head; `generate_layer` writes a crate (template
`layer_template/`) with

  attn_block_c{K}   rmsnorm -> q/k/v GEMV -> [per-head q/k RMSNorm] -> RoPE -> KV-cache write -> attention over K
                    chunks of 512 tokens -> o_proj -> residual -> post-norm -> router logits, one launch per layer
  ffn_k             routed experts from resident tables (the expert kernel of `generate.plan`)
  lm_head           weight-streaming GEMV over the vocabulary
  bin/eval          CPU-emulator / silicon checks of each kernel against a host oracle
  bin/engine        decode engine (host: embedding, top-k, cold experts on the CPU, sampling)

Tiling rules (the ones the hand-written Solar-Open engine uses, written as functions of the geometry):

  q / [k; v] GEMV   rows in chunks of RW = 2 * kv_heads * head_dim; the 2 clusters split a chunk, the 256 slices of
                    a cluster are (A = 8192 / H row groups) x (H / 32 column blocks), RPS = RW / (2A) rows per slice
  RoPE / q-k norm   one kv head per 256 / N slices (k, v); one q head per 256 / (N * G) slices
  attention         N / 2 kv heads per cluster, 16 tokens per slice, 32 token-slices per head
  o_proj            rows in chunks of Hr = 1024; the 256 slices are (Ao = 8192 / Q row groups) x (Q / 32 column
                    blocks), RPSo = Hr / (2 Ao) rows per slice
  epilogue          H split in halves over the clusters
  router            one chunk of E rows: (A row groups) x (H / 32 column blocks), RR = E / (2A) rows per slice
"""
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from string import Template

from .generate import plan as plan_expert

_TPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layer_template")


@dataclass(frozen=True)
class LayerGeometry:
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    qk_norm: bool                 # per-head RMSNorm of q and k before RoPE (Qwen3)
    rms_eps: float
    rope_theta: float
    experts: int
    top_k: int
    moe_ffn: int                  # routed expert intermediate size
    shared_ffn: int               # 0 = no shared expert
    vocab: int
    layers: int
    scoring: str = "softmax"      # router scoring: "softmax" (Qwen3) | "sigmoid" (Solar / DeepSeek-V3 style)
    norm_topk: bool = True

    @property
    def name(self):
        return f"h{self.hidden}_q{self.heads}_kv{self.kv_heads}_d{self.head_dim}_e{self.experts}_f{self.moe_ffn}"


def geometry_from_hf(cfg: dict) -> LayerGeometry:
    """LayerGeometry from a HF config.json dict (Qwen3-MoE, Solar-Open)."""
    mt = cfg.get("model_type", "")
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    return LayerGeometry(
        hidden=cfg["hidden_size"], heads=cfg["num_attention_heads"], kv_heads=cfg["num_key_value_heads"], head_dim=hd,
        qk_norm=mt.startswith("qwen3"), rms_eps=cfg["rms_norm_eps"], rope_theta=float(cfg.get("rope_theta", 10000.0)),
        experts=cfg.get("num_experts") or cfg.get("n_routed_experts"), top_k=cfg["num_experts_per_tok"],
        moe_ffn=cfg["moe_intermediate_size"], shared_ffn=cfg.get("n_shared_experts", 0) * cfg["moe_intermediate_size"]
        if cfg.get("n_shared_experts") else cfg.get("shared_expert_intermediate_size", 0) or 0,
        vocab=cfg["vocab_size"], layers=cfg["num_hidden_layers"],
        scoring="softmax" if mt.startswith("qwen") else "sigmoid", norm_topk=bool(cfg.get("norm_topk_prob", True)))


QWEN3_30B_A3B = LayerGeometry(2048, 32, 4, 128, True, 1e-6, 1e7, 128, 8, 768, 0, 151936, 48, "softmax", True)
SOLAR_OPEN_100B = LayerGeometry(4096, 64, 8, 128, False, 1e-5, 1e6, 128, 8, 1280, 1280, 196608, 48, "sigmoid", True)


@dataclass(frozen=True)
class LayerPlan:
    g: LayerGeometry
    # attention
    Q: int
    P: int
    N: int
    G: int
    D: int
    T: int
    Cm: int
    RW: int
    Qc: int
    A: int
    RPS: int
    SPN: int
    SPH: int
    NPC: int
    Hr: int
    Hc: int
    Ao: int
    RPSo: int
    # router
    RR: int
    # LM head
    LF: int
    LNC: int
    LR: int
    # attention launches compiled (chunks of T tokens)
    chunks: tuple = (1, 2, 4)
    # expert kernel (resident table of E_T experts, K per launch)
    E_T: int = 64
    K: int = 8

    def expert_plan(self):
        return plan_expert(self.g.hidden, self.g.moe_ffn, self.E_T, self.K)

    @property
    def slot_bits(self):
        return 6 if self.E_T <= 64 else 7

    def summary(self):
        d = asdict(self)
        d["g"] = asdict(self.g)
        return d


def _lm_plan(vocab, H, chunk_bytes=16 << 20):
    A = 8192 // H
    unit = 2 * A * 8
    f_max = max(unit, (chunk_bytes // (2 * H)) // unit * unit)
    rows_u = -(-vocab // unit) * unit
    nc = -(-rows_u // f_max)
    F = -(-rows_u // nc // unit) * unit
    return F, nc, F // (2 * A)


def plan_layer(g: LayerGeometry, chunks=(1, 2, 4), E_T: int = 64, K: int = 8, T: int = 512, Cm: int = 16) -> LayerPlan:
    H, D, N = g.hidden, g.head_dim, g.kv_heads
    if g.heads % N:
        raise ValueError("heads must be a multiple of kv_heads")
    G = g.heads // N
    Q, P = g.heads * D, N * D
    if D != 128:
        raise ValueError(f"head_dim={D}: the attention template assumes 128 (RoPE halves, 16-token slices)")
    if H % 32 or 8192 % H:
        raise ValueError(f"hidden={H}: needs hidden | 8192 and hidden % 32 == 0")
    if N not in (4, 8, 16):
        raise ValueError(f"kv_heads={N}: the attention layout needs 4, 8 or 16 kv heads (N / 2 per cluster)")
    if 256 % g.heads:
        raise ValueError(f"heads={g.heads}: per-head q layout needs heads | 256")
    if Q % 32 or 8192 % Q:
        raise ValueError(f"q width={Q}: o_proj tiling needs Q | 8192")
    A = 8192 // H
    RW = 2 * P
    if Q % RW:
        raise ValueError(f"q width {Q} is not a multiple of the k/v chunk {RW}")
    RPS = RW // (2 * A)
    if RPS % 8 or RPS < 8:
        raise ValueError(f"q/k/v GEMV: {RPS} rows per slice (must be a multiple of 8)")
    Hr = 1024
    if H % Hr:
        raise ValueError(f"hidden={H} not a multiple of the o_proj chunk {Hr}")
    Ao = 8192 // Q
    RPSo = Hr // (2 * Ao)
    if g.experts % (2 * A) or (g.experts // (2 * A)) % 8:
        raise ValueError(f"router: {g.experts} experts do not give 8-aligned rows per slice")
    RR = g.experts // (2 * A)
    LF, LNC, LR = _lm_plan(g.vocab, H)
    if E_T > 128 or (E_T > 64 and K > 8):
        raise ValueError("expert table: at most 128 slots (7-bit ids x 8)")
    p = LayerPlan(g, Q, P, N, G, D, T, Cm, RW, Q // RW, A, RPS, 256 // N, 256 // g.heads, N // 2, Hr, H // Hr, Ao, RPSo,
                  RR, LF, LNC, LR, tuple(chunks), E_T, K)
    p.expert_plan()          # validates the expert tiling
    return p


def crate_name(p: LayerPlan) -> str:
    return f"rngd_layer_{p.g.name}_et{p.E_T}"


def _f32(v: float) -> str:
    s = repr(float(v))
    return s + "f32" if "e" in s or "." in s else s + ".0f32"


def _subs(p: LayerPlan) -> dict:
    g = p.g
    ep = p.expert_plan()
    s = dict(
        H=p.g.hidden, Q=p.Q, P=p.P, N=p.N, G=p.G, D=p.D, DH=p.D // 2, T=p.T, Cm=p.Cm, RW=p.RW, Qc=p.Qc, A=p.A,
        RPS=p.RPS, RPS8=p.RPS // 8, RCD=p.RW // 2, NH32=g.hidden // 32, SPN=p.SPN, SPH=p.SPH, NPC=p.NPC,
        SAP=8 // p.NPC, SNP=256 // p.NPC, GD=p.G * p.D, QH=p.Q // 2, HH=g.hidden // 2, Hr=p.Hr, HrH=p.Hr // 2, Hc=p.Hc,
        Ao=p.Ao, RPSo=p.RPSo, RPSo8=p.RPSo // 8, NQ32=p.Q // 32,
        NE=g.experts, NEH=g.experts // 2, RR=p.RR, RR8=p.RR // 8,
        LF=p.LF, LFH=p.LF // 2, LNC=p.LNC, LR=p.LR, LR8=p.LR // 8, VOCAB=g.vocab,
        HF=_f32(g.hidden), DF=_f32(p.D), EPS=_f32(g.rms_eps), SQRT_D=_f32(p.D ** 0.5), THETA=_f32(g.rope_theta),
        LAYERS=g.layers, TOPK=g.top_k, QK_NORM="true" if g.qk_norm else "false",
        SCORING=g.scoring, NORM_TOPK="true" if g.norm_topk else "false",
        # expert kernel
        F=ep.F, E=ep.E, K=ep.K, EA=ep.A, R=ep.R, HR=ep.HR, C=ep.C, FC=ep.FC, FH=ep.F // 2, R8=ep.R // 8, R4=ep.R // 4,
        NFC=ep.F // ep.FC, HR8=ep.HR // 8, FC4=ep.FC // 4, FC16=ep.FC // 16, SB=p.slot_bits, SBM=1 << p.slot_bits,
        CHUNK_LIST=", ".join(str(c) for c in p.chunks), MAXC=max(p.chunks),
    )
    s["OS"] = f"m![Hr / {p.RPSo} % {p.Ao}, Q / 32]" if p.Ao > 1 else "m![Q / 32]"
    s["OSO"] = f"m![Hr / {p.RPSo} % {p.Ao}, 1 # {p.Q // 32}]" if p.Ao > 1 else "m![1 # 256]"
    return {k: str(v) for k, v in s.items()}


def _block_kernel(p: LayerPlan, k: int) -> str:
    qk = p.g.qk_norm
    norm_args = ("\n    q_norm: &HbmTensor<bf16, Chip, m![D]>,\n    k_norm: &HbmTensor<bf16, Chip, m![D]>," if qk else "")
    norm_pass = "q_norm, k_norm, " if qk else ""
    nexts = "".join(f"""
    let mask = attention::mask_vrf(device, &tbl, {c});
    let (m2, s2, o2) = attention::next_dm(device, &q, kv_chunk(k_cache, {c}), kv_chunk(v_cache, {c}), &mask, &max, &sum, &out);
    max = m2;
    sum = s2;
    out = o2;""" for c in range(1, k))
    return f"""
/// Whole layer front for contexts `<= {p.T * k}` tokens: the current token is written at `kv_offset` and attended to in
/// the same launch; chunks `0..{k}` are attended with `masks` (fully masked chunks contribute 0).
#[device(chip = 1, pe = 8)]
#[expect(clippy::too_many_arguments)]
pub fn attn_block_c{k}(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    w_in: &HbmTensor<bf16, Chip, m![H]>,
    wq: &HbmTensor<bf16, Chip, WQ>,
    wkv: &HbmTensor<bf16, Chip, WKV>,{norm_args}
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![D]>,
    sin: &HbmTensor<bf16, Chip, m![D]>,
    k_cache: &mut HbmTensor<bf16, Chip, KvCache>,
    v_cache: &mut HbmTensor<bf16, Chip, KvCache>,
    masks: &HbmTensor<f32, Chip, Masks>,
    wo: &HbmTensor<bf16, Chip, WO>,
    w_post: &HbmTensor<bf16, Chip, m![H]>,
    wr: &HbmTensor<bf16, Chip, WR>,
    x2_out: &mut HbmTensor<bf16, Chip, m![H]>,
    h2_out: &mut HbmTensor<bf16, Chip, m![H]>,
    logits_out: &mut HbmTensor<bf16, Chip, m![NE]>,
) {{
    let q = qkv_rope(device, x, w_in, wq, wkv, {norm_pass}kv_offset, cos, sin, k_cache, v_cache);
    let q = attention::q_trf_from_hbm(device, &q);
    let tbl = attention::mask_table(device, masks);
    let mask = attention::mask_vrf(device, &tbl, 0);
    #[allow(unused_mut)]
    let (mut max, mut sum, mut out) = attention::first_dm(device, &q, kv_chunk(k_cache, 0), kv_chunk(v_cache, 0), &mask);{nexts}
    let _ = &max;
    let a = attention::norm_dm(device, &out, &sum);
    let (x2, h2) = o_proj_epilogue(device, a, x, wo, w_post);
    let lg = router_dm(device, &h2, wr);
    x2.view().to_hbm_view(&mut device.tdma, x2_out.view_mut());
    h2.view().to_hbm_view(&mut device.tdma, h2_out.view_mut());
    lg.view().to_hbm_view(&mut device.tdma, logits_out.view_mut());
}}
"""


def _slot_block(p: LayerPlan, k: int) -> str:
    ep = p.expert_plan()
    sb = p.slot_bits
    wg = f"m![1 # {ep.E}, F / {ep.R}, H / 32, F / 8 % {ep.R // 8}, H / 4 % 8, F % 8, H % 4]"
    wd = f"m![1 # {ep.E}, H / {ep.HR}, F / {ep.FC}, H / 8 % {ep.HR // 8}, F / 4 % {ep.FC // 4}, H % 8, F % 4]"
    return f"""    if {k}usize < active {{
        let e = ((packed / {1 << (sb * k)}u64) % {1 << sb}u64) as usize;
        let wg = wg_t.view().tile::<m![E], 1, {wg}>(e);
        let wu = wu_t.view().tile::<m![E], 1, {wg}>(e);
        let wd = wd_t.view().tile::<m![E], 1, {wd}>(e);
        let o = out.view_mut().tile::<m![K], 1, m![1 #{{!}} {ep.K}, H]>({k});
        expert(device, &x_trf, wg, wu, wd, o);
    }}"""


def _lm_chunk(p: LayerPlan, c: int) -> str:
    wt = f"m![1 # {p.LNC}, LF / {p.LR}, H / 32, LF / 8 % {p.LR // 8}, H / 4 % 8, LF % 8, H % 4]"
    return f"""    {{
        let w: DmTensor<bf16, Chip, LGC, LGS, LGElem> = w.view().tile::<m![LNC], 1, {wt}>({c}).to_dm(&mut device.tdma);
        let y = gemv_lm(device, &w, &x_trf);
        y.view().to_hbm_view(&mut device.tdma, out.view_mut().tile::<m![LNC], 1, m![1 #{{!}} {p.LNC}, LF]>({c}));
    }}"""


def generate_layer(p: LayerPlan, out_dir: str, crate: str = None) -> str:
    """write the crate for plan `p` into out_dir (overwritten)."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    subs = _subs(p)
    subs["CRATE"] = crate or crate_name(p)
    subs["BLOCKS"] = "\n".join(_block_kernel(p, k) for k in p.chunks)
    subs["SLOTS"] = "\n".join(_slot_block(p, k) for k in range(p.K))
    subs["LM_CHUNKS"] = "\n".join(_lm_chunk(p, c) for c in range(p.LNC))
    qk = p.g.qk_norm
    subs["QK_PARAMS"] = ("\n    q_norm: &HbmTensor<bf16, Chip, m![D]>,\n    k_norm: &HbmTensor<bf16, Chip, m![D]>,"
                         if qk else "")
    subs["QK_FRONT"] = _QK_FRONT if qk else _PLAIN_FRONT
    subs["ATTN_IMPORTS"] = "use crate::attn::{" + ", ".join(f"attn_block_c{k}" for k in p.chunks) + "};"
    norm = "&l.qn, &l.kn, " if qk else ""
    subs["BLOCK_ARMS"] = "\n".join(
        f"        {k} => launch(attn_block_c{k}, (&mut *dev, &s.x, &l.w_in, &l.wq, &l.wkv, {norm}&s.off, &s.cos, &s.sin, "
        f"&mut l.kc, &mut l.vc, &s.masks, &l.wo, &l.w_post, &l.wr, &mut s.x2, &mut s.h2, &mut s.lg)).await?,"
        for k in p.chunks)
    for root, _, files in os.walk(_TPL):
        for f in files:
            src = os.path.join(root, f)
            rel = os.path.relpath(src, _TPL)
            dst = os.path.join(out_dir, rel[:-4] if rel.endswith(".tpl") else rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            text = open(src).read()
            if src.endswith(".tpl"):
                # QK_FRONT itself holds ${...} names: substitute twice
                text = Template(Template(text).safe_substitute(subs)).substitute(subs)
            open(dst, "w").write(text)
    with open(os.path.join(out_dir, "plan.json"), "w") as f:
        json.dump(p.summary(), f, indent=1)
    return out_dir


# q path after the GEMV, written into qkv_rope: per-head RMSNorm of q and k (Qwen3), then RoPE without the deferred
# input-norm factor (the head norm is scale invariant); v still gets 1 / rms.
_QK_FRONT = """    let qs: DmTensor<bf16, Chip, C1, SliceQ, m![D]> = q3.to_dm(&mut device.tdma);
    let qs = rmsnorm::head_norm(device, &qs, q_norm);
    let qs = rope::rope_vec(device, &qs, cos, sin);
    let q_out: HbmTensor<bf16, Chip, m![N, G, D]> = qs.to_hbm(&mut device.tdma);
    let k = rmsnorm::head_norm(device, &k, k_norm);
    let k = rope::rope_vec(device, &k, cos, sin);
    store_kv(device, kv_offset, k, k_cache);
    q_out"""

# q path without q/k norm (Solar-Open): RoPE divides by the deferred input-norm factor.
_PLAIN_FRONT = """    let qs: DmTensor<bf16, Chip, C1, SliceQ, m![D]> = q3.to_dm(&mut device.tdma);
    let xq: DmTensor<bf16, Chip, C1, SliceQ, m![H]> = x.to_dm(&mut device.tdma);
    let rq = rmsnorm::rms_s1(device, &xq);
    let rq_vrf: VrfTensor<f32, Chip, C1, SliceQ, m![1 # 8]> =
        device.sub.begin(rq.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let qs = rope::rope_vec_div(device, &qs, cos, sin, &rq_vrf);
    let q_out: HbmTensor<bf16, Chip, m![N, G, D]> = qs.to_hbm(&mut device.tdma);
    let k = rope::rope_vec_div(device, &k, cos, sin, &rms_vrf);
    store_kv(device, kv_offset, k, k_cache);
    q_out"""
