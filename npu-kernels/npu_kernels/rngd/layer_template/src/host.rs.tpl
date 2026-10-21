//! Host side of the generated engine: geometry constants, bf16 helpers, weight permutations into the device HBM
//! layouts, and scalar oracles (HF semantics) for the eval checks. bf16 values are carried as raw `u16` bits.

pub const H: usize = ${H};
pub const NH: usize = ${N} * ${G};
pub const N: usize = ${N};
pub const G: usize = ${G};
pub const D: usize = ${D};
pub const Q: usize = ${Q};
pub const P: usize = ${P};
pub const T: usize = ${T};
pub const CM: usize = ${Cm};
pub const NE: usize = ${NE};
pub const TOPK: usize = ${TOPK};
pub const LAYERS: usize = ${LAYERS};
pub const VOCAB: usize = ${VOCAB};
pub const EPS: f32 = ${EPS};
pub const THETA: f32 = ${THETA};
pub const QK_NORM: bool = ${QK_NORM};
pub const NORM_TOPK: bool = ${NORM_TOPK};
pub const SCORING: &str = "${SCORING}";
/// attention launches compiled: contexts up to `T * c` for c in CHUNKS
pub const CHUNKS: &[usize] = &[${CHUNK_LIST}];
// GEMV tilings (rows per slice)
pub const RPS: usize = ${RPS};
pub const RPSO: usize = ${RPSo};
pub const RR: usize = ${RR};
// expert kernel
pub const F: usize = ${F};
pub const E_T: usize = ${E};
pub const K: usize = ${K};
pub const SLOT_BITS: u32 = ${SB};
pub const ER: usize = ${R};
pub const EHR: usize = ${HR};
pub const EFC: usize = ${FC};
// LM head
pub const LF: usize = ${LF};
pub const LNC: usize = ${LNC};
pub const LR: usize = ${LR};

#[inline]
pub fn f2bf(f: f32) -> u16 {
    if f.is_nan() {
        return 0x7fc0;
    }
    let u = f.to_bits();
    let u = u.wrapping_add(0x7FFF + ((u >> 16) & 1));
    (u >> 16) as u16
}
#[inline]
pub fn bf2f(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
#[inline]
pub fn r(v: f32) -> f32 {
    bf2f(f2bf(v))
}

/// Generic GEMV weight permutation: `w` is `[rows, cols]` row-major (HF Linear weight).
/// Output order: `[rows / rps, cols / cb, (rows % rps) / 8, (cols % cb) / 4, rows % 8, cols % 4]`.
pub fn perm_blocks(w: &[u16], rows: usize, cols: usize, rps: usize, cb: usize) -> Vec<u16> {
    assert_eq!(w.len(), rows * cols);
    let mut out = Vec::with_capacity(w.len());
    for rg in 0..rows / rps {
        for c in 0..cols / cb {
            for r8 in 0..rps / 8 {
                for c4 in 0..cb / 4 {
                    for ri in 0..8 {
                        out.extend_from_slice(&w[(rg * rps + r8 * 8 + ri) * cols + c * cb + c4 * 4..][..4]);
                    }
                }
            }
        }
    }
    out
}
pub fn perm_gemv(w: &[u16], rows: usize, cols: usize, rps: usize) -> Vec<u16> {
    perm_blocks(w, rows, cols, rps, 32)
}
/// `q_proj.weight` [Q, H] -> `attn::gemv::WQ`.
pub fn perm_wq(w: &[u16]) -> Vec<u16> {
    perm_gemv(w, Q, H, RPS)
}
/// `k_proj.weight`, `v_proj.weight` (each [P, H]) -> `attn::gemv::WKV` (rows = [k; v]).
pub fn perm_wkv(wk: &[u16], wv: &[u16]) -> Vec<u16> {
    let kv: Vec<u16> = wk.iter().chain(wv).copied().collect();
    perm_gemv(&kv, 2 * P, H, RPS)
}
/// `o_proj.weight` [H, Q] -> `attn::gemv::WO`.
pub fn perm_wo(w: &[u16]) -> Vec<u16> {
    perm_gemv(w, H, Q, RPSO)
}
/// `mlp.gate.weight` [NE, H] -> `attn::gemv::WR`.
pub fn perm_wr(w: &[u16]) -> Vec<u16> {
    perm_gemv(w, NE, H, RR)
}
/// `lm_head.weight` [VOCAB, H] -> `lm::WLM` (rows padded with zeros to LNC * LF).
pub fn perm_lm(w: &[u16]) -> Vec<u16> {
    let mut p = w.to_vec();
    p.resize(LNC * LF * H, 0);
    perm_gemv(&p, LNC * LF, H, LR)
}
/// expert `gate_proj` / `up_proj` [F, H] -> one slot of `ffn::WGT`.
pub fn perm_wg(w: &[u16]) -> Vec<u16> {
    perm_blocks(w, F, H, ER, 32)
}
/// expert `down_proj` [H, F] -> one slot of `ffn::WDT`.
pub fn perm_wd(w: &[u16]) -> Vec<u16> {
    perm_blocks(w, H, F, EHR, EFC)
}
/// `ffn_k` ids argument: slots in SLOT_BITS-bit fields, count in bits 56..63.
pub fn pack_slots(slots: &[usize]) -> u64 {
    assert!(slots.len() <= K);
    let mut p = (slots.len() as u64) << 56;
    for (k, &s) in slots.iter().enumerate() {
        assert!(s < E_T);
        p += (s as u64) << (SLOT_BITS as usize * k);
    }
    p
}

/// Element index of (kv head `n`, token `pos`, `d`) in an `attn::KvCache` table `[N, CM * T, D]`.
pub fn cache_index(n: usize, pos: usize, d: usize) -> usize {
    (n * CM * T + pos) * D + d
}
/// `kv_offset` argument for writing token `pos`: byte offset inside each head's plane.
pub fn kv_offset(pos: usize) -> i32 {
    (pos * D * 2) as i32
}
/// `attn::Masks` (`[CM, T]` f32) for a context of `ctx` tokens: 0 = valid, -1e30 = masked.
pub fn masks(ctx: usize) -> Vec<f32> {
    (0..CM * T).map(|i| if i < ctx { 0.0 } else { -1.0e30 }).collect()
}
/// smallest compiled attention launch that covers `ctx` tokens
pub fn chunks_for(ctx: usize) -> Option<usize> {
    CHUNKS.iter().copied().find(|&c| ctx <= c * T)
}

/// HF rotary table for `pos` (default rope): `(cos, sin)` bf16 `[D]`, `cat(freqs, freqs)`.
pub fn cos_sin(pos: usize) -> (Vec<u16>, Vec<u16>) {
    let mut c = vec![0u16; D];
    let mut s = vec![0u16; D];
    for i in 0..D / 2 {
        let inv = 1.0f32 / THETA.powf((2 * i) as f32 / D as f32);
        let a = pos as f32 * inv;
        c[i] = f2bf(a.cos());
        c[i + D / 2] = c[i];
        s[i] = f2bf(a.sin());
        s[i + D / 2] = s[i];
    }
    (c, s)
}
/// The device RoPE does `rotate_half` as a plain half swap; the sign lives in `sin`.
pub fn rope_sin_signed(sin: &[u16]) -> Vec<u16> {
    sin.iter().enumerate().map(|(d, &s)| if d < D / 2 { f2bf(-bf2f(s)) } else { s }).collect()
}

// ---------------- oracles (HF semantics, f32 math, bf16 rounding where HF rounds) ----------------

/// HF RMSNorm: fp32 normalize, cast to bf16, multiply by the bf16 weight -> bf16.
pub fn rmsnorm(x: &[f32], w: &[u16]) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + EPS).sqrt();
    x.iter().zip(w).map(|(&v, &wb)| r(r(v * s) * bf2f(wb))).collect()
}
pub fn gemv(w: &[u16], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows).map(|i| w[i * cols..(i + 1) * cols].iter().zip(x).map(|(&a, &b)| bf2f(a) * b).sum()).collect()
}
/// rotate_half RoPE on each D-wide head (HF: bf16 products, bf16 sum).
pub fn rope(v: &[f32], cos: &[u16], sin: &[u16]) -> Vec<f32> {
    let mut out = vec![0f32; v.len()];
    for (hd, o) in v.chunks(D).zip(out.chunks_mut(D)) {
        for d in 0..D {
            let rot = if d < D / 2 { -hd[d + D / 2] } else { hd[d - D / 2] };
            o[d] = r(r(hd[d] * bf2f(cos[d])) + r(rot * bf2f(sin[d])));
        }
    }
    out
}
/// per-head RMSNorm over D
pub fn head_norm(v: &[f32], w: &[u16]) -> Vec<f32> {
    v.chunks(D).flat_map(|h| rmsnorm(h, w)).collect()
}
/// attention of q `[NH, D]` over `len` tokens of a logical cache `[len, N, D]`; returns `[NH, D]` (bf16-rounded).
pub fn attention_ref(q: &[f32], k: &[u16], v: &[u16], len: usize) -> Vec<f32> {
    let mut out = vec![0f32; NH * D];
    let scale = 1.0 / (D as f32).sqrt();
    for head in 0..NH {
        let n = head / G;
        let qh = &q[head * D..(head + 1) * D];
        let s: Vec<f32> = (0..len)
            .map(|t| qh.iter().zip(&k[(t * N + n) * D..][..D]).map(|(&a, &b)| a * bf2f(b)).sum::<f32>() * scale)
            .collect();
        let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
        let z: f32 = e.iter().sum();
        for t in 0..len {
            for d in 0..D {
                out[head * D + d] += e[t] / z * bf2f(v[(t * N + n) * D + d]);
            }
        }
    }
    out.iter().map(|&x| r(x)).collect()
}

/// layer weights of the attention block + router (HF layouts, bf16 bits)
pub struct BlockW<'a> {
    pub w_in: &'a [u16],
    pub wq: &'a [u16],
    pub wk: &'a [u16],
    pub wv: &'a [u16],
    pub qn: &'a [u16],
    pub kn: &'a [u16],
    pub wo: &'a [u16],
    pub w_post: &'a [u16],
    pub wr: &'a [u16],
}

/// q/k/v for token `pos` of a layer: (q `[NH, D]` after norm+RoPE, k `[N, D]` after norm+RoPE, v `[N, D]`).
pub fn qkv_ref(w: &BlockW, x: &[f32], cos: &[u16], sin: &[u16]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let h = rmsnorm(x, w.w_in);
    let q: Vec<f32> = gemv(w.wq, &h, Q, H).into_iter().map(r).collect();
    let k: Vec<f32> = gemv(w.wk, &h, P, H).into_iter().map(r).collect();
    let v: Vec<f32> = gemv(w.wv, &h, P, H).into_iter().map(r).collect();
    let (q, k) = if QK_NORM { (head_norm(&q, w.qn), head_norm(&k, w.kn)) } else { (q, k) };
    (rope(&q, cos, sin), rope(&k, cos, sin), v)
}

/// attention output `a` `[Q]` -> (x2, h2, router logits)
pub fn block_out_ref(w: &BlockW, x: &[f32], a: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let o: Vec<f32> = gemv(w.wo, a, H, Q).into_iter().map(r).collect();
    let x2: Vec<f32> = o.iter().zip(x).map(|(&o, &x)| r(o + x)).collect();
    let h2 = rmsnorm(&x2, w.w_post);
    let lg: Vec<f32> = gemv(w.wr, &h2, NE, H).into_iter().map(r).collect();
    (x2, h2, lg)
}

/// one expert (HF): down(bf16(silu(bf16(gate x)) * bf16(up x)))
pub fn expert_ref(g: &[u16], u: &[u16], d: &[u16], x: &[f32]) -> Vec<f32> {
    let gt: Vec<f32> = gemv(g, x, F, H).into_iter().map(r).collect();
    let up: Vec<f32> = gemv(u, x, F, H).into_iter().map(r).collect();
    let a: Vec<f32> = gt.iter().zip(&up).map(|(&g, &u)| r(r(g / (1.0 + (-g).exp())) * u)).collect();
    gemv(d, &a, H, F).into_iter().map(r).collect()
}

/// router: (expert ids, weights) for one token's logits (bf16 values as f32).
pub fn route(logits: &[f32], bias: Option<&[f32]>) -> (Vec<usize>, Vec<f32>) {
    let s: Vec<f32> = if SCORING == "softmax" {
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
        let z: f32 = e.iter().sum();
        e.iter().map(|v| v / z).collect()
    } else {
        logits.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
    };
    let key = |i: usize| s[i] + bias.map(|b| b[i]).unwrap_or(0.0);
    let mut idx: Vec<usize> = (0..NE).collect();
    // stable: ties keep the lower index first (torch.topk on CPU picks the lower index for equal values)
    idx.sort_by(|&a, &b| key(b).partial_cmp(&key(a)).unwrap().then(a.cmp(&b)));
    idx.truncate(TOPK);
    let w: Vec<f32> = idx.iter().map(|&i| s[i]).collect();
    if NORM_TOPK {
        let tot: f32 = w.iter().sum::<f32>() + if SCORING == "softmax" { 0.0 } else { 1e-20 };
        (idx, w.iter().map(|v| v / tot).collect())
    } else {
        (idx, w)
    }
}
