//! Host-side helpers: weight permutations into the device HBM layouts and a scalar f32 oracle.
use furiosa_opt_std::prelude::bf16;

pub const H: usize = 4096;
pub const Q: usize = 8192;
pub const P: usize = 1024;
pub const N: usize = 8;
pub const G: usize = 8;
pub const D: usize = 128;
pub const T: usize = 512;
pub const EPS: f32 = 1e-5;

/// Generic GEMV weight permutation: `w` is `[rows, cols]` row-major (HF Linear weight).
/// Output order: `[rows / rps, cols / 32, (rows % rps) / 8, (cols % 32) / 4, rows % 8, cols % 4]`.
pub fn perm_gemv(w: &[bf16], rows: usize, cols: usize, rps: usize) -> Vec<bf16> {
    assert_eq!(w.len(), rows * cols);
    let mut out = Vec::with_capacity(w.len());
    for rg in 0..rows / rps {
        for c32 in 0..cols / 32 {
            for r8 in 0..rps / 8 {
                for c4 in 0..8 {
                    for ri in 0..8 {
                        let row = &w[(rg * rps + r8 * 8 + ri) * cols + c32 * 32 + c4 * 4..][..4];
                        out.extend_from_slice(row);
                    }
                }
            }
        }
    }
    out
}

/// `q_proj.weight` [8192, 4096] -> `gemv::WQ`.
pub fn perm_wq(w: &[bf16]) -> Vec<bf16> {
    perm_gemv(w, Q, H, 512)
}
/// `k_proj.weight`, `v_proj.weight` (each [1024, 4096]) -> `gemv::WKV` (rows = [k; v]).
pub fn perm_wkv(wk: &[bf16], wv: &[bf16]) -> Vec<bf16> {
    let kv: Vec<bf16> = wk.iter().chain(wv).copied().collect();
    perm_gemv(&kv, 2 * P, H, 512)
}
/// `o_proj.weight` [4096, 8192] -> `gemv::WO`.
pub fn perm_wo(w: &[bf16]) -> Vec<bf16> {
    perm_gemv(w, H, Q, 512)
}

fn r(v: f32) -> f32 {
    f32::from(bf16::from_f32(v))
}
fn f(v: &[bf16]) -> Vec<f32> {
    v.iter().map(|&x| f32::from(x)).collect()
}

/// rmsnorm in f32, rounded to bf16 (as the device does: x * w / rms, one rounding).
pub fn rmsnorm(x: &[f32], w: &[bf16]) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = (ms + EPS).sqrt();
    x.iter().zip(w).map(|(&v, &w)| r(v * f32::from(w) / s)).collect()
}

pub fn gemv(w: &[bf16], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows).map(|i| w[i * cols..(i + 1) * cols].iter().zip(x).map(|(&a, &b)| f32::from(a) * b).sum()).collect()
}

/// rotate_half RoPE on each D-wide head.
pub fn rope(v: &[f32], cos: &[bf16], sin: &[bf16]) -> Vec<f32> {
    let mut out = vec![0f32; v.len()];
    for (hd, o) in v.chunks(D).zip(out.chunks_mut(D)) {
        for d in 0..D {
            let rot = if d < D / 2 { -hd[d + D / 2] } else { hd[d - D / 2] };
            o[d] = r(hd[d] * f32::from(cos[d]) + rot * f32::from(sin[d]));
        }
    }
    out
}

/// Oracle for `attn_in`: returns (q[N*G*D] after rope, k[N*D] after rope, v[N*D]).
pub fn attn_in_ref(
    x: &[bf16],
    w_in: &[bf16],
    wq: &[bf16],
    wk: &[bf16],
    wv: &[bf16],
    cos: &[bf16],
    sin: &[bf16],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let h = rmsnorm(&f(x), w_in);
    let q: Vec<f32> = gemv(wq, &h, Q, H).into_iter().map(r).collect();
    let k: Vec<f32> = gemv(wk, &h, P, H).into_iter().map(r).collect();
    let v: Vec<f32> = gemv(wv, &h, P, H).into_iter().map(r).collect();
    (rope(&q, cos, sin), rope(&k, cos, sin), v)
}

/// Oracle for attention over `len` tokens of a contiguous cache `[len, N, D]`; returns normalized `[N, G, D]`.
pub fn attention_ref(q: &[bf16], k: &[bf16], v: &[bf16], len: usize) -> Vec<f32> {
    let mut out = vec![0f32; N * G * D];
    let scale = 1.0 / (D as f32).sqrt();
    for head in 0..N * G {
        let n = head / G;
        let qh = &q[head * D..(head + 1) * D];
        let s: Vec<f32> = (0..len)
            .map(|t| qh.iter().zip(&k[(t * N + n) * D..][..D]).map(|(&a, &b)| f32::from(a) * f32::from(b)).sum::<f32>() * scale)
            .collect();
        let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
        let z: f32 = e.iter().sum();
        for t in 0..len {
            for d in 0..D {
                out[head * D + d] += e[t] / z * f32::from(v[(t * N + n) * D + d]);
            }
        }
    }
    out
}

/// Oracle for `attn_out`: input `a` is the normalized attention `[Q]`. Returns (x2, h2).
pub fn attn_out_ref(a: &[f32], x: &[bf16], wo: &[bf16], w_post: &[bf16]) -> (Vec<f32>, Vec<f32>) {
    let o = gemv(wo, a, H, Q);
    let x2: Vec<f32> = o.iter().zip(x).map(|(&o, &x)| r(r(o) + f32::from(x))).collect();
    let h2 = rmsnorm(&x2, w_post);
    (x2, h2)
}

/// The device RoPE does `rotate_half` as a plain half swap; the sign lives in `sin`.
/// Pass `sin_signed = cat(-sin[..D/2], sin[D/2..])` (bf16 [D]) as the kernel's `sin` input.
pub fn rope_sin_signed(sin: &[bf16]) -> Vec<bf16> {
    sin.iter().enumerate().map(|(d, &s)| if d < D / 2 { bf16::from_f32(-f32::from(s)) } else { s }).collect()
}

/// Chunks in the per-layer KV cache table (`kernel::Cm`).
pub const CM: usize = 16;

/// Element index of (kv head `n`, token `pos`, `d`) in a `kernel::KvCache` table `[N, CM * T, D]`.
pub fn cache_index(n: usize, pos: usize, d: usize) -> usize {
    (n * CM * T + pos) * D + d
}

/// `kv_offset` argument for writing token `pos`: byte offset inside each head's plane.
pub fn kv_offset(pos: usize) -> i32 {
    (pos * D * 2) as i32
}

/// `kernel::Masks` (`[CM, T]` f32) for a context of `ctx` tokens: 0 = valid, -1e30 = masked.
pub fn masks(ctx: usize) -> Vec<f32> {
    (0..CM * T).map(|i| if i < ctx { 0.0 } else { -1.0e30 }).collect()
}

/// Logical cache `[ctx, N, D]` -> `kernel::KvCache` table (zeros beyond `ctx`).
pub fn to_cache_table(x: &[bf16], ctx: usize) -> Vec<bf16> {
    let mut o = vec![bf16::from_f32(0.0); N * CM * T * D];
    for t in 0..ctx {
        for n in 0..N {
            o[cache_index(n, t, 0)..][..D].copy_from_slice(&x[(t * N + n) * D..][..D]);
        }
    }
    o
}
