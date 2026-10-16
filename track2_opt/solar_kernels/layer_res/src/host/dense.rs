//! Host-side helpers: weight permutations into the kernels' HBM layouts, oracles, checks.
//! (included via #[path] by the eval binary)
#![allow(dead_code)]
use furiosa_opt_std::prelude::bf16;
use rand::Rng;

pub const F: usize = 1280;
pub const H: usize = 4096;
pub const R: usize = 128;
pub const K: usize = 8;
pub const W: usize = 196608;

pub fn rand_bf16(rng: &mut impl Rng, n: usize, scale: f32) -> Vec<bf16> {
    (0..n).map(|_| bf16::from_f32((rng.random::<f32>() * 2.0 - 1.0) * scale)).collect()
}

/// Wsg / Wsu: [F, H] row-major -> shared::WG = m![F/320, H/32, F/8%40, H/4%8, F%8, H%4].
pub fn perm_wg(w: &[bf16]) -> Vec<bf16> {
    let mut out = Vec::with_capacity(w.len());
    for f320 in 0..4 { for h32 in 0..128 { for f8_40 in 0..40 { for h4_8 in 0..8 { for f8 in 0..8 { for h4 in 0..4 {
        let f = f320 * 320 + f8_40 * 8 + f8; let h = h32 * 32 + h4_8 * 4 + h4;
        out.push(w[f * H + h]);
    }}}}}}
    out
}

/// Wsd: [H, F] row-major -> shared::WD = m![H/128, F/80, H/8%16, F/4%20, H%8, F%4].
pub fn perm_wd(w: &[bf16]) -> Vec<bf16> {
    let mut out = Vec::with_capacity(w.len());
    for h128 in 0..32 { for f80 in 0..16 { for h8_16 in 0..16 { for f4_20 in 0..20 { for h8 in 0..8 { for f4 in 0..4 {
        let h = h128 * 128 + h8_16 * 8 + h8; let f = f80 * 80 + f4_20 * 4 + f4;
        out.push(w[h * F + f]);
    }}}}}}
    out
}

/// Wr: [R, H] row-major -> shared::WR = m![R/32, H/32, R/8%4, H/4%8, R%8, H%4].
pub fn perm_wr(w: &[bf16]) -> Vec<bf16> {
    let mut out = Vec::with_capacity(w.len());
    for r32 in 0..4 { for h32 in 0..128 { for r8_4 in 0..4 { for h4_8 in 0..8 { for r8 in 0..8 { for h4 in 0..4 {
        let r = r32 * 32 + r8_4 * 8 + r8; let h = h32 * 32 + h4_8 * 4 + h4;
        out.push(w[r * H + h]);
    }}}}}}
    out
}

/// Wlm: [W, H] row-major -> lm_head::WLM =
/// m![W/8192, W/4096%2, W/512%8, H/128, W/8%64, H/4%32, W%8, H%4] (= m![W/512, H/128, W/8%64, H/4%32, W%8, H%4]).
/// Works for any row count that is a multiple of 512 (used for reduced-vocab CPU checks too).
pub fn perm_wlm(w: &[bf16]) -> Vec<bf16> {
    let rows = w.len() / H;
    assert_eq!(rows % 512, 0);
    let mut out = vec![bf16::from_f32(0.0); w.len()];
    let mut o = 0;
    for b in 0..rows / 512 { for h128 in 0..32 { for w8_64 in 0..64 { for h4_32 in 0..32 { for w8 in 0..8 {
        let r = b * 512 + w8_64 * 8 + w8;
        let base = r * H + h128 * 128 + h4_32 * 4;
        out[o..o + 4].copy_from_slice(&w[base..base + 4]);
        o += 4;
    }}}}}
    out
}

pub fn r(v: f32) -> f32 { f32::from(bf16::from_f32(v)) }
pub fn dot(row: &[bf16], v: &[bf16]) -> f32 { row.iter().zip(v).map(|(&a, &b)| f32::from(a) * f32::from(b)).sum() }

/// One SwiGLU expert, mimicking device rounding (bf16 after gate/up/h), f32 out.
pub fn expert_ref(x: &[bf16], wg: &[bf16], wu: &[bf16], wd: &[bf16]) -> Vec<f32> {
    let mut h = vec![bf16::from_f32(0.0); F];
    for f in 0..F {
        let g = r(dot(&wg[f * H..(f + 1) * H], x));
        let u = r(dot(&wu[f * H..(f + 1) * H], x));
        let s = g / (1.0 + (-g).exp());
        h[f] = bf16::from_f32(s * u);
    }
    (0..H).map(|i| dot(&wd[i * F..(i + 1) * F], &h)).collect()
}

/// HF rmsnorm: bf16(x * rsqrt(mean(x^2)+eps)) * w, rounded to bf16.
pub fn rmsnorm_ref(x: &[bf16], w: &[bf16], eps: f32) -> Vec<bf16> {
    let ms = x.iter().map(|&v| f32::from(v) * f32::from(v)).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(&v, &g)| bf16::from_f32(r(f32::from(v) * inv) * f32::from(g))).collect()
}

/// Pass rule shared by all checks: |got - exp| <= 0.02 |exp| + 0.011, < 1% outliers, all finite.
pub fn check_vals(got: &[f32], exp: &[f32]) -> (bool, f32, usize) {
    let (mut max_err, mut bad, mut finite) = (0f32, 0usize, true);
    for (&g, &e) in got.iter().zip(exp) {
        finite &= g.is_finite();
        let d = (g - e).abs();
        max_err = max_err.max(d);
        if !(d <= 0.02 * e.abs() + 0.01 + 1e-3) { bad += 1; }
    }
    (finite && bad * 100 < got.len() && got.len() == exp.len(), max_err, bad)
}
