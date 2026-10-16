// (included via #[path] by bench binaries)
use furiosa_opt_std::prelude::bf16;
use rand::Rng;

pub const F: usize = 1280;
pub const H: usize = 4096;

pub fn rand_bf16(rng: &mut impl Rng, n: usize, scale: f32) -> Vec<bf16> {
    (0..n).map(|_| bf16::from_f32((rng.random::<f32>() * 2.0 - 1.0) * scale)).collect()
}

/// [F, H] row-major -> WG device layout.
pub fn perm_wg(w: &[bf16]) -> Vec<bf16> {
    let mut out = Vec::with_capacity(w.len());
    for f320 in 0..4 { for h32 in 0..128 { for f8_40 in 0..40 { for h4_8 in 0..8 { for f8 in 0..8 { for h4 in 0..4 {
        let f = f320 * 320 + f8_40 * 8 + f8; let h = h32 * 32 + h4_8 * 4 + h4;
        out.push(w[f * H + h]);
    }}}}}}
    out
}

/// [H, F] row-major -> WD device layout.
pub fn perm_wd(w: &[bf16]) -> Vec<bf16> {
    let mut out = Vec::with_capacity(w.len());
    for h128 in 0..32 { for f80 in 0..16 { for h8_16 in 0..16 { for f4_20 in 0..20 { for h8 in 0..8 { for f4 in 0..4 {
        let h = h128 * 128 + h8_16 * 8 + h8; let f = f80 * 80 + f4_20 * 4 + f4;
        out.push(w[h * F + f]);
    }}}}}}
    out
}

fn r(v: f32) -> f32 { f32::from(bf16::from_f32(v)) }

/// CPU reference of one expert, mimicking device rounding points (bf16 after gate/up/h).
pub fn expert_ref(x: &[bf16], wg: &[bf16], wu: &[bf16], wd: &[bf16]) -> Vec<f32> {
    let dot = |row: &[bf16], v: &[bf16]| -> f32 { row.iter().zip(v).map(|(&a, &b)| f32::from(a) * f32::from(b)).sum() };
    let mut h = vec![bf16::from_f32(0.0); F];
    for f in 0..F {
        let g = r(dot(&wg[f * H..(f + 1) * H], x));
        let u = r(dot(&wu[f * H..(f + 1) * H], x));
        let s = g / (1.0 + (-g).exp());
        h[f] = bf16::from_f32(s * u);
    }
    (0..H).map(|i| dot(&wd[i * F..(i + 1) * F], &h)).collect()
}

pub fn check(name: &str, got: &[bf16], exp: &[f32]) -> bool {
    let mut max_err = 0f32; let mut max_ref = 0f32; let mut bad = 0;
    for (&g, &e) in got.iter().zip(exp) {
        let d = (f32::from(g) - e).abs();
        max_err = max_err.max(d); max_ref = max_ref.max(e.abs());
        if d > 0.02 * e.abs() + 0.01 * 1.0f32.max(0.0) + 1e-3 { bad += 1; }
    }
    let rms_ref = (exp.iter().map(|e| e * e).sum::<f32>() / exp.len() as f32).sqrt();
    println!("{name}: max_abs_err={max_err:.5} max|ref|={max_ref:.4} rms_ref={rms_ref:.4} outliers(>2%+1e-3)={bad}/{}", got.len());
    bad * 100 < got.len()
}

pub fn report(name: &str, bytes: usize, mut v: Vec<f64>) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len(); let p50 = v[n / 2]; let min = v[0];
    println!("{name}: p50={p50:.1}us min={min:.1}us p90={:.1}us -> {:.0} GB/s (p50)", v[n * 9 / 10], bytes as f64 / p50 / 1e3);
}
