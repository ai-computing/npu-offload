//! `eval check`: CPU emulation (`cargo run`) or RNGD (`cargo furiosa-opt run`) check of ffn_k against a host oracle.
//! Generated for H=${H}, F=${F}.
use furiosa_opt_std::prelude::*;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use ${CRATE}::{WDT, WGT, ffn_k};
use ${CRATE}::axes::{H as AH, K as AK};

type HA = m![AH];
type KH = m![AK, AH];
const H: usize = ${H};
const F: usize = ${F};
const E: usize = ${E};
const K: usize = ${K};
const R: usize = ${R};
const HR: usize = ${HR};
const FC: usize = ${FC};

fn rand_bf16(rng: &mut impl Rng, n: usize, scale: f32) -> Vec<bf16> {
    (0..n).map(|_| bf16::from_f32((rng.random::<f32>() * 2.0 - 1.0) * scale)).collect()
}
fn r(v: f32) -> f32 { f32::from(bf16::from_f32(v)) }

/// [F, H] row-major -> WG layout m![F / R, H / 32, F / 8 % (R/8), H / 4 % 8, F % 8, H % 4]
fn perm_wg(w: &[bf16]) -> Vec<bf16> {
    let mut out = Vec::with_capacity(w.len());
    for fr in 0..F / R { for h32 in 0..H / 32 { for f8 in 0..R / 8 { for h4 in 0..8 { for f in 0..8 { for h in 0..4 {
        out.push(w[(fr * R + f8 * 8 + f) * H + h32 * 32 + h4 * 4 + h]);
    }}}}}}
    out
}
/// [H, F] row-major -> WD layout m![H / HR, F / FC, H / 8 % (HR/8), F / 4 % (FC/4), H % 8, F % 4]
fn perm_wd(w: &[bf16]) -> Vec<bf16> {
    let mut out = Vec::with_capacity(w.len());
    for hr in 0..H / HR { for fc in 0..F / FC { for h8 in 0..HR / 8 { for f4 in 0..FC / 4 { for h in 0..8 { for f in 0..4 {
        out.push(w[(hr * HR + h8 * 8 + h) * F + fc * FC + f4 * 4 + f]);
    }}}}}}
    out
}
fn expert_ref(x: &[bf16], wg: &[bf16], wu: &[bf16], wd: &[bf16]) -> Vec<f32> {
    let dot = |row: &[bf16], v: &[bf16]| -> f32 { row.iter().zip(v).map(|(&a, &b)| f32::from(a) * f32::from(b)).sum() };
    let mut h = vec![bf16::from_f32(0.0); F];
    for f in 0..F {
        let g = r(dot(&wg[f * H..(f + 1) * H], x));
        let u = r(dot(&wu[f * H..(f + 1) * H], x));
        h[f] = bf16::from_f32(g / (1.0 + (-g).exp()) * u);
    }
    (0..H).map(|i| r(dot(&wd[i * F..(i + 1) * F], &h))).collect()
}
fn check(name: &str, got: &[bf16], exp: &[f32]) -> bool {
    let (mut max_err, mut bad) = (0f32, 0usize);
    for (&g, &e) in got.iter().zip(exp) {
        let d = (f32::from(g) - e).abs();
        max_err = max_err.max(d);
        if d > 0.02 * e.abs() + 1e-3 { bad += 1; }
    }
    println!("{name}: max_abs_err={max_err:.5} outliers={bad}/{}", got.len());
    bad * 100 < got.len()
}
fn pack(slots: &[usize]) -> u64 {
    let mut p = (slots.len() as u64) << 56;
    for (k, &s) in slots.iter().enumerate() { p += (s as u64) << (6 * k); }
    p
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> eyre::Result<()> {
    let mut rng = SmallRng::seed_from_u64(7);
    let used: Vec<usize> = (0..K).map(|k| (k * 5 + 3) % E).collect();
    let (mut tg, mut tu, mut td) = (vec![], vec![], vec![]);
    let mut model = vec![(vec![], vec![], vec![]); E];
    for e in 0..E {
        let (g, u, d) = if used.contains(&e) {
            (rand_bf16(&mut rng, F * H, 0.03), rand_bf16(&mut rng, F * H, 0.03), rand_bf16(&mut rng, H * F, 0.03))
        } else {
            (vec![bf16::from_f32(0.0); F * H], vec![bf16::from_f32(0.0); F * H], vec![bf16::from_f32(0.0); H * F])
        };
        tg.extend(perm_wg(&g)); tu.extend(perm_wg(&u)); td.extend(perm_wd(&d));
        model[e] = (g, u, d);
    }
    let x = rand_bf16(&mut rng, H, 1.0);
    let mut dev = Device::new(ffn_k.topology())?;
    let wg_t = HostTensor::<bf16, WGT>::from_vec(tg).to_hbm(&mut dev.pdma).await?;
    let wu_t = HostTensor::<bf16, WGT>::from_vec(tu).to_hbm(&mut dev.pdma).await?;
    let wd_t = HostTensor::<bf16, WDT>::from_vec(td).to_hbm(&mut dev.pdma).await?;
    let x_hbm = HostTensor::<bf16, HA>::from_vec(x.clone()).to_hbm(&mut dev.pdma).await?;
    let mut ids = HbmScalar::<u64>::from_host(0, &mut dev.pdma).await?;
    let mut rows = HostTensor::<bf16, KH>::zero().to_hbm(&mut dev.pdma).await?;
    let mut ok = true;
    for n in [1usize, 3, K] {
        let slots = &used[..n];
        ids.write(pack(slots), &mut dev.pdma).await?;
        let t = std::time::Instant::now();
        launch(ffn_k, (&mut dev, &x_hbm, &wg_t, &wu_t, &wd_t, &ids, &mut rows)).await?;
        let dt = t.elapsed().as_secs_f64();
        let got: Vec<bf16> = rows.to_host::<KH>(&mut dev.pdma).await?.into_vec();
        for (k, &e) in slots.iter().enumerate() {
            let (g, u, d) = &model[e];
            ok &= check(&format!("n={n} row {k} (slot {e})"), &got[k * H..(k + 1) * H], &expert_ref(&x, g, u, d));
        }
        println!("n={n}: launch {:.3}s", dt);
    }
    println!("RESULT {}", if ok { "PASS" } else { "FAIL" });
    if !ok { std::process::exit(1) }
    Ok(())
}
