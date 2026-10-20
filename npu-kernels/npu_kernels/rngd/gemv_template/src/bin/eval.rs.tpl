//! `eval check|bench`: GEMV ROWS=${ROWS} COLS=${H} against a host oracle; CPU emulation (`cargo run`) or RNGD
//! (`cargo furiosa-opt run`).
use furiosa_opt_std::prelude::*;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use ${CRATE}::{WT, gemv_k};
use ${CRATE}::axes::{F as AF, H as AH, NC as ANC};

type HA = m![AH];
type OUT = m![ANC, AF];
const H: usize = ${H};
const F: usize = ${F};
const NC: usize = ${NC};
const ROWS: usize = ${ROWS};
const R: usize = ${R};

fn r(v: f32) -> f32 { f32::from(bf16::from_f32(v)) }

/// W [ROWS_P, H] row-major -> WT layout m![NC, F / R, H / 32, F / 8 % (R/8), H / 4 % 8, F % 8, H % 4]
fn perm(w: &[bf16]) -> Vec<bf16> {
    let mut out = Vec::with_capacity(w.len());
    for c in 0..NC { for fr in 0..F / R { for h32 in 0..H / 32 { for f8 in 0..R / 8 { for h4 in 0..8 { for f in 0..8 { for h in 0..4 {
        out.push(w[(c * F + fr * R + f8 * 8 + f) * H + h32 * 32 + h4 * 4 + h]);
    }}}}}}}
    out
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> eyre::Result<()> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "check".into());
    let mut rng = SmallRng::seed_from_u64(11);
    let mut w = vec![bf16::from_f32(0.0); NC * F * H];
    for (i, v) in w.iter_mut().enumerate() { if i / H < ROWS { *v = bf16::from_f32((rng.random::<f32>() * 2.0 - 1.0) * 0.03); } }
    let x: Vec<bf16> = (0..H).map(|_| bf16::from_f32(rng.random::<f32>() * 2.0 - 1.0)).collect();
    let mut dev = Device::new(gemv_k.topology())?;
    let w_hbm = HostTensor::<bf16, WT>::from_vec(perm(&w)).to_hbm(&mut dev.pdma).await?;
    let x_hbm = HostTensor::<bf16, HA>::from_vec(x.clone()).to_hbm(&mut dev.pdma).await?;
    let mut out = HostTensor::<bf16, OUT>::zero().to_hbm(&mut dev.pdma).await?;
    launch(gemv_k, (&mut dev, &x_hbm, &w_hbm, &mut out)).await?;
    let got: Vec<bf16> = out.to_host::<OUT>(&mut dev.pdma).await?.into_vec();
    let (mut max_err, mut bad) = (0f32, 0usize);
    for row in 0..ROWS {
        let e = r(w[row * H..(row + 1) * H].iter().zip(&x).map(|(&a, &b)| f32::from(a) * f32::from(b)).sum());
        let d = (f32::from(got[row]) - e).abs();
        max_err = max_err.max(d);
        if d > 0.02 * e.abs() + 1e-3 { bad += 1; }
    }
    println!("rows={ROWS} cols={H}: max_abs_err={max_err:.5} outliers={bad}/{ROWS}");
    let ok = bad * 100 < ROWS;
    if mode == "bench" {
        let mut ts = Vec::new();
        for _ in 0..53 {
            let t = std::time::Instant::now();
            launch(gemv_k, (&mut dev, &x_hbm, &w_hbm, &mut out)).await?;
            ts.push(t.elapsed().as_secs_f64());
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = ts[ts.len() / 2];
        println!("launch p50 {:.1} us, {:.0} GB/s", p50 * 1e6, (NC * F * H * 2) as f64 / p50 / 1e9);
    }
    println!("RESULT {}", if ok { "PASS" } else { "FAIL" });
    if !ok { std::process::exit(1) }
    Ok(())
}
