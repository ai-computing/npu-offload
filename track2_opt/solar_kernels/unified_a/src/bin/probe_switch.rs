//! probe_switch: per-launch time of tiny kernels: same program repeated vs two programs alternating
//! vs one program with a runtime mode alternating.
use furiosa_opt_std::prelude::*;
use solar_unified_a::axes::{H as AH, K as AK};
use solar_unified_a::probe::{p_a, p_b, p_u};
use std::time::Instant;
type VH = m![AH];
type KH = m![AK, AH];
fn p50(mut v: Vec<f64>) -> String {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    format!("p50 {:.1} us  p10 {:.1}  p90 {:.1}", v[v.len() / 2], v[v.len() / 10], v[v.len() * 9 / 10])
}
#[tokio::main]
async fn main() -> eyre::Result<()> {
    let n: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut dev = Device::new(p_a.topology())?;
    let a = HostTensor::<bf16, VH>::from_vec((0..4096).map(|i| bf16::from_f32(i as f32)).collect::<Vec<_>>()).to_hbm(&mut dev.pdma).await?;
    let mut b = HostTensor::<bf16, KH>::from_vec(vec![bf16::from_f32(0.0); 8 * 4096]).to_hbm(&mut dev.pdma).await?;
    let m0 = HbmScalar::<u64>::from_host(0, &mut dev.pdma).await?;
    let m1 = HbmScalar::<u64>::from_host(1, &mut dev.pdma).await?;
    // correctness of p_u: mode 1 -> row 0, mode 0 -> row 1
    launch(p_u, (&mut dev, &m1, &a, &mut b)).await?;
    let v: Vec<bf16> = b.to_host::<KH>(&mut dev.pdma).await?.into_vec();
    println!("p_u mode1: row0[5]={} row1[5]={}", f32::from(v[5]), f32::from(v[4096 + 5]));
    launch(p_u, (&mut dev, &m0, &a, &mut b)).await?;
    let v: Vec<bf16> = b.to_host::<KH>(&mut dev.pdma).await?.into_vec();
    println!("p_u mode0: row0[5]={} row1[5]={} row2[5]={}", f32::from(v[5]), f32::from(v[4096 + 5]), f32::from(v[2 * 4096 + 5]));
    for _ in 0..10 {
        launch(p_a, (&mut dev, &a, &mut b)).await?;
        launch(p_b, (&mut dev, &a, &mut b)).await?;
        launch(p_u, (&mut dev, &m0, &a, &mut b)).await?;
    }
    let mut t = Vec::new();
    for _ in 0..n { let s = Instant::now(); launch(p_a, (&mut dev, &a, &mut b)).await?; t.push(s.elapsed().as_secs_f64() * 1e6); }
    println!("p_a repeated          : {}", p50(t));
    let mut t = Vec::new();
    for i in 0..n { let s = Instant::now(); if i % 2 == 0 { launch(p_a, (&mut dev, &a, &mut b)).await?; } else { launch(p_b, (&mut dev, &a, &mut b)).await?; } t.push(s.elapsed().as_secs_f64() * 1e6); }
    println!("p_a/p_b alternating   : {}", p50(t));
    let mut t = Vec::new();
    for _ in 0..n { let s = Instant::now(); launch(p_u, (&mut dev, &m0, &a, &mut b)).await?; t.push(s.elapsed().as_secs_f64() * 1e6); }
    println!("p_u mode0 repeated    : {}", p50(t));
    let mut t = Vec::new();
    for i in 0..n { let s = Instant::now(); launch(p_u, (&mut dev, if i % 2 == 0 { &m0 } else { &m1 }, &a, &mut b)).await?; t.push(s.elapsed().as_secs_f64() * 1e6); }
    println!("p_u mode alternating  : {}", p50(t));
    Ok(())
}
