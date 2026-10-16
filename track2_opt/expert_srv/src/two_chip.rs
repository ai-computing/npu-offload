//! Can one process drive two chips concurrently? One OS thread + runtime per chip.
use expert_srv::ffn::{WDT, WGT, ffn_k8u};
use furiosa_opt_std::prelude::*;
use std::sync::{Arc, Barrier};
use std::time::Instant;
type HA = m![4096];
type KH = m![8, 4096];

fn worker(bar: Arc<Barrier>, tag: usize, iters: usize) -> eyre::Result<f64> {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    rt.block_on(async move {
        let mut d = Device::new(ffn_k8u.topology())?;
        let z = |n: usize| vec![bf16::from_f32(0.01); n];
        let e = 32 * 1280 * 4096;
        let g = HostTensor::<bf16, WGT>::from_vec(z(e)).to_hbm(&mut d.pdma).await?;
        let u = HostTensor::<bf16, WGT>::from_vec(z(e)).to_hbm(&mut d.pdma).await?;
        let w = HostTensor::<bf16, WDT>::from_vec(z(e)).to_hbm(&mut d.pdma).await?;
        let x = HostTensor::<bf16, HA>::from_vec(z(4096)).to_hbm(&mut d.pdma).await?;
        let mut o = HostTensor::<bf16, KH>::zero().to_hbm(&mut d.pdma).await?;
        let i = HbmScalar::<u64>::from_host((8u64 << 56) + (1 << 6), &mut d.pdma).await?;
        for _ in 0..5 { launch(ffn_k8u, (&mut d, &x, &g, &u, &w, &i, &mut o)).await?; }
        bar.wait();
        let t = Instant::now();
        for _ in 0..iters { launch(ffn_k8u, (&mut d, &x, &g, &u, &w, &i, &mut o)).await?; }
        let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("chip-thread {tag}: {us:.1} us/launch");
        Ok(us)
    })
}

fn main() -> eyre::Result<()> {
    let solo = { let b = Arc::new(Barrier::new(1)); worker(b, 9, 50)? };
    println!("solo (one chip only): {solo:.1} us");
    let bar = Arc::new(Barrier::new(2));
    let hs: Vec<_> = (0..2).map(|k| { let b = bar.clone(); std::thread::spawn(move || worker(b, k, 50)) }).collect();
    for h in hs { h.join().unwrap()?; }
    Ok(())
}
