use furiosa_opt_std::prelude::*;
use moe_opt::kernel::tiny::{P8, tiny_add};
use std::time::Instant;

#[tokio::main(flavor = "current_thread")]
async fn main() -> eyre::Result<()> {
    let mut device = Device::new(tiny_add.topology())?;
    let input = HostTensor::<i32, m![P8]>::from_vec((0..2048i32).collect::<Vec<_>>());
    let input_hbm = input.to_hbm(&mut device.pdma).await?;
    let mut out = HostTensor::<i32, m![P8]>::zero().to_hbm(&mut device.pdma).await?;
    launch(tiny_add, (&mut device, &input_hbm, &mut out)).await?;
    let got: Vec<i32> = out.to_host::<m![P8]>(&mut device.pdma).await?.into_vec();
    let ok = got.iter().enumerate().all(|(i, &v)| v == i as i32 + 1);
    println!("tiny_add correct: {ok}");
    for _ in 0..50 { launch(tiny_add, (&mut device, &input_hbm, &mut out)).await?; }
    let n = 1000;
    let t = Instant::now();
    let mut lat = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = Instant::now();
        launch(tiny_add, (&mut device, &input_hbm, &mut out)).await?;
        lat.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    let total = t.elapsed().as_secs_f64() * 1e6;
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("sync std launch: n={n} mean={:.1}us p50={:.1}us p10={:.1}us p90={:.1}us p99={:.1}us",
        total / n as f64, lat[n / 2], lat[n / 10], lat[n * 9 / 10], lat[n * 99 / 100]);
    Ok(())
}
