use furiosa_opt_std::prelude::*;
use moe_opt::kernel::gemv::{A, F, H, WGu3, gemv_down, gemv_gu, gemv_gu3};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::time::Instant;

fn gemv_ref(w: &[bf16], x: &[bf16], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows).map(|r| w[r * cols..(r + 1) * cols].iter().zip(x).map(|(&a, &b)| f32::from(a) * f32::from(b)).sum()).collect()
}

fn check(name: &str, got: &[bf16], exp: &[f32]) {
    let mut max_err = 0f32; let mut max_ref = 0f32; let mut bad = 0;
    for (&g, &e) in got.iter().zip(exp) {
        let d = (f32::from(g) - e).abs();
        max_err = max_err.max(d); max_ref = max_ref.max(e.abs());
        if d > 0.02 * e.abs() + 0.05 { bad += 1; }
    }
    println!("{name}: max_abs_err={max_err:.4} max|ref|={max_ref:.3} bad={bad}/{}", got.len());
}

fn report(name: &str, bytes: usize, mut v: Vec<f64>) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len(); let p50 = v[n / 2]; let min = v[0];
    println!("{name}: p50={p50:.1}us min={min:.1}us p90={:.1}us -> {:.0} GB/s (p50), {:.0} GB/s (min)",
        v[n * 9 / 10], bytes as f64 / p50 / 1e3, bytes as f64 / min / 1e3);
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> eyre::Result<()> {
    let mut device = Device::new(gemv_gu.topology())?;
    let mut rng = SmallRng::seed_from_u64(1);
    // scale weights ~ 0.02 like real models: rand gives [0,1)? we just check relative.
    let w = HostTensor::<bf16, m![A, H]>::rand(&mut rng);
    let x = HostTensor::<bf16, m![H]>::rand(&mut rng);
    let w_hbm = w.to_hbm(&mut device.pdma).await?;
    let x_hbm = x.to_hbm(&mut device.pdma).await?;
    let mut y = HostTensor::<bf16, m![A]>::zero().to_hbm(&mut device.pdma).await?;
    launch(gemv_gu, (&mut device, &w_hbm, &x_hbm, &mut y)).await?;
    let got: Vec<bf16> = y.to_host::<m![A]>(&mut device.pdma).await?.into_vec();
    let exp = gemv_ref(&w.clone().into_vec(), &x.clone().into_vec(), A::SIZE, H::SIZE);
    check("gemv_gu [2560x4096]", &got, &exp);
    let mut lat = vec![];
    for _ in 0..20 { launch(gemv_gu, (&mut device, &w_hbm, &x_hbm, &mut y)).await?; }
    for _ in 0..300 { let t = Instant::now(); launch(gemv_gu, (&mut device, &w_hbm, &x_hbm, &mut y)).await?; lat.push(t.elapsed().as_secs_f64() * 1e6); }
    report("gemv_gu sync launch", A::SIZE * H::SIZE * 2, lat);

    // gu3: pre-permuted weight layout
    {
        let wv: Vec<bf16> = w.clone().into_vec();
        let mut perm = Vec::with_capacity(wv.len());
        for a640 in 0..4 { for h32 in 0..128 { for a8_80 in 0..80 { for h4_8 in 0..8 { for a8 in 0..8 { for h4 in 0..4 {
            let r = a640 * 640 + a8_80 * 8 + a8; let c = h32 * 32 + h4_8 * 4 + h4;
            perm.push(wv[r * H::SIZE + c]);
        }}}}}}
        let w3 = HostTensor::<bf16, WGu3>::from_vec(perm);
        let w3_hbm = w3.to_hbm(&mut device.pdma).await?;
        launch(gemv_gu3, (&mut device, &w3_hbm, &x_hbm, &mut y)).await?;
        let got: Vec<bf16> = y.to_host::<m![A]>(&mut device.pdma).await?.into_vec();
        check("gemv_gu3 [2560x4096]", &got, &exp);
        let mut lat = vec![];
        for _ in 0..20 { launch(gemv_gu3, (&mut device, &w3_hbm, &x_hbm, &mut y)).await?; }
        for _ in 0..300 { let t = Instant::now(); launch(gemv_gu3, (&mut device, &w3_hbm, &x_hbm, &mut y)).await?; lat.push(t.elapsed().as_secs_f64() * 1e6); }
        report("gemv_gu3 sync launch", A::SIZE * H::SIZE * 2, lat);
    }

    let wd = HostTensor::<bf16, m![H, F]>::rand(&mut rng);
    let xd = HostTensor::<bf16, m![F]>::rand(&mut rng);
    let wd_hbm = wd.to_hbm(&mut device.pdma).await?;
    let xd_hbm = xd.to_hbm(&mut device.pdma).await?;
    let mut yd = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut device.pdma).await?;
    launch(gemv_down, (&mut device, &wd_hbm, &xd_hbm, &mut yd)).await?;
    let got: Vec<bf16> = yd.to_host::<m![H]>(&mut device.pdma).await?.into_vec();
    let exp = gemv_ref(&wd.clone().into_vec(), &xd.clone().into_vec(), H::SIZE, F::SIZE);
    check("gemv_down [4096x1280]", &got, &exp);
    let mut lat = vec![];
    for _ in 0..20 { launch(gemv_down, (&mut device, &wd_hbm, &xd_hbm, &mut yd)).await?; }
    for _ in 0..300 { let t = Instant::now(); launch(gemv_down, (&mut device, &wd_hbm, &xd_hbm, &mut yd)).await?; lat.push(t.elapsed().as_secs_f64() * 1e6); }
    report("gemv_down sync launch", H::SIZE * F::SIZE * 2, lat);
    Ok(())
}
