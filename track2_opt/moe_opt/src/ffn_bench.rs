use furiosa_opt_std::prelude::*;
use moe_opt::kernel::ffn::{WD, WG, ffn1};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::time::Instant;
#[path = "host_util.rs"]
mod host_util;
use host_util::*;
type HA = m![4096];

#[tokio::main(flavor = "current_thread")]
async fn main() -> eyre::Result<()> {
    let mut device = Device::new(ffn1.topology())?;
    let mut rng = SmallRng::seed_from_u64(7);
    let x = rand_bf16(&mut rng, H, 1.0);
    let wg = rand_bf16(&mut rng, F * H, 0.03);
    let wu = rand_bf16(&mut rng, F * H, 0.03);
    let wd = rand_bf16(&mut rng, H * F, 0.03);
    let exp = expert_ref(&x, &wg, &wu, &wd);
    let x_hbm = HostTensor::<bf16, HA>::from_vec(x.clone()).to_hbm(&mut device.pdma).await?;
    let wg_hbm = HostTensor::<bf16, WG>::from_vec(perm_wg(&wg)).to_hbm(&mut device.pdma).await?;
    let wu_hbm = HostTensor::<bf16, WG>::from_vec(perm_wg(&wu)).to_hbm(&mut device.pdma).await?;
    let wd_hbm = HostTensor::<bf16, WD>::from_vec(perm_wd(&wd)).to_hbm(&mut device.pdma).await?;
    let mut out = HostTensor::<bf16, HA>::zero().to_hbm(&mut device.pdma).await?;
    launch(ffn1, (&mut device, &x_hbm, &wg_hbm, &wu_hbm, &wd_hbm, &mut out)).await?;
    let got: Vec<bf16> = out.to_host::<HA>(&mut device.pdma).await?.into_vec();
    check("ffn1 (1 expert)", &got, &exp);
    for _ in 0..20 { launch(ffn1, (&mut device, &x_hbm, &wg_hbm, &wu_hbm, &wd_hbm, &mut out)).await?; }
    let mut lat = vec![];
    for _ in 0..300 { let t = Instant::now(); launch(ffn1, (&mut device, &x_hbm, &wg_hbm, &wu_hbm, &wd_hbm, &mut out)).await?; lat.push(t.elapsed().as_secs_f64() * 1e6); }
    report("ffn1 sync launch", 3 * F * H * 2, lat);
    Ok(())
}
