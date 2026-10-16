//! Harness entry point: `fko_eval <kernel> <check|bench>`.
//! Plain `cargo run` = CPU emulation, `cargo furiosa-opt run` = RNGD.
//! Emits one `FKO {json}` line per check/metric for harness/fko.py to parse.
use furiosa_opt_std::prelude::*;
use moe_opt::kernel::ffn::{H as FfnH, K as FfnK, WD, WDT, WG, WGT, ffn_k8u, ffn1};
use moe_opt::kernel::gemv::{A as GemvA, F as GemvF, H as GemvH, WDn3, WGu3, gemv_dn3, gemv_gu3};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::time::Instant;
#[path = "host_util.rs"]
mod host_util;
use host_util::*;

// Host tensors must use the kernel module's own axis symbols (an anonymous m![4096] breaks CPU emulation).
type HA = m![FfnH];
type KH = m![FfnK, FfnH];
type GH = m![GemvH];
type GF = m![GemvF];
type GA = m![GemvA];
const A: usize = 2560;
const E: usize = 64;
const WARMUP: usize = 20;
const ITERS: usize = 200;

fn gemv_ref(w: &[bf16], x: &[bf16], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows).map(|r| w[r * cols..(r + 1) * cols].iter().zip(x).map(|(&a, &b)| f32::from(a) * f32::from(b)).sum()).collect()
}

/// [A, H] row-major -> WGu3 layout.
fn perm_gu3(w: &[bf16]) -> Vec<bf16> {
    let mut out = Vec::with_capacity(w.len());
    for a640 in 0..4 { for h32 in 0..128 { for a8_80 in 0..80 { for h4_8 in 0..8 { for a8 in 0..8 { for h4 in 0..4 {
        out.push(w[(a640 * 640 + a8_80 * 8 + a8) * H + h32 * 32 + h4_8 * 4 + h4]);
    }}}}}}
    out
}

/// Same tolerance rule as host_util::check, reported as an FKO line.
fn fko_check(kernel: &str, got: &[bf16], exp: &[f32]) -> bool {
    let (mut max_err, mut bad) = (0f32, 0usize);
    let mut finite = true;
    for (&g, &e) in got.iter().zip(exp) {
        let g = f32::from(g);
        finite &= g.is_finite();
        let d = (g - e).abs();
        max_err = max_err.max(d);
        if !(d <= 0.02 * e.abs() + 0.01 + 1e-3) { bad += 1; }
    }
    let pass = finite && bad * 100 < got.len() && got.len() == exp.len();
    println!(
        "FKO {{\"kernel\":\"{kernel}\",\"check\":\"{}\",\"max_abs_err\":{max_err},\"outliers\":{bad},\"n\":{}}}",
        if pass { "pass" } else { "fail" },
        got.len()
    );
    pass
}

fn fko_metric(kernel: &str, bytes: usize, mut v: Vec<f64>) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let p50 = v[n / 2];
    println!(
        "FKO {{\"kernel\":\"{kernel}\",\"metric\":\"launch_us\",\"p50\":{p50:.2},\"p90\":{:.2},\"min\":{:.2},\"n\":{n},\"bytes\":{bytes},\"gbps_p50\":{:.1}}}",
        v[n * 9 / 10],
        v[0],
        bytes as f64 / p50 / 1e3
    );
}

macro_rules! time_launches {
    ($kernel:expr, $args:expr) => {{
        for _ in 0..WARMUP { launch($kernel, $args).await?; }
        let mut lat = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t = Instant::now();
            launch($kernel, $args).await?;
            lat.push(t.elapsed().as_secs_f64() * 1e6);
        }
        lat
    }};
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> eyre::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (kernel, mode) = (args.get(1).map(String::as_str).unwrap_or(""), args.get(2).map(String::as_str).unwrap_or("check"));
    let bench = mode == "bench";
    let mut rng = SmallRng::seed_from_u64(7);
    let ok = match kernel {
        "gemv_gu3" => {
            let mut device = Device::new(gemv_gu3.topology())?;
            let (w, x) = (rand_bf16(&mut rng, A * H, 0.03), rand_bf16(&mut rng, H, 1.0));
            let w_hbm = HostTensor::<bf16, WGu3>::from_vec(perm_gu3(&w)).to_hbm(&mut device.pdma).await?;
            let x_hbm = HostTensor::<bf16, GH>::from_vec(x.clone()).to_hbm(&mut device.pdma).await?;
            let mut y = HostTensor::<bf16, GA>::zero().to_hbm(&mut device.pdma).await?;
            launch(gemv_gu3, (&mut device, &w_hbm, &x_hbm, &mut y)).await?;
            let got: Vec<bf16> = y.to_host::<GA>(&mut device.pdma).await?.into_vec();
            let ok = fko_check(kernel, &got, &gemv_ref(&w, &x, A, H));
            if bench { fko_metric(kernel, A * H * 2, time_launches!(gemv_gu3, (&mut device, &w_hbm, &x_hbm, &mut y))); }
            ok
        }
        "gemv_dn3" => {
            let mut device = Device::new(gemv_dn3.topology())?;
            let (w, x) = (rand_bf16(&mut rng, H * F, 0.03), rand_bf16(&mut rng, F, 1.0));
            let w_hbm = HostTensor::<bf16, WDn3>::from_vec(perm_wd(&w)).to_hbm(&mut device.pdma).await?;
            let x_hbm = HostTensor::<bf16, GF>::from_vec(x.clone()).to_hbm(&mut device.pdma).await?;
            let mut y = HostTensor::<bf16, GH>::zero().to_hbm(&mut device.pdma).await?;
            launch(gemv_dn3, (&mut device, &w_hbm, &x_hbm, &mut y)).await?;
            let got: Vec<bf16> = y.to_host::<GH>(&mut device.pdma).await?.into_vec();
            let ok = fko_check(kernel, &got, &gemv_ref(&w, &x, H, F));
            if bench { fko_metric(kernel, H * F * 2, time_launches!(gemv_dn3, (&mut device, &w_hbm, &x_hbm, &mut y))); }
            ok
        }
        "ffn1" => {
            let mut device = Device::new(ffn1.topology())?;
            let x = rand_bf16(&mut rng, H, 1.0);
            let (wg, wu, wd) = (rand_bf16(&mut rng, F * H, 0.03), rand_bf16(&mut rng, F * H, 0.03), rand_bf16(&mut rng, H * F, 0.03));
            let x_hbm = HostTensor::<bf16, HA>::from_vec(x.clone()).to_hbm(&mut device.pdma).await?;
            let wg_hbm = HostTensor::<bf16, WG>::from_vec(perm_wg(&wg)).to_hbm(&mut device.pdma).await?;
            let wu_hbm = HostTensor::<bf16, WG>::from_vec(perm_wg(&wu)).to_hbm(&mut device.pdma).await?;
            let wd_hbm = HostTensor::<bf16, WD>::from_vec(perm_wd(&wd)).to_hbm(&mut device.pdma).await?;
            let mut out = HostTensor::<bf16, HA>::zero().to_hbm(&mut device.pdma).await?;
            launch(ffn1, (&mut device, &x_hbm, &wg_hbm, &wu_hbm, &wd_hbm, &mut out)).await?;
            let got: Vec<bf16> = out.to_host::<HA>(&mut device.pdma).await?.into_vec();
            let ok = fko_check(kernel, &got, &expert_ref(&x, &wg, &wu, &wd));
            if bench { fko_metric(kernel, 3 * F * H * 2, time_launches!(ffn1, (&mut device, &x_hbm, &wg_hbm, &wu_hbm, &wd_hbm, &mut out))); }
            ok
        }
        "ffn_k8u" => {
            let mut device = Device::new(ffn_k8u.topology())?;
            // Only the 8 selected experts get random weights; the rest stay zero to keep host setup cheap.
            let ids = [3usize, 17, 42, 63, 0, 8, 29, 51];
            let x = rand_bf16(&mut rng, H, 1.0);
            let (mut tg, mut tu, mut td) = (vec![bf16::from_f32(0.0); E * F * H], vec![bf16::from_f32(0.0); E * F * H], vec![bf16::from_f32(0.0); E * F * H]);
            let mut experts = Vec::new();
            for &e in &ids {
                let (g, u, d) = (rand_bf16(&mut rng, F * H, 0.03), rand_bf16(&mut rng, F * H, 0.03), rand_bf16(&mut rng, H * F, 0.03));
                tg[e * F * H..(e + 1) * F * H].copy_from_slice(&perm_wg(&g));
                tu[e * F * H..(e + 1) * F * H].copy_from_slice(&perm_wg(&u));
                td[e * F * H..(e + 1) * F * H].copy_from_slice(&perm_wd(&d));
                experts.push((g, u, d));
            }
            let tg_hbm = HostTensor::<bf16, WGT>::from_vec(tg).to_hbm(&mut device.pdma).await?;
            let tu_hbm = HostTensor::<bf16, WGT>::from_vec(tu).to_hbm(&mut device.pdma).await?;
            let td_hbm = HostTensor::<bf16, WDT>::from_vec(td).to_hbm(&mut device.pdma).await?;
            let x_hbm = HostTensor::<bf16, HA>::from_vec(x.clone()).to_hbm(&mut device.pdma).await?;
            let mut packed = (ids.len() as u64) << 56;
            for (k, &e) in ids.iter().enumerate() { packed += (e as u64) << (6 * k); }
            let ids_hbm = HbmScalar::<u64>::from_host(packed, &mut device.pdma).await?;
            let mut out = HostTensor::<bf16, KH>::zero().to_hbm(&mut device.pdma).await?;
            launch(ffn_k8u, (&mut device, &x_hbm, &tg_hbm, &tu_hbm, &td_hbm, &ids_hbm, &mut out)).await?;
            let got: Vec<bf16> = out.to_host::<KH>(&mut device.pdma).await?.into_vec();
            let exp: Vec<f32> = experts.iter().flat_map(|(g, u, d)| expert_ref(&x, g, u, d)).collect();
            let ok = fko_check(kernel, &got, &exp);
            if bench { fko_metric(kernel, 8 * 3 * F * H * 2, time_launches!(ffn_k8u, (&mut device, &x_hbm, &tg_hbm, &tu_hbm, &td_hbm, &ids_hbm, &mut out))); }
            ok
        }
        _ => eyre::bail!("unknown kernel {kernel:?}; expected gemv_gu3|gemv_dn3|ffn1|ffn_k8u"),
    };
    std::process::exit(if ok { 0 } else { 1 });
}
