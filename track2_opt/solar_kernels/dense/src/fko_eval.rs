//! Harness entry point: `fko_eval <kernel> <check|bench>`.
//! Plain `cargo run` = CPU emulation, `cargo furiosa-opt run` = RNGD.
//! Emits one `FKO {json}` line per check/metric for harness/fko.py to parse.
use furiosa_opt_std::prelude::*;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use solar_dense::kernel::axes::{H as AH, K as AK, R as AR, W as AW};
use solar_dense::kernel::combine::combine;
use solar_dense::kernel::lm_head::{WLM, lm_head, lm_head_am};
use solar_dense::kernel::shared::{P as AP, WD, WG, WR, shared_router, shared_router_cat};
use std::time::Instant;
#[path = "host_util.rs"]
mod host_util;
use host_util::*;

type HA = m![AH];
type RA = m![AR];
type KH = m![AK, AH];
type KA = m![AK];
type WA = m![AW];
type PR = m![AP, AR];
type WS = m![AW / 512];
type WI = m![AW % 512];
const WARMUP: usize = 20;

fn fko_check(kernel: &str, got: &[f32], exp: &[f32]) -> bool {
    let (pass, max_err, bad) = check_vals(got, exp);
    println!(
        "FKO {{\"kernel\":\"{kernel}\",\"check\":\"{}\",\"max_abs_err\":{max_err},\"outliers\":{bad},\"n\":{}}}",
        if pass { "pass" } else { "fail" },
        got.len()
    );
    pass
}

fn fko_metric(kernel: &str, metric: &str, bytes: usize, mut v: Vec<f64>) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let p50 = v[n / 2];
    println!(
        "FKO {{\"kernel\":\"{kernel}\",\"metric\":\"{metric}\",\"p50\":{p50:.2},\"p90\":{:.2},\"min\":{:.2},\"n\":{n},\"bytes\":{bytes},\"gbps_p50\":{:.1}}}",
        v[n * 9 / 10],
        v[0],
        bytes as f64 / p50 / 1e3
    );
}

macro_rules! time_n {
    ($n:expr, $body:expr) => {{
        for _ in 0..WARMUP { $body; }
        let mut lat = Vec::with_capacity($n);
        for _ in 0..$n {
            let t = Instant::now();
            $body;
            lat.push(t.elapsed().as_secs_f64() * 1e6);
        }
        lat
    }};
}

fn f32s(v: &[bf16]) -> Vec<f32> { v.iter().map(|&x| f32::from(x)).collect() }

#[tokio::main(flavor = "current_thread")]
async fn main() -> eyre::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (kernel, mode) = (args.get(1).map(String::as_str).unwrap_or(""), args.get(2).map(String::as_str).unwrap_or("check"));
    let bench = mode == "bench";
    let iters: usize = std::env::var("FKO_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut rng = SmallRng::seed_from_u64(7);
    let ok = match kernel {
        "shared_router" => {
            let mut device = Device::new(shared_router.topology())?;
            let (h2, x2) = (rand_bf16(&mut rng, H, 1.0), rand_bf16(&mut rng, H, 1.0));
            let (wg, wu, wd) = (rand_bf16(&mut rng, F * H, 0.03), rand_bf16(&mut rng, F * H, 0.03), rand_bf16(&mut rng, H * F, 0.03));
            let wr = rand_bf16(&mut rng, R * H, 0.03);
            let h2_hbm = HostTensor::<bf16, HA>::from_vec(h2.clone()).to_hbm(&mut device.pdma).await?;
            let x2_hbm = HostTensor::<bf16, HA>::from_vec(x2.clone()).to_hbm(&mut device.pdma).await?;
            let wg_hbm = HostTensor::<bf16, WG>::from_vec(perm_wg(&wg)).to_hbm(&mut device.pdma).await?;
            let wu_hbm = HostTensor::<bf16, WG>::from_vec(perm_wg(&wu)).to_hbm(&mut device.pdma).await?;
            let wd_hbm = HostTensor::<bf16, WD>::from_vec(perm_wd(&wd)).to_hbm(&mut device.pdma).await?;
            let wr_hbm = HostTensor::<bf16, WR>::from_vec(perm_wr(&wr)).to_hbm(&mut device.pdma).await?;
            let mut y = HostTensor::<bf16, HA>::zero().to_hbm(&mut device.pdma).await?;
            let mut lg = HostTensor::<f32, RA>::zero().to_hbm(&mut device.pdma).await?;
            launch(shared_router, (&mut device, &h2_hbm, &x2_hbm, &wg_hbm, &wu_hbm, &wd_hbm, &wr_hbm, &mut y, &mut lg)).await?;
            let got_y: Vec<bf16> = y.to_host::<HA>(&mut device.pdma).await?.into_vec();
            let got_l: Vec<f32> = lg.to_host::<RA>(&mut device.pdma).await?.into_vec();
            let e = expert_ref(&h2, &wg, &wu, &wd);
            let exp_y: Vec<f32> = e.iter().zip(&x2).map(|(&a, &b)| r(a + f32::from(b))).collect();
            let exp_l: Vec<f32> = (0..R).map(|i| dot(&wr[i * H..(i + 1) * H], &h2)).collect();
            let ok_y = fko_check("shared_router.y", &f32s(&got_y), &exp_y);
            let ok_l = fko_check("shared_router.logits", &got_l, &exp_l);
            let lmax = got_l.iter().zip(&exp_l).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            println!("router logits f32 max abs err {lmax:e}");
            if bench {
                let lat = time_n!(iters, launch(shared_router, (&mut device, &h2_hbm, &x2_hbm, &wg_hbm, &wu_hbm, &wd_hbm, &wr_hbm, &mut y, &mut lg)).await?);
                fko_metric(kernel, "launch_us", (3 * F * H + R * H) * 2, lat);
            }
            ok_y && ok_l
        }
        "shared_router_cat" => {
            let mut device = Device::new(shared_router_cat.topology())?;
            let (h2, x2) = (rand_bf16(&mut rng, H, 1.0), rand_bf16(&mut rng, H, 1.0));
            let (wg, wu, wd) = (rand_bf16(&mut rng, F * H, 0.03), rand_bf16(&mut rng, F * H, 0.03), rand_bf16(&mut rng, H * F, 0.03));
            let wr = rand_bf16(&mut rng, R * H, 0.03);
            let h2_hbm = HostTensor::<bf16, HA>::from_vec(h2.clone()).to_hbm(&mut device.pdma).await?;
            let x2_hbm = HostTensor::<bf16, HA>::from_vec(x2.clone()).to_hbm(&mut device.pdma).await?;
            let wg_hbm = HostTensor::<bf16, WG>::from_vec(perm_wg(&wg)).to_hbm(&mut device.pdma).await?;
            let wu_hbm = HostTensor::<bf16, WG>::from_vec(perm_wg(&wu)).to_hbm(&mut device.pdma).await?;
            let wd_hbm = HostTensor::<bf16, WD>::from_vec(perm_wd(&wd)).to_hbm(&mut device.pdma).await?;
            let wr_hbm = HostTensor::<bf16, WR>::from_vec(perm_wr(&wr)).to_hbm(&mut device.pdma).await?;
            let mut y = HostTensor::<bf16, HA>::zero().to_hbm(&mut device.pdma).await?;
            let mut cat = HostTensor::<f32, PR>::zero().to_hbm(&mut device.pdma).await?;
            launch(shared_router_cat, (&mut device, &h2_hbm, &x2_hbm, &wg_hbm, &wu_hbm, &wd_hbm, &wr_hbm, &mut y, &mut cat)).await?;
            let got_y: Vec<bf16> = y.to_host::<HA>(&mut device.pdma).await?.into_vec();
            let got_c: Vec<f32> = cat.to_host::<PR>(&mut device.pdma).await?.into_vec();
            let e = expert_ref(&h2, &wg, &wu, &wd);
            let exp_y: Vec<f32> = e.iter().zip(&x2).map(|(&a, &b)| r(a + f32::from(b))).collect();
            let exp_l: Vec<f32> = (0..R).map(|i| dot(&wr[i * H..(i + 1) * H], &h2)).collect();
            let ok_y = fko_check("shared_router_cat.y", &f32s(&got_y), &exp_y);
            let ycat: Vec<f32> = got_c[..H].iter().map(|&v| r(v)).collect();
            let ok_c = fko_check("shared_router_cat.cat_y", &ycat, &exp_y);
            let same = ycat.iter().zip(&got_y).filter(|(a, b)| **a == f32::from(**b)).count();
            println!("cat y rounded == y_out: {same}/{H}");
            let ok_l = fko_check("shared_router_cat.cat_logits", &got_c[H..], &exp_l);
            if bench {
                let lat = time_n!(iters, launch(shared_router_cat, (&mut device, &h2_hbm, &x2_hbm, &wg_hbm, &wu_hbm, &wd_hbm, &wr_hbm, &mut y, &mut cat)).await?);
                fko_metric(kernel, "launch_us", (3 * F * H + R * H) * 2, lat);
                let mut cat_h = HostTensor::<f32, PR>::zero();
                fko_metric("down_cat_f32[4224]", "us", (H + R) * 4, time_n!(iters, cat.to_host::<PR>(&mut device.pdma).output(&mut cat_h).await?));
            }
            ok_y && ok_c && ok_l && same == H
        }
        "combine" => {
            let mut device = Device::new(combine.topology())?;
            let y = rand_bf16(&mut rng, H, 1.0);
            let eo = rand_bf16(&mut rng, K * H, 1.0);
            let extra = rand_bf16(&mut rng, H, 1.0);
            let gates: Vec<f32> = (0..K).map(|k| 0.05 + 0.1 * k as f32 + 0.001234).collect();
            let y_hbm = HostTensor::<bf16, HA>::from_vec(y.clone()).to_hbm(&mut device.pdma).await?;
            let eo_hbm = HostTensor::<bf16, KH>::from_vec(eo.clone()).to_hbm(&mut device.pdma).await?;
            let g_hbm = HostTensor::<f32, KA>::from_vec(gates.clone()).to_hbm(&mut device.pdma).await?;
            let ex_hbm = HostTensor::<bf16, HA>::from_vec(extra.clone()).to_hbm(&mut device.pdma).await?;
            let mut xn = HostTensor::<bf16, HA>::zero().to_hbm(&mut device.pdma).await?;
            launch(combine, (&mut device, &y_hbm, &eo_hbm, &g_hbm, &ex_hbm, &mut xn)).await?;
            let got: Vec<bf16> = xn.to_host::<HA>(&mut device.pdma).await?.into_vec();
            let exp: Vec<f32> = (0..H)
                .map(|h| r(f32::from(y[h]) + f32::from(extra[h]) + (0..K).map(|k| gates[k] * f32::from(eo[k * H + h])).sum::<f32>()))
                .collect();
            let ok = fko_check(kernel, &f32s(&got), &exp);
            let exact = got.iter().zip(&exp).filter(|(g, e)| f32::from(**g) == **e).count();
            println!("combine: {exact}/{H} bit-exact vs f32-then-round oracle");
            if bench {
                let lat = time_n!(iters, launch(combine, (&mut device, &y_hbm, &eo_hbm, &g_hbm, &ex_hbm, &mut xn)).await?);
                fko_metric(kernel, "launch_us", (K + 3) * H * 2 + K * 4, lat);
            }
            ok
        }
        "lm_head" => {
            let mut device = Device::new(lm_head.topology())?;
            let t0 = Instant::now();
            let x = rand_bf16(&mut rng, H, 2.0);
            let wf = (0..H).map(|_| bf16::from_f32(0.5 + rng.random::<f32>())).collect::<Vec<_>>();
            let wlm = rand_bf16(&mut rng, W * H, 0.03);
            let x_hbm = HostTensor::<bf16, HA>::from_vec(x.clone()).to_hbm(&mut device.pdma).await?;
            let wf_hbm = HostTensor::<bf16, HA>::from_vec(wf.clone()).to_hbm(&mut device.pdma).await?;
            let w_hbm = HostTensor::<bf16, WLM>::from_vec(perm_wlm(&wlm)).to_hbm(&mut device.pdma).await?;
            let mut lg = HostTensor::<bf16, WA>::zero().to_hbm(&mut device.pdma).await?;
            eprintln!("lm_head setup {:.1}s", t0.elapsed().as_secs_f64());
            let t0 = Instant::now();
            launch(lm_head, (&mut device, &x_hbm, &wf_hbm, &w_hbm, &mut lg)).await?;
            eprintln!("lm_head first launch {:.1}s", t0.elapsed().as_secs_f64());
            let got: Vec<bf16> = lg.to_host::<WA>(&mut device.pdma).await?.into_vec();
            let hn = rmsnorm_ref(&x, &wf, 1e-5);
            let exp: Vec<f32> = (0..W).map(|i| dot(&wlm[i * H..(i + 1) * H], &hn)).collect();
            let ok = fko_check(kernel, &f32s(&got), &exp);
            let am = |v: &[f32]| v.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b });
            println!("argmax got {:?} exp {:?}", am(&f32s(&got)), am(&exp));
            if bench {
                let lat = time_n!(iters.min(50), launch(lm_head, (&mut device, &x_hbm, &wf_hbm, &w_hbm, &mut lg)).await?);
                fko_metric(kernel, "launch_us", W * H * 2, lat);
            }
            ok
        }
        "lm_head_am" => {
            let mut device = Device::new(lm_head_am.topology())?;
            let x = rand_bf16(&mut rng, H, 2.0);
            let wf = (0..H).map(|_| bf16::from_f32(0.5 + rng.random::<f32>())).collect::<Vec<_>>();
            let wlm = rand_bf16(&mut rng, W * H, 0.03);
            let x_hbm = HostTensor::<bf16, HA>::from_vec(x.clone()).to_hbm(&mut device.pdma).await?;
            let wf_hbm = HostTensor::<bf16, HA>::from_vec(wf.clone()).to_hbm(&mut device.pdma).await?;
            let w_hbm = HostTensor::<bf16, WLM>::from_vec(perm_wlm(&wlm)).to_hbm(&mut device.pdma).await?;
            let iota = HostTensor::<f32, WI>::from_vec((0..512u32).map(|j| f32::from_bits(511 - j))).to_hbm(&mut device.pdma).await?;
            let mut lg = HostTensor::<bf16, WA>::zero().to_hbm(&mut device.pdma).await?;
            let mut keys = HostTensor::<f32, WS>::zero().to_hbm(&mut device.pdma).await?;
            launch(lm_head_am, (&mut device, &x_hbm, &wf_hbm, &w_hbm, &iota, &mut lg, &mut keys)).await?;
            let got: Vec<bf16> = lg.to_host::<WA>(&mut device.pdma).await?.into_vec();
            let kv: Vec<f32> = keys.to_host::<WS>(&mut device.pdma).await?.into_vec();
            let hn = rmsnorm_ref(&x, &wf, 1e-5);
            let exp: Vec<f32> = (0..W).map(|i| dot(&wlm[i * H..(i + 1) * H], &hn)).collect();
            let ok = fko_check(kernel, &f32s(&got), &exp);
            // decode keys; compare with exact argmax of the device's own bf16 logits (first max wins)
            let (mut best, mut bv) = (0usize, f32::MIN);
            let mut seg_ok = 0;
            for (s_, &k) in kv.iter().enumerate() {
                let b = k.to_bits();
                let v = f32::from_bits(b & 0xFFFF_0000);
                let row = s_ * 512 + 511 - (b & 0x1FF) as usize;
                let seg = &got[s_ * 512..(s_ + 1) * 512];
                let (mut sr, mut sv) = (0usize, f32::MIN);
                for (j, &g) in seg.iter().enumerate() { if f32::from(g) > sv { sv = f32::from(g); sr = s_ * 512 + j; } }
                if sr == row && sv == v { seg_ok += 1; }
                if v > bv { bv = v; best = row; }
            }
            let (mut gr, mut gv) = (0usize, f32::MIN);
            for (i, &g) in got.iter().enumerate() { if f32::from(g) > gv { gv = f32::from(g); gr = i; } }
            let am_ok = seg_ok == W / 512 && best == gr;
            println!("keys: {seg_ok}/384 segments exact; argmax from keys {best} ({bv}) vs logits {gr} ({gv}) -> {}", if am_ok { "OK" } else { "MISMATCH" });
            if bench {
                let lat = time_n!(iters.min(50), launch(lm_head_am, (&mut device, &x_hbm, &wf_hbm, &w_hbm, &iota, &mut lg, &mut keys)).await?);
                fko_metric(kernel, "launch_us", W * H * 2, lat);
                let mut kh = HostTensor::<f32, WS>::zero();
                fko_metric("down_keys_f32[384]", "us", 384 * 4, time_n!(iters, keys.to_host::<WS>(&mut device.pdma).output(&mut kh).await?));
            }
            ok && am_ok
        }
        "xfer" => {
            // host<->HBM transfer costs for the per-token buffers the host would move.
            let mut device = Device::new(combine.topology())?;
            let mut h8: HbmTensor<bf16, m![1], HA> = HostTensor::<bf16, HA>::zero().to_hbm(&mut device.pdma).await?;
            let mut l128: HbmTensor<f32, m![1], RA> = HostTensor::<f32, RA>::zero().to_hbm(&mut device.pdma).await?;
            let mut g8: HbmTensor<f32, m![1], KA> = HostTensor::<f32, KA>::zero().to_hbm(&mut device.pdma).await?;
            let mut lw: HbmTensor<bf16, m![1], WA> = HostTensor::<bf16, WA>::zero().to_hbm(&mut device.pdma).await?;
            let mut kh: HbmTensor<bf16, m![1], KH> = HostTensor::<bf16, KH>::zero().to_hbm(&mut device.pdma).await?;
            let hh = HostTensor::<bf16, HA>::from_vec(rand_bf16(&mut rng, H, 1.0));
            let gh = HostTensor::<f32, KA>::from_vec(vec![0.1f32; K]);
            let mut hh_out = HostTensor::<bf16, HA>::zero();
            let mut lh_out = HostTensor::<f32, RA>::zero();
            let mut lw_out = HostTensor::<bf16, WA>::zero();
            let mut kh_out = HostTensor::<bf16, KH>::zero();
            let n = iters;
            let ks: HbmTensor<f32, m![1], WS> = HostTensor::<f32, WS>::zero().to_hbm(&mut device.pdma).await?;
            let mut ks_out = HostTensor::<f32, WS>::zero();
            fko_metric("down_f32[384]", "us", 384 * 4, time_n!(n, ks.to_host::<WS>(&mut device.pdma).output(&mut ks_out).await?));
            fko_metric("down_f32[384] fresh", "us", 384 * 4, time_n!(n, { let _v: HostTensor<f32, WS> = ks.to_host::<WS>(&mut device.pdma).await?; }));
            fko_metric("up_bf16[4096]", "us", H * 2, time_n!(n, hh.to_hbm(&mut device.pdma).output(&mut h8).await?));
            fko_metric("up_f32[8]", "us", K * 4, time_n!(n, gh.to_hbm(&mut device.pdma).output(&mut g8).await?));
            fko_metric("down_f32[128]", "us", R * 4, time_n!(n, l128.to_host::<RA>(&mut device.pdma).output(&mut lh_out).await?));
            fko_metric("down_bf16[4096]", "us", H * 2, time_n!(n, h8.to_host::<HA>(&mut device.pdma).output(&mut hh_out).await?));
            fko_metric("down_bf16[8,4096]", "us", K * H * 2, time_n!(n, kh.to_host::<KH>(&mut device.pdma).output(&mut kh_out).await?));
            fko_metric("down_bf16[196608]", "us", W * 2, time_n!(n, lw.to_host::<WA>(&mut device.pdma).output(&mut lw_out).await?));
            let _ = (&mut l128, &mut lw, &mut kh);
            true
        }
        _ => eyre::bail!("unknown kernel {kernel:?}; expected shared_router|shared_router_cat|combine|lm_head|lm_head_am|xfer"),
    };
    std::process::exit(if ok { 0 } else { 1 });
}
