//! Harness entry point: `fko_eval <attn_in|attn|attn_out|attn_block> <check|bench>`.
//! Plain `cargo run` = CPU emulation, `cargo furiosa-opt run` = RNGD.
//! Emits one `FKO {json}` line per check/metric (harness/fko.py format).
//! Env: `CTX` (attn: context length, default 701), `K` (attn_block: 1|2|4|8|16, default 1),
//! `POS` (attn_block: current position, default min(K*512, 1300)-1... see code).
use furiosa_opt_std::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use solar_attn::host::{self, CM, D, G, H, N, P, Q, T};
use solar_attn::kernel::gemv::{WKV, WO, WQ};
use solar_attn::kernel::{
    D as KD, G as KG, H as KH, KvCache, Masks, N as KN, attn_block_c1, attn_block_c2, attn_block_c4, attn_block_c8,
    attn_block_c16, attn_first, attn_in, attn_next, attn_out,
};
use std::time::Instant;

type VH = m![KH];
type VD = m![KD];
type NG = m![KN, KG];
type NGD = m![KN, KG, KD];

const WARMUP: usize = 10;
const ITERS: usize = 100;

fn rnd(rng: &mut SmallRng, n: usize, lo: f32, hi: f32) -> Vec<bf16> {
    (0..n).map(|_| bf16::from_f32(rng.random_range(lo..hi))).collect()
}
fn f(v: &[bf16]) -> Vec<f32> {
    v.iter().map(|&x| f32::from(x)).collect()
}
fn b(v: &[f32]) -> Vec<bf16> {
    v.iter().map(|&x| bf16::from_f32(x)).collect()
}
fn env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn fko_check(kernel: &str, got: &[f32], exp: &[f32], rtol: f32, atol: f32) -> bool {
    let (mut max_err, mut bad, mut finite) = (0f32, 0usize, true);
    for (&g, &e) in got.iter().zip(exp) {
        finite &= g.is_finite();
        let d = (g - e).abs();
        max_err = max_err.max(d);
        if !(d <= rtol * e.abs() + atol) {
            bad += 1;
        }
    }
    let pass = finite && bad * 100 < got.len() && got.len() == exp.len();
    let rms = (exp.iter().map(|e| e * e).sum::<f32>() / exp.len().max(1) as f32).sqrt();
    println!(
        "FKO {{\"kernel\":\"{kernel}\",\"check\":\"{}\",\"max_abs_err\":{max_err},\"outliers\":{bad},\"n\":{},\"rms_ref\":{rms}}}",
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
        for _ in 0..WARMUP {
            launch($kernel, $args).await?;
        }
        let mut lat = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t = Instant::now();
            launch($kernel, $args).await?;
            lat.push(t.elapsed().as_secs_f64() * 1e6);
        }
        lat
    }};
}

/// Plain RoPE table for a position (test data only; the real one is the yarn table).
fn cos_sin(pos: usize) -> (Vec<bf16>, Vec<bf16>) {
    let mut c = vec![bf16::from_f32(0.0); D];
    let mut s = vec![bf16::from_f32(0.0); D];
    for i in 0..D / 2 {
        let a = pos as f32 / 1.0e6f32.powf(2.0 * i as f32 / D as f32);
        c[i] = bf16::from_f32(a.cos());
        c[i + D / 2] = c[i];
        s[i] = bf16::from_f32(a.sin());
        s[i + D / 2] = s[i];
    }
    (c, s)
}

struct Weights {
    x: Vec<bf16>,
    w_in: Vec<bf16>,
    wq: Vec<bf16>,
    wk: Vec<bf16>,
    wv: Vec<bf16>,
    wo: Vec<bf16>,
    w_post: Vec<bf16>,
}

fn weights(rng: &mut SmallRng) -> Weights {
    Weights {
        x: rnd(rng, H, -1.0, 1.0),
        w_in: rnd(rng, H, 0.5, 1.5),
        wq: rnd(rng, Q * H, -0.03, 0.03),
        wk: rnd(rng, P * H, -0.03, 0.03),
        wv: rnd(rng, P * H, -0.03, 0.03),
        wo: rnd(rng, H * Q, -0.02, 0.02),
        w_post: rnd(rng, H, 0.5, 1.5),
    }
}

/// Fused-block oracle: returns (k row, x2, h2, x2 computed with the stale cache row).
fn block_ref(w: &Weights, kc: &[bf16], vc: &[bf16], pos: usize, cos: &[bf16], sin: &[bf16]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (q_r, k_r, v_r) = host::attn_in_ref(&w.x, &w.w_in, &w.wq, &w.wk, &w.wv, cos, sin);
    let (mut kr, mut vr) = (kc.to_vec(), vc.to_vec());
    kr[pos * N * D..(pos + 1) * N * D].copy_from_slice(&b(&k_r));
    vr[pos * N * D..(pos + 1) * N * D].copy_from_slice(&b(&v_r));
    let a = host::attention_ref(&b(&q_r), &kr, &vr, pos + 1);
    let a: Vec<f32> = a.iter().map(|&v| f32::from(bf16::from_f32(v))).collect();
    let (x2_r, h2_r) = host::attn_out_ref(&a, &w.x, &w.wo, &w.w_post);
    let a0 = host::attention_ref(&b(&q_r), kc, vc, pos + 1);
    let a0: Vec<f32> = a0.iter().map(|&v| f32::from(bf16::from_f32(v))).collect();
    let (x2_0, _) = host::attn_out_ref(&a0, &w.x, &w.wo, &w.w_post);
    (k_r, x2_r, h2_r, x2_0)
}

macro_rules! run_block {
    ($kern:ident, $k:expr, $bench:expr, $rng:expr) => {{
        let kk: usize = $k;
        let pos = env("POS", (kk * T).min(1300) - 1 - if kk == 1 { 211 } else { 0 });
        assert!(pos < kk * T);
        let mut device = Device::new($kern.topology())?;
        let w = weights($rng);
        // logical cache [pos + 1, N, D]; row `pos` holds stale data that the kernel must overwrite
        let kc0 = rnd($rng, (pos + 1) * N * D, -2.0, 2.0);
        let vc0 = rnd($rng, (pos + 1) * N * D, -1.0, 1.0);
        let (cos, sin) = cos_sin(pos);
        let x_h = HostTensor::<bf16, VH>::from_vec(w.x.clone()).to_hbm(&mut device.pdma).await?;
        let w_in_h = HostTensor::<bf16, VH>::from_vec(w.w_in.clone()).to_hbm(&mut device.pdma).await?;
        let wq_h = HostTensor::<bf16, WQ>::from_vec(host::perm_wq(&w.wq)).to_hbm(&mut device.pdma).await?;
        let wkv_h = HostTensor::<bf16, WKV>::from_vec(host::perm_wkv(&w.wk, &w.wv)).to_hbm(&mut device.pdma).await?;
        let wo_h = HostTensor::<bf16, WO>::from_vec(host::perm_wo(&w.wo)).to_hbm(&mut device.pdma).await?;
        let wp_h = HostTensor::<bf16, VH>::from_vec(w.w_post.clone()).to_hbm(&mut device.pdma).await?;
        let off_h = HostTensor::<i32, m![1]>::from_vec(vec![host::kv_offset(pos)]).to_hbm(&mut device.pdma).await?;
        let cos_h = HostTensor::<bf16, VD>::from_vec(cos.clone()).to_hbm(&mut device.pdma).await?;
        let sin_h = HostTensor::<bf16, VD>::from_vec(host::rope_sin_signed(&sin)).to_hbm(&mut device.pdma).await?;
        let mut kc = HostTensor::<bf16, KvCache>::from_vec(host::to_cache_table(&kc0, pos + 1)).to_hbm(&mut device.pdma).await?;
        let mut vc = HostTensor::<bf16, KvCache>::from_vec(host::to_cache_table(&vc0, pos + 1)).to_hbm(&mut device.pdma).await?;
        let mask_h = HostTensor::<f32, Masks>::from_vec(host::masks(pos + 1)).to_hbm(&mut device.pdma).await?;
        let mut x2_h = HostTensor::<bf16, VH>::zero().to_hbm(&mut device.pdma).await?;
        let mut h2_h = HostTensor::<bf16, VH>::zero().to_hbm(&mut device.pdma).await?;
        launch(
            $kern,
            (
                &mut device, &x_h, &w_in_h, &wq_h, &wkv_h, &off_h, &cos_h, &sin_h, &mut kc, &mut vc, &mask_h, &wo_h, &wp_h,
                &mut x2_h, &mut h2_h,
            ),
        )
        .await?;
        let (k_r, x2_r, h2_r, x2_0) = block_ref(&w, &kc0, &vc0, pos, &cos, &sin);
        let x2 = f(&x2_h.to_host::<VH>(&mut device.pdma).await?.into_vec());
        let h2 = f(&h2_h.to_host::<VH>(&mut device.pdma).await?.into_vec());
        let kt = f(&kc.to_host::<KvCache>(&mut device.pdma).await?.into_vec());
        let k_row: Vec<f32> = (0..N * D).map(|i| kt[host::cache_index(i / D, pos, i % D)]).collect();
        let name = stringify!($kern);
        let mut ok = fko_check(&format!("{name}.k"), &k_row, &k_r, 0.02, 0.01);
        ok &= fko_check(&format!("{name}.x2"), &x2, &x2_r, 0.02, 0.01);
        ok &= fko_check(&format!("{name}.h2"), &h2, &h2_r, 0.02, 0.01);
        let d0 = x2_0.iter().zip(&x2_r).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("{name}: ctx={} max |x2(stale row) - x2(ref)| = {d0}", pos + 1);
        if $bench {
            let lat = time_launches!(
                $kern,
                (
                    &mut device, &x_h, &w_in_h, &wq_h, &wkv_h, &off_h, &cos_h, &sin_h, &mut kc, &mut vc, &mask_h, &wo_h,
                    &wp_h, &mut x2_h, &mut h2_h,
                )
            );
            fko_metric(name, (2 * Q + 2 * P) * H * 2 + kk * 2 * T * N * D * 2, lat);
        }
        ok
    }};
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> eyre::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (kernel, mode) = (args.get(1).map(String::as_str).unwrap_or(""), args.get(2).map(String::as_str).unwrap_or("check"));
    let bench = mode == "bench";
    let mut rng = SmallRng::seed_from_u64(0x501a);
    let ok = match kernel {
        "attn_in" => {
            let mut device = Device::new(attn_in.topology())?;
            let pos = 3 * T + 37;
            let w = weights(&mut rng);
            let (cos, sin) = cos_sin(pos);
            let x_h = HostTensor::<bf16, VH>::from_vec(w.x.clone()).to_hbm(&mut device.pdma).await?;
            let w_in_h = HostTensor::<bf16, VH>::from_vec(w.w_in.clone()).to_hbm(&mut device.pdma).await?;
            let wq_h = HostTensor::<bf16, WQ>::from_vec(host::perm_wq(&w.wq)).to_hbm(&mut device.pdma).await?;
            let wkv_h = HostTensor::<bf16, WKV>::from_vec(host::perm_wkv(&w.wk, &w.wv)).to_hbm(&mut device.pdma).await?;
            let off_h = HostTensor::<i32, m![1]>::from_vec(vec![host::kv_offset(pos)]).to_hbm(&mut device.pdma).await?;
            let cos_h = HostTensor::<bf16, VD>::from_vec(cos.clone()).to_hbm(&mut device.pdma).await?;
            let sin_h = HostTensor::<bf16, VD>::from_vec(host::rope_sin_signed(&sin)).to_hbm(&mut device.pdma).await?;
            let mut kc = HostTensor::<bf16, KvCache>::zero().to_hbm(&mut device.pdma).await?;
            let mut vc = HostTensor::<bf16, KvCache>::zero().to_hbm(&mut device.pdma).await?;
            let mut q_out = HostTensor::<bf16, NGD>::zero().to_hbm(&mut device.pdma).await?;
            launch(attn_in, (&mut device, &x_h, &w_in_h, &wq_h, &wkv_h, &off_h, &cos_h, &sin_h, &mut kc, &mut vc, &mut q_out)).await?;
            let (q_r, k_r, v_r) = host::attn_in_ref(&w.x, &w.w_in, &w.wq, &w.wk, &w.wv, &cos, &sin);
            let q_g = f(&q_out.to_host::<NGD>(&mut device.pdma).await?.into_vec());
            let kt = f(&kc.to_host::<KvCache>(&mut device.pdma).await?.into_vec());
            let vt = f(&vc.to_host::<KvCache>(&mut device.pdma).await?.into_vec());
            let row = |t: &[f32]| -> Vec<f32> { (0..N * D).map(|i| t[host::cache_index(i / D, pos, i % D)]).collect() };
            let mut ok = fko_check("attn_in.q", &q_g, &q_r, 0.02, 0.01);
            ok &= fko_check("attn_in.k", &row(&kt), &k_r, 0.02, 0.01);
            ok &= fko_check("attn_in.v", &row(&vt), &v_r, 0.02, 0.01);
            let nz = kt.iter().chain(&vt).filter(|v| **v != 0.0).count();
            println!("attn_in: nonzero cache entries = {nz} (expect <= {})", 2 * N * D);
            ok &= nz <= 2 * N * D;
            if bench {
                let lat = time_launches!(attn_in, (&mut device, &x_h, &w_in_h, &wq_h, &wkv_h, &off_h, &cos_h, &sin_h, &mut kc, &mut vc, &mut q_out));
                fko_metric("attn_in", (Q + 2 * P) * H * 2, lat);
            }
            ok
        }
        "attn" => {
            // split path: attn_first on chunk 0, attn_next on chunks 1.. ; ctx from env CTX
            let ctx = env("CTX", 701);
            let nck = ctx.div_ceil(T);
            assert!(nck <= CM);
            let mut device = Device::new(attn_first.topology())?;
            let q = rnd(&mut rng, N * G * D, -1.0, 1.0);
            let k = rnd(&mut rng, ctx * N * D, -6.0, 6.0);
            let v = rnd(&mut rng, ctx * N * D, -1.0, 1.0);
            let q_h = HostTensor::<bf16, NGD>::from_vec(q.clone()).to_hbm(&mut device.pdma).await?;
            let kc = HostTensor::<bf16, KvCache>::from_vec(host::to_cache_table(&k, ctx)).to_hbm(&mut device.pdma).await?;
            let vc = HostTensor::<bf16, KvCache>::from_vec(host::to_cache_table(&v, ctx)).to_hbm(&mut device.pdma).await?;
            let mask_h = HostTensor::<f32, Masks>::from_vec(host::masks(ctx)).to_hbm(&mut device.pdma).await?;
            let mut chunks = Vec::new();
            for c in 0..nck {
                chunks.push(HbmScalar::<usize>::from_host(c, &mut device.pdma).await?);
            }
            let mut max_h = HostTensor::<f32, NG>::zero().to_hbm(&mut device.pdma).await?;
            let mut sum_h = HostTensor::<f32, NG>::zero().to_hbm(&mut device.pdma).await?;
            let mut out_h = HostTensor::<bf16, NGD>::zero().to_hbm(&mut device.pdma).await?;
            launch(attn_first, (&mut device, &q_h, &kc, &vc, &mask_h, &chunks[0], &mut max_h, &mut sum_h, &mut out_h)).await?;
            for c in 1..nck {
                launch(attn_next, (&mut device, &q_h, &kc, &vc, &mask_h, &chunks[c], &mut max_h, &mut sum_h, &mut out_h)).await?;
            }
            let exp = host::attention_ref(&q, &k, &v, ctx);
            let sums = sum_h.to_host::<NG>(&mut device.pdma).await?.into_vec();
            let out = f(&out_h.to_host::<NGD>(&mut device.pdma).await?.into_vec());
            let got: Vec<f32> = out.iter().enumerate().map(|(i, o)| o / sums[i / D]).collect();
            let ok = fko_check(&format!("attn(ctx={ctx})"), &got, &exp, 0.03, 0.005);
            if bench {
                let lat = time_launches!(attn_first, (&mut device, &q_h, &kc, &vc, &mask_h, &chunks[0], &mut max_h, &mut sum_h, &mut out_h));
                fko_metric("attn_first", 2 * T * N * D * 2, lat);
                if nck > 1 {
                    let lat = time_launches!(attn_next, (&mut device, &q_h, &kc, &vc, &mask_h, &chunks[1], &mut max_h, &mut sum_h, &mut out_h));
                    fko_metric("attn_next", 2 * T * N * D * 2, lat);
                }
            }
            ok
        }
        "attn_out" => {
            let mut device = Device::new(attn_out.topology())?;
            let a = rnd(&mut rng, Q, -3.0, 3.0);
            let sum: Vec<f32> = (0..N * G).map(|i| 1.0 + (i % 7) as f32).collect();
            let x = rnd(&mut rng, H, -1.0, 1.0);
            let wo = rnd(&mut rng, H * Q, -0.02, 0.02);
            let w_post = rnd(&mut rng, H, 0.5, 1.5);
            let a_h = HostTensor::<bf16, NGD>::from_vec(a.clone()).to_hbm(&mut device.pdma).await?;
            let s_h = HostTensor::<f32, NG>::from_vec(sum.clone()).to_hbm(&mut device.pdma).await?;
            let x_h = HostTensor::<bf16, VH>::from_vec(x.clone()).to_hbm(&mut device.pdma).await?;
            let wo_h = HostTensor::<bf16, WO>::from_vec(host::perm_wo(&wo)).to_hbm(&mut device.pdma).await?;
            let wp_h = HostTensor::<bf16, VH>::from_vec(w_post.clone()).to_hbm(&mut device.pdma).await?;
            let mut x2_h = HostTensor::<bf16, VH>::zero().to_hbm(&mut device.pdma).await?;
            let mut h2_h = HostTensor::<bf16, VH>::zero().to_hbm(&mut device.pdma).await?;
            launch(attn_out, (&mut device, &a_h, &s_h, &x_h, &wo_h, &wp_h, &mut x2_h, &mut h2_h)).await?;
            let an: Vec<f32> = a.iter().enumerate().map(|(i, &v)| f32::from(bf16::from_f32(f32::from(v) / sum[i / D]))).collect();
            let (x2_r, h2_r) = host::attn_out_ref(&an, &x, &wo, &w_post);
            let x2 = f(&x2_h.to_host::<VH>(&mut device.pdma).await?.into_vec());
            let h2 = f(&h2_h.to_host::<VH>(&mut device.pdma).await?.into_vec());
            let mut ok = fko_check("attn_out.x2", &x2, &x2_r, 0.02, 0.01);
            ok &= fko_check("attn_out.h2", &h2, &h2_r, 0.02, 0.01);
            if bench {
                let lat = time_launches!(attn_out, (&mut device, &a_h, &s_h, &x_h, &wo_h, &wp_h, &mut x2_h, &mut h2_h));
                fko_metric("attn_out", H * Q * 2, lat);
            }
            ok
        }
        "attn_block" => match env("K", 1) {
            1 => run_block!(attn_block_c1, 1, bench, &mut rng),
            2 => run_block!(attn_block_c2, 2, bench, &mut rng),
            4 => run_block!(attn_block_c4, 4, bench, &mut rng),
            8 => run_block!(attn_block_c8, 8, bench, &mut rng),
            16 => run_block!(attn_block_c16, 16, bench, &mut rng),
            k => eyre::bail!("K={k} not in 1|2|4|8|16"),
        },
        _ => eyre::bail!("unknown kernel {kernel:?}; expected attn_in|attn|attn_out|attn_block"),
    };
    std::process::exit(if ok { 0 } else { 1 });
}
