//! `res_eval <check|bench>` — resident-weight chip-A layer (`prefetch_layer` + `layer_a_res_cK`).
//! Plain `cargo run` = CPU emulation, `cargo furiosa-opt run` = RNGD.
//!
//! Env: `K` = 1|4 (context chunks, default 1), `POS`, `ITERS`, `SKIP_FFN` (CPU: skip ffn_k8),
//! `ATTN_ONLY` (only the attention weights resident; shared streamed: `layer_a_ares_cK`).
//!
//! check: two layers L0, L1 with different random weights.
//!   ref sequence (streaming, no residents allocated):
//!     layer_a_cK(L0) -> ffn_k8(h2_0) -> layer_a_cK(L1, y_in = y0, expert_out = ffn out)
//!   then allocate the resident slot (`DmTensor::alloc`) and run
//!     prefetch(L0) -> layer_a_res(L0) -> ffn_k8 -> layer_a_res(L0) again (residents must be intact
//!     after ffn_k8) -> prefetch(L1) -> layer_a_res(L1)
//!   and require every output (x, h2, y, packed out, k/v cache, ffn out) bit-identical to the ref.
//!   Plus a scalar f32 oracle on L0's outputs.
//! bench: launch p50 of prefetch / layer_a_res / layer_a (streaming) / ffn_k8 and the per-layer
//!   sequence layer_a_res(l) -> ffn_k8 -> prefetch(l+1).
use furiosa_opt_std::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use solar_layer_res::attn::gemv::{WKV, WO, WQ};
use solar_layer_res::attn::{D as KD, KvCache, Masks};
use solar_layer_res::axes::{H as AH, K as AK, Pk as AP, R as AR};
use solar_layer_res::dense::shared::{WD, WG, WR};
use solar_layer_res::ffn::{WDT, WGT, ffn_k8};
use solar_layer_res::host::attn::{self as ah, D, H, N, P as KVP, Q, T};
use solar_layer_res::host::dense as dh;
use solar_layer_res::host::pack;
use solar_layer_res::layer::*;
use std::time::Instant;

type VH = m![AH];
type VD = m![KD];
type KH = m![AK, AH];
type PK = m![AP, AR];
const F: usize = 1280;
const R: usize = 128;
const K: usize = 8;
const E: usize = 24;

fn env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}
fn rnd(rng: &mut SmallRng, n: usize, lo: f32, hi: f32) -> Vec<bf16> {
    (0..n).map(|_| bf16::from_f32(rng.random_range(lo..hi))).collect()
}
fn f(v: &[bf16]) -> Vec<f32> {
    v.iter().map(|&x| f32::from(x)).collect()
}
fn b(v: &[f32]) -> Vec<bf16> {
    v.iter().map(|&x| bf16::from_f32(x)).collect()
}
fn r(v: f32) -> f32 {
    f32::from(bf16::from_f32(v))
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
    println!(
        "FKO {{\"kernel\":\"{kernel}\",\"check\":\"{}\",\"max_abs_err\":{max_err},\"outliers\":{bad},\"n\":{}}}",
        if pass { "pass" } else { "fail" },
        got.len()
    );
    pass
}
fn same(name: &str, a: &[f32], b: &[f32]) -> bool {
    let n = a.iter().zip(b).filter(|(x, y)| x.to_bits() == y.to_bits()).count();
    let md = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let ok = n == a.len() && a.len() == b.len();
    println!("  {name}: {n}/{} bit-identical, max |diff| = {md:e}{}", a.len(), if ok { "" } else { "  <-- MISMATCH" });
    ok
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
        for _ in 0..5 {
            $body;
        }
        let mut lat = Vec::with_capacity($n);
        for _ in 0..$n {
            let t = Instant::now();
            $body;
            lat.push(t.elapsed().as_secs_f64() * 1e6);
        }
        lat
    }};
}

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

struct Wts {
    w_in: Vec<bf16>,
    wq: Vec<bf16>,
    wk: Vec<bf16>,
    wv: Vec<bf16>,
    wo: Vec<bf16>,
    w_post: Vec<bf16>,
    sg: Vec<bf16>,
    su: Vec<bf16>,
    sd: Vec<bf16>,
    wr: Vec<bf16>,
}
fn wts(rng: &mut SmallRng) -> Wts {
    Wts {
        w_in: rnd(rng, H, 0.5, 1.5),
        wq: rnd(rng, Q * H, -0.03, 0.03),
        wk: rnd(rng, KVP * H, -0.03, 0.03),
        wv: rnd(rng, KVP * H, -0.03, 0.03),
        wo: rnd(rng, H * Q, -0.02, 0.02),
        w_post: rnd(rng, H, 0.5, 1.5),
        sg: rnd(rng, F * H, -0.03, 0.03),
        su: rnd(rng, F * H, -0.03, 0.03),
        sd: rnd(rng, H * F, -0.03, 0.03),
        wr: rnd(rng, R * H, -0.03, 0.03),
    }
}

/// One layer's device tensors (weights in HBM + KV cache).
struct LayerH {
    w_in: HbmTensor<bf16, m![1], VH>,
    wq: HbmTensor<bf16, m![1], WQ>,
    wkv: HbmTensor<bf16, m![1], WKV>,
    wo: HbmTensor<bf16, m![1], WO>,
    w_post: HbmTensor<bf16, m![1], VH>,
    sg: HbmTensor<bf16, m![1], WG>,
    su: HbmTensor<bf16, m![1], WG>,
    sd: HbmTensor<bf16, m![1], WD>,
    sr: HbmTensor<bf16, m![1], WR>,
    kc: HbmTensor<bf16, m![1], KvCache>,
    vc: HbmTensor<bf16, m![1], KvCache>,
}
/// Per-launch outputs.
struct Outs {
    out: HbmTensor<f32, m![1], PK>,
    x: HbmTensor<bf16, m![1], VH>,
    y: HbmTensor<bf16, m![1], VH>,
    h2: HbmTensor<bf16, m![1], VH>,
}
struct Got {
    out: Vec<f32>,
    x: Vec<f32>,
    y: Vec<f32>,
    h2: Vec<f32>,
    kc: Vec<f32>,
    vc: Vec<f32>,
}

macro_rules! up {
    ($dev:expr, $ty:ty, $v:expr) => {
        HostTensor::<_, $ty>::from_vec($v).to_hbm(&mut $dev.pdma).await?
    };
}

async fn layer_h(dev: &mut Device, w: &Wts, kc0: &[bf16], vc0: &[bf16], ctx: usize) -> eyre::Result<LayerH> {
    Ok(LayerH {
        w_in: up!(dev, VH, w.w_in.clone()),
        wq: up!(dev, WQ, ah::perm_wq(&w.wq)),
        wkv: up!(dev, WKV, ah::perm_wkv(&w.wk, &w.wv)),
        wo: up!(dev, WO, ah::perm_wo(&w.wo)),
        w_post: up!(dev, VH, w.w_post.clone()),
        sg: up!(dev, WG, dh::perm_wg(&w.sg)),
        su: up!(dev, WG, dh::perm_wg(&w.su)),
        sd: up!(dev, WD, dh::perm_wd(&w.sd)),
        sr: up!(dev, WR, dh::perm_wr(&w.wr)),
        kc: up!(dev, KvCache, ah::to_cache_table(kc0, ctx)),
        vc: up!(dev, KvCache, ah::to_cache_table(vc0, ctx)),
    })
}
async fn outs(dev: &mut Device) -> eyre::Result<Outs> {
    Ok(Outs {
        out: up!(dev, PK, vec![-1f32; pack::PACK]),
        x: up!(dev, VH, vec![bf16::from_f32(0.0); H]),
        y: up!(dev, VH, vec![bf16::from_f32(0.0); H]),
        h2: up!(dev, VH, vec![bf16::from_f32(0.0); H]),
    })
}
async fn got(dev: &mut Device, o: &Outs, l: &LayerH) -> eyre::Result<Got> {
    Ok(Got {
        out: o.out.to_host::<PK>(&mut dev.pdma).await?.into_vec(),
        x: f(&o.x.to_host::<VH>(&mut dev.pdma).await?.into_vec()),
        y: f(&o.y.to_host::<VH>(&mut dev.pdma).await?.into_vec()),
        h2: f(&o.h2.to_host::<VH>(&mut dev.pdma).await?.into_vec()),
        kc: f(&l.kc.to_host::<KvCache>(&mut dev.pdma).await?.into_vec()),
        vc: f(&l.vc.to_host::<KvCache>(&mut dev.pdma).await?.into_vec()),
    })
}
fn same_got(tag: &str, a: &Got, b: &Got) -> bool {
    println!("{tag}:");
    let mut ok = same("x", &a.x, &b.x);
    ok &= same("h2", &a.h2, &b.h2);
    ok &= same("y", &a.y, &b.y);
    ok &= same("out(h2|logits)", &a.out, &b.out);
    ok &= same("k_cache", &a.kc, &b.kc);
    ok &= same("v_cache", &a.vc, &b.vc);
    ok
}

/// Shared per-step inputs.
struct Step {
    y_in: HbmTensor<bf16, m![1], VH>,
    eo: HbmTensor<bf16, m![1], KH>,
    inp: HbmTensor<f32, m![1], PK>,
    off: HbmTensor<i32, m![1], m![1]>,
    cos: HbmTensor<bf16, m![1], VD>,
    sin: HbmTensor<bf16, m![1], VD>,
    mask: HbmTensor<f32, m![1], Masks>,
}

// ---- launch wrappers (one per kernel; K selects the context-chunk variant) ----
macro_rules! kdispatch {
    ($k:expr, $c1:ident, $c4:ident, $c16:ident, |$fname:ident| $body:expr) => {
        match $k {
            1 => {
                let $fname = $c1;
                $body
            }
            4 => {
                let $fname = $c4;
                $body
            }
            16 => {
                let $fname = $c16;
                $body
            }
            _ => panic!("K must be 1, 4 or 16"),
        }
    };
}

async fn run_stream(dev: &mut Device, kk: usize, s: &Step, y_in: &HbmTensor<bf16, m![1], VH>, eo: &HbmTensor<bf16, m![1], KH>, l: &mut LayerH, o: &mut Outs) -> eyre::Result<()> {
    kdispatch!(kk, layer_a_c1, layer_a_c4, layer_a_c16, |fun| launch(
        fun,
        (
            &mut *dev, y_in, eo, &s.inp, &l.w_in, &l.wq, &l.wkv, &s.off, &s.cos, &s.sin, &mut l.kc, &mut l.vc, &s.mask,
            &l.wo, &l.w_post, &l.sg, &l.su, &l.sd, &l.sr, &mut o.out, &mut o.x, &mut o.y, &mut o.h2,
        ),
    )
    .await?);
    Ok(())
}
#[allow(clippy::too_many_arguments)]
async fn run_res(dev: &mut Device, kk: usize, s: &Step, y_in: &HbmTensor<bf16, m![1], VH>, eo: &HbmTensor<bf16, m![1], KH>, l: &mut LayerH, a: &AttnSlot, sh: Option<&SharedSlot>, o: &mut Outs) -> eyre::Result<()> {
    match sh {
        Some(sh) => kdispatch!(kk, layer_a_res_c1, layer_a_res_c4, layer_a_res_c16, |fun| launch(
            fun,
            (
                &mut *dev, y_in, eo, &s.inp, &l.w_in, &s.off, &s.cos, &s.sin, &mut l.kc, &mut l.vc, &s.mask, &l.w_post,
                a.r(), sh.r(), &mut o.out, &mut o.x, &mut o.y, &mut o.h2,
            ),
        )
        .await?),
        None => kdispatch!(kk, layer_a_ares_c1, layer_a_ares_c4, layer_a_ares_c16, |fun| launch(
            fun,
            (
                &mut *dev, y_in, eo, &s.inp, &l.w_in, &s.off, &s.cos, &s.sin, &mut l.kc, &mut l.vc, &s.mask, &l.w_post,
                a.r(), &l.sg, &l.su, &l.sd, &l.sr, &mut o.out, &mut o.x, &mut o.y, &mut o.h2,
            ),
        )
        .await?),
    }
    Ok(())
}
async fn run_prefetch(dev: &mut Device, l: &LayerH, a: &mut AttnSlot, sh: Option<&mut SharedSlot>) -> eyre::Result<()> {
    match sh {
        Some(sh) => launch(prefetch_layer, (&mut *dev, &l.wq, &l.wkv, &l.wo, &l.sg, &l.su, &l.sd, &l.sr, a.w(), sh.w())).await?,
        None => launch(prefetch_attn, (&mut *dev, &l.wq, &l.wkv, &l.wo, a.w())).await?,
    }
    Ok(())
}

struct Ffn {
    x: HbmTensor<bf16, m![1], VH>,
    wg: HbmTensor<bf16, m![1], WGT>,
    wu: HbmTensor<bf16, m![1], WGT>,
    wd: HbmTensor<bf16, m![1], WDT>,
    ids: HbmScalar<u64>,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let mode = std::env::args().nth(1).unwrap_or("check".into());
    let bench = mode == "bench";
    let kk = env("K", 1);
    let attn_only = std::env::var("ATTN_ONLY").is_ok();
    let skip_ffn = std::env::var("SKIP_FFN").is_ok();
    let pos = env("POS", (kk * T).min(1300) - 1 - if kk == 1 { 211 } else { 0 });
    assert!(pos < kk * T);
    let tag = format!("c{kk}{}", if attn_only { ".attn_only" } else { "" });
    let mut rng = SmallRng::seed_from_u64(7);
    let mut dev = Device::new(layer_a_res_c1.topology())?;
    let t0 = Instant::now();

    let w0 = wts(&mut rng);
    let w1 = wts(&mut rng);
    let y_in = rnd(&mut rng, H, -1.0, 1.0);
    let eo = rnd(&mut rng, K * H, -0.5, 0.5);
    let extra = rnd(&mut rng, H, -0.5, 0.5);
    let gates: Vec<f32> = (0..K).map(|k| 0.05 + 0.03 * k as f32 + 0.001234).collect();
    let kc0 = rnd(&mut rng, (pos + 1) * N * D, -2.0, 2.0);
    let vc0 = rnd(&mut rng, (pos + 1) * N * D, -1.0, 1.0);
    let (cos, sin) = cos_sin(pos);

    let s = Step {
        y_in: up!(dev, VH, y_in.clone()),
        eo: up!(dev, KH, eo.clone()),
        inp: up!(dev, PK, pack::pack_in_bf16(&extra, &gates)),
        off: up!(dev, m![1], vec![ah::kv_offset(pos)]),
        cos: up!(dev, VD, cos.clone()),
        sin: up!(dev, VD, ah::rope_sin_signed(&sin)),
        mask: up!(dev, Masks, ah::masks(pos + 1)),
    };
    // reference (streaming) and resident copies of each layer, so caches evolve identically
    let mut l0s = layer_h(&mut dev, &w0, &kc0, &vc0, pos + 1).await?;
    let mut l1s = layer_h(&mut dev, &w1, &kc0, &vc0, pos + 1).await?;
    let mut l0r = layer_h(&mut dev, &w0, &kc0, &vc0, pos + 1).await?;
    let mut l1r = layer_h(&mut dev, &w1, &kc0, &vc0, pos + 1).await?;
    let mut o0s = outs(&mut dev).await?;
    let mut o1s = outs(&mut dev).await?;
    let mut o0r = outs(&mut dev).await?;
    let mut o0r2 = outs(&mut dev).await?;
    let mut o1r = outs(&mut dev).await?;
    let ffn = if skip_ffn {
        None
    } else {
        let ids: Vec<u64> = vec![3, 7, 0, 12, 23, 5, 9, 18];
        let packed = ids.iter().enumerate().map(|(k, &e)| e << (6 * k)).sum::<u64>() + (8u64 << 56);
        Some(Ffn {
            x: up!(dev, VH, vec![bf16::from_f32(0.0); H]),
            wg: up!(dev, WGT, rnd(&mut rng, E * F * H, -0.03, 0.03)),
            wu: up!(dev, WGT, rnd(&mut rng, E * F * H, -0.03, 0.03)),
            wd: up!(dev, WDT, rnd(&mut rng, E * H * F, -0.03, 0.03)),
            ids: HbmScalar::<u64>::from_host(packed, &mut dev.pdma).await?,
        })
    };
    let mut ffo_s = up!(dev, KH, vec![bf16::from_f32(0.0); K * H]);
    let mut ffo_r = up!(dev, KH, vec![bf16::from_f32(0.0); K * H]);
    eprintln!("setup {:.1}s", t0.elapsed().as_secs_f64());

    // ---------------- reference: streaming, no residents allocated ----------------
    let t0 = Instant::now();
    run_stream(&mut dev, kk, &s, &s.y_in, &s.eo, &mut l0s, &mut o0s).await?;
    if let Some(ff) = &ffn {
        launch(ffn_k8, (&mut dev, &o0s.h2, &ff.wg, &ff.wu, &ff.wd, &ff.ids, &mut ffo_s)).await?;
    }
    let eo1_s: &HbmTensor<bf16, m![1], KH> = if ffn.is_some() { &ffo_s } else { &s.eo };
    run_stream(&mut dev, kk, &s, &o0s.y, eo1_s, &mut l1s, &mut o1s).await?;
    eprintln!("reference sequence {:.1}s", t0.elapsed().as_secs_f64());
    let g0s = got(&mut dev, &o0s, &l0s).await?;
    let g1s = got(&mut dev, &o1s, &l1s).await?;
    let ffs = f(&ffo_s.to_host::<KH>(&mut dev.pdma).await?.into_vec());

    // ---------------- resident slot ----------------
    let t0 = Instant::now();
    let mut aslot = AttnSlot::alloc(&dev)?;
    let mut sslot = if attn_only { None } else { Some(SharedSlot::alloc(&dev)?) };
    let res_bytes = 9 * 32768 + if attn_only { 0 } else { 2 * 20480 + 20480 + 2048 };
    println!("resident slot allocated: {} B/slice = {:.1} MB/chip ({:.2}s)", res_bytes, res_bytes as f64 * 512.0 / 1e6, t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    run_prefetch(&mut dev, &l0r, &mut aslot, sslot.as_mut()).await?;
    run_res(&mut dev, kk, &s, &s.y_in, &s.eo, &mut l0r, &aslot, sslot.as_ref(), &mut o0r).await?;
    if let Some(ff) = &ffn {
        launch(ffn_k8, (&mut dev, &o0r.h2, &ff.wg, &ff.wu, &ff.wd, &ff.ids, &mut ffo_r)).await?;
    }
    // residents must survive ffn_k8 (and the previous launch's scratch): rerun L0 without prefetch
    run_res(&mut dev, kk, &s, &s.y_in, &s.eo, &mut l0r, &aslot, sslot.as_ref(), &mut o0r2).await?;
    run_prefetch(&mut dev, &l1r, &mut aslot, sslot.as_mut()).await?;
    let eo1_r: &HbmTensor<bf16, m![1], KH> = if ffn.is_some() { &ffo_r } else { &s.eo };
    run_res(&mut dev, kk, &s, &o0r.y, eo1_r, &mut l1r, &aslot, sslot.as_ref(), &mut o1r).await?;
    eprintln!("resident sequence {:.1}s", t0.elapsed().as_secs_f64());
    let g0r = got(&mut dev, &o0r, &l0r).await?;
    let g0r2 = got(&mut dev, &o0r2, &l0r).await?;
    let g1r = got(&mut dev, &o1r, &l1r).await?;
    let ffr = f(&ffo_r.to_host::<KH>(&mut dev.pdma).await?.into_vec());

    let mut ok = true;
    ok &= same_got("L0 resident vs streaming", &g0r, &g0s);
    ok &= same_got("L0 resident rerun after ffn_k8 (no re-prefetch) vs streaming", &g0r2, &g0s);
    if ffn.is_some() {
        println!("ffn_k8 with residents allocated vs without:");
        ok &= same("ffn_out", &ffr, &ffs);
    }
    ok &= same_got("L1 (after prefetch(L1) over the same slot) resident vs streaming", &g1r, &g1s);
    let differs = g1r.y.iter().zip(&g0r.y).any(|(a, b)| a != b);
    println!("L1 output differs from L0 (prefetch really swapped weights): {differs}");
    ok &= differs;
    println!("FKO {{\"kernel\":\"layer_a_res_{tag}.vs_stream\",\"check\":\"{}\",\"max_abs_err\":0,\"outliers\":0,\"n\":1}}", if ok { "pass" } else { "fail" });

    // scalar oracle for L0
    {
        let w = &w0;
        let x_r: Vec<f32> = (0..H)
            .map(|h| r((0..K).map(|k| gates[k] * f32::from(eo[k * H + h])).sum::<f32>() + (f32::from(y_in[h]) + f32::from(extra[h]))))
            .collect();
        let xb = b(&x_r);
        let (q_r, k_r, v_r) = ah::attn_in_ref(&xb, &w.w_in, &w.wq, &w.wk, &w.wv, &cos, &sin);
        let (mut kr, mut vr) = (kc0.clone(), vc0.clone());
        kr[pos * N * D..(pos + 1) * N * D].copy_from_slice(&b(&k_r));
        vr[pos * N * D..(pos + 1) * N * D].copy_from_slice(&b(&v_r));
        let a = ah::attention_ref(&b(&q_r), &kr, &vr, pos + 1);
        let a: Vec<f32> = a.iter().map(|&v| r(v)).collect();
        let (x2_r, h2_r) = ah::attn_out_ref(&a, &xb, &w.wo, &w.w_post);
        let h2b = b(&h2_r);
        let e = dh::expert_ref(&h2b, &w.sg, &w.su, &w.sd);
        let y_r: Vec<f32> = e.iter().zip(&x2_r).map(|(&a, &b)| r(a + b)).collect();
        let lg_r: Vec<f32> = (0..R).map(|i| dh::dot(&w.wr[i * H..(i + 1) * H], &h2b)).collect();
        let (_, lgg) = pack::unpack_out(&g0r.out);
        ok &= fko_check(&format!("layer_a_res_{tag}.x"), &g0r.x, &x_r, 0.01, 0.01);
        ok &= fko_check(&format!("layer_a_res_{tag}.h2"), &g0r.h2, &h2_r, 0.02, 0.01);
        ok &= fko_check(&format!("layer_a_res_{tag}.y"), &g0r.y, &y_r, 0.02, 0.011);
        ok &= fko_check(&format!("layer_a_res_{tag}.logits"), &lgg, &lg_r, 0.02, 0.011);
    }

    if bench {
        let iters = env("ITERS", 100);
        let attn_b = (2 * Q + 2 * KVP) * H * 2;
        let sh_b = (3 * F * H + R * H) * 2;
        let pf_b = attn_b + if attn_only { 0 } else { sh_b };
        let lat = time_n!(iters, run_prefetch(&mut dev, &l0r, &mut aslot, sslot.as_mut()).await?);
        fko_metric(if attn_only { "prefetch_attn" } else { "prefetch_layer" }, "launch_us", pf_b, lat);
        let lat = time_n!(iters, run_res(&mut dev, kk, &s, &s.y_in, &s.eo, &mut l0r, &aslot, sslot.as_ref(), &mut o0r).await?);
        fko_metric(&format!("layer_a_{}res_c{kk}", if attn_only { "a" } else { "" }), "launch_us", if attn_only { sh_b } else { 0 }, lat);
        // the streaming kernel may not fit next to the residents (runtime admission error): report it
        match run_stream(&mut dev, kk, &s, &s.y_in, &s.eo, &mut l0s, &mut o0s).await {
            Ok(()) => {
                let lat = time_n!(iters, run_stream(&mut dev, kk, &s, &s.y_in, &s.eo, &mut l0s, &mut o0s).await?);
                fko_metric(&format!("layer_a_c{kk}(stream, residents allocated)"), "launch_us", attn_b + sh_b, lat);
            }
            Err(e) => println!("layer_a_c{kk} (streaming) cannot launch while residents are allocated: {e}"),
        }
        if let Some(ff) = &ffn {
            let lat = time_n!(iters, launch(ffn_k8, (&mut dev, &ff.x, &ff.wg, &ff.wu, &ff.wd, &ff.ids, &mut ffo_r)).await?);
            fko_metric("ffn_k8(residents allocated)", "launch_us", 8 * 3 * F * H * 2, lat);
            // per-layer sequence on the chip: res(l) -> ffn_k8 -> prefetch(l+1), alternating L0/L1
            let mut seq = Vec::new();
            let (mut t_res, mut t_ffn, mut t_pf) = (Vec::new(), Vec::new(), Vec::new());
            run_prefetch(&mut dev, &l0r, &mut aslot, sslot.as_mut()).await?;
            for i in 0..iters + 4 {
                let (l, nx) = if i % 2 == 0 { (&mut l0r, &l1r) } else { (&mut l1r, &l0r) };
                let ta = Instant::now();
                run_res(&mut dev, kk, &s, &s.y_in, &s.eo, l, &aslot, sslot.as_ref(), &mut o0r).await?;
                let tb = Instant::now();
                launch(ffn_k8, (&mut dev, &o0r.h2, &ff.wg, &ff.wu, &ff.wd, &ff.ids, &mut ffo_r)).await?;
                let tc = Instant::now();
                run_prefetch(&mut dev, nx, &mut aslot, sslot.as_mut()).await?;
                let td = Instant::now();
                if i >= 4 {
                    t_res.push((tb - ta).as_secs_f64() * 1e6);
                    t_ffn.push((tc - tb).as_secs_f64() * 1e6);
                    t_pf.push((td - tc).as_secs_f64() * 1e6);
                    seq.push((td - ta).as_secs_f64() * 1e6);
                }
            }
            fko_metric("seq.layer_a_res", "us", 0, t_res);
            fko_metric("seq.ffn_k8", "us", 0, t_ffn);
            fko_metric("seq.prefetch", "us", pf_b, t_pf);
            fko_metric("seq.total(res+ffn+prefetch)", "us", 0, seq);
            // streaming baseline sequence: layer_a(l) -> ffn_k8 (residents freed first)
            drop(aslot);
            drop(sslot);
            let mut seq = Vec::new();
            for i in 0..iters + 4 {
                let l = if i % 2 == 0 { &mut l0s } else { &mut l1s };
                let ta = Instant::now();
                run_stream(&mut dev, kk, &s, &s.y_in, &s.eo, l, &mut o0s).await?;
                launch(ffn_k8, (&mut dev, &o0s.h2, &ff.wg, &ff.wu, &ff.wd, &ff.ids, &mut ffo_s)).await?;
                if i >= 4 {
                    seq.push(ta.elapsed().as_secs_f64() * 1e6);
                }
            }
            fko_metric("seq_stream.total(layer_a+ffn)", "us", 0, seq);
        }
    }
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
