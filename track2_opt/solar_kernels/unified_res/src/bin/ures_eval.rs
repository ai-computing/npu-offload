//! `ures_eval <check|bench>` — unified resident chip-A program (`unified_res_cK`:
//! mode = pos+2 -> layer step on resident weights, 0 -> ffn_k8, 1 -> prefetch) vs the separate
//! programs (`prefetch_layer`, `layer_a_res_cK`, `ffn_k8`). Plain `cargo run` = CPU emulation,
//! `cargo furiosa-opt run` = RNGD.
//!
//! Env: `K` = 1|2|4|8|16 (context chunks, default 1), `POS`, `ITERS`, `REPS` (check repeats),
//! `NO_REF` (skip the separate-program reference build/launch: bench unified only).
//!
//! check: two layers L0, L1 (different random weights), two expert tables T0, T1. One resident slot.
//!   separate: pf(L0) -> res(L0) -> ffn(T0) -> pf(L1) -> res(L1) -> ffn(T1)
//!   unified:  pf(L1) [poison] -> pf(L0) -> layer(L0) -> ffn(T0) -> layer(L0) rerun (residents intact
//!             after the ffn mode) -> pf(L1) -> layer(L1) -> ffn(T1)
//!   every output (h2, y, packed out, k/v caches, ffn out) must be bit-identical, and L1 != L0.
//! bench: launch p50 of each mode alone, and the per-layer sequence a(l) -> b(T_l) -> c(l+1),
//!   alternating L0/L1, vs the separate programs res -> ffn_k8 -> prefetch_layer.
use furiosa_opt_std::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use solar_unified_res::attn::gemv::{WKV, WO, WQ};
use solar_unified_res::attn::{D as KD, KvCache, Masks};
use solar_unified_res::axes::{H as AH, K as AK, Pk as AP, R as AR, U as AU};
use solar_unified_res::dense::shared::{WD, WG, WR};
use solar_unified_res::ffn::{WDT, WGT, ffn_k8};
use solar_unified_res::host::attn::{self as ah, D, H, N, P as KVP, Q, T};
use solar_unified_res::host::dense as dh;
use solar_unified_res::host::pack;
use solar_unified_res::layer::*;
use solar_unified_res::unified::*;
use std::time::Instant;

type VH = m![AH];
type VD = m![KD];
type KH = m![AK, AH];
type PK = m![AP, AR];
type UVH = m![AU, AH];
type UPK = m![AU, AP, AR];
type Ch = m![1];
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
        "FKO {{\"kernel\":\"{kernel}\",\"metric\":\"{metric}\",\"p50\":{p50:.2},\"p10\":{:.2},\"p90\":{:.2},\"min\":{:.2},\"n\":{n},\"bytes\":{bytes},\"gbps_p50\":{:.1}}}",
        v[n / 10],
        v[n * 9 / 10],
        v[0],
        bytes as f64 / p50 / 1e3
    );
}
macro_rules! up {
    ($dev:expr, $ty:ty, $v:expr) => {
        HostTensor::<_, $ty>::from_vec($v).to_hbm(&mut $dev.pdma).await?
    };
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

/// One layer's device tensors (weights + KV cache).
struct LayerH {
    w_in: HbmTensor<bf16, Ch, VH>,
    wq: HbmTensor<bf16, Ch, WQ>,
    wkv: HbmTensor<bf16, Ch, WKV>,
    wo: HbmTensor<bf16, Ch, WO>,
    w_post: HbmTensor<bf16, Ch, VH>,
    sg: HbmTensor<bf16, Ch, WG>,
    su: HbmTensor<bf16, Ch, WG>,
    sd: HbmTensor<bf16, Ch, WD>,
    sr: HbmTensor<bf16, Ch, WR>,
    kc: HbmTensor<bf16, Ch, KvCache>,
    vc: HbmTensor<bf16, Ch, KvCache>,
}
impl LayerH {
    fn pw(&self) -> PfW<'_> {
        PfW { wq: &self.wq, wkv: &self.wkv, wo: &self.wo, wg: &self.sg, wu: &self.su, wd: &self.sd, wr: &self.sr }
    }
}
struct Tabs {
    wg: HbmTensor<bf16, Ch, WGT>,
    wu: HbmTensor<bf16, Ch, WGT>,
    wd: HbmTensor<bf16, Ch, WDT>,
}
impl Tabs {
    fn et(&self) -> ExpertT<'_> {
        ExpertT { wg_t: &self.wg, wu_t: &self.wu, wd_t: &self.wd }
    }
}

async fn layer_h(dev: &mut Device, rng: &mut SmallRng, kc0: &[bf16], vc0: &[bf16], ctx: usize) -> eyre::Result<LayerH> {
    let wk = rnd(rng, KVP * H, -0.03, 0.03);
    let wv = rnd(rng, KVP * H, -0.03, 0.03);
    Ok(LayerH {
        w_in: up!(dev, VH, rnd(rng, H, 0.5, 1.5)),
        wq: up!(dev, WQ, ah::perm_wq(&rnd(rng, Q * H, -0.03, 0.03))),
        wkv: up!(dev, WKV, ah::perm_wkv(&wk, &wv)),
        wo: up!(dev, WO, ah::perm_wo(&rnd(rng, H * Q, -0.02, 0.02))),
        w_post: up!(dev, VH, rnd(rng, H, 0.5, 1.5)),
        sg: up!(dev, WG, dh::perm_wg(&rnd(rng, F * H, -0.03, 0.03))),
        su: up!(dev, WG, dh::perm_wg(&rnd(rng, F * H, -0.03, 0.03))),
        sd: up!(dev, WD, dh::perm_wd(&rnd(rng, H * F, -0.03, 0.03))),
        sr: up!(dev, WR, dh::perm_wr(&rnd(rng, R * H, -0.03, 0.03))),
        kc: up!(dev, KvCache, ah::to_cache_table(kc0, ctx)),
        vc: up!(dev, KvCache, ah::to_cache_table(vc0, ctx)),
    })
}
async fn tabs(dev: &mut Device, rng: &mut SmallRng) -> eyre::Result<Tabs> {
    Ok(Tabs {
        wg: up!(dev, WGT, rnd(rng, E * F * H, -0.03, 0.03)),
        wu: up!(dev, WGT, rnd(rng, E * F * H, -0.03, 0.03)),
        wd: up!(dev, WDT, rnd(rng, E * H * F, -0.03, 0.03)),
    })
}

/// Per-step inputs shared by both paths.
struct Step {
    inp: HbmTensor<f32, Ch, PK>,
    off: HbmTensor<i32, Ch, m![1]>,
    cos: HbmTensor<bf16, Ch, VD>,
    sin: HbmTensor<bf16, Ch, VD>,
    mask: HbmTensor<f32, Ch, Masks>,
    ids: HbmScalar<u64>,
    mode_l: HbmScalar<u64>,  // pos + 2
    mode_f: HbmScalar<u64>,  // 0
    mode_pf: HbmScalar<u64>, // 1
}

/// Separate-program buffers.
struct SepB {
    y: [HbmTensor<bf16, Ch, VH>; 3], // y[2] = initial y_in
    eo: HbmTensor<bf16, Ch, KH>,
    out: HbmTensor<f32, Ch, PK>,
    x: HbmTensor<bf16, Ch, VH>,
    h2: HbmTensor<bf16, Ch, VH>,
}
/// Unified-program buffers.
struct UniB {
    y: [HbmTensor<bf16, Ch, UVH>; 3],
    eo: HbmTensor<bf16, Ch, KH>,
    out: HbmTensor<f32, Ch, UPK>,
    h2: HbmTensor<bf16, Ch, UVH>,
}

macro_rules! kd {
    ($k:expr, |$fname:ident| $body:expr, $c1:ident, $c2:ident, $c4:ident, $c8:ident, $c16:ident) => {
        match $k {
            1 => { let $fname = $c1; $body }
            2 => { let $fname = $c2; $body }
            4 => { let $fname = $c4; $body }
            8 => { let $fname = $c8; $body }
            16 => { let $fname = $c16; $body }
            _ => panic!("K must be 1, 2, 4, 8 or 16"),
        }
    };
}
fn yio<'a, V: M>(y: &'a mut [HbmTensor<bf16, Ch, V>; 3], yi: usize, yo: usize) -> (&'a HbmTensor<bf16, Ch, V>, &'a mut HbmTensor<bf16, Ch, V>) {
    let [y0, y1, y2] = y;
    match (yi, yo) {
        (2, 0) => (&*y2, y0),
        (2, 1) => (&*y2, y1),
        (0, 1) => (&*y0, y1),
        (1, 0) => (&*y1, y0),
        _ => panic!(),
    }
}

/// layer_a_res_cK for layer l (residents must hold l): reads y[yi], writes y[yo].
#[allow(clippy::too_many_arguments)]
async fn sep_layer(dev: &mut Device, kk: usize, s: &Step, l: &mut LayerH, a: &AttnSlot, sh: &SharedSlot, b: &mut SepB, yi: usize, yo: usize) -> eyre::Result<()> {
    let (y_in, y_out) = yio(&mut b.y, yi, yo);
    kd!(kk, |fun| launch(fun, (
        &mut *dev, y_in, &b.eo, &s.inp, &l.w_in, &s.off, &s.cos, &s.sin, &mut l.kc, &mut l.vc, &s.mask, &l.w_post,
        a.r(), sh.r(), &mut b.out, &mut b.x, y_out, &mut b.h2,
    )).await?, layer_a_res_c1, layer_a_res_c2, layer_a_res_c4, layer_a_res_c8, layer_a_res_c16);
    Ok(())
}
async fn sep_ffn(dev: &mut Device, s: &Step, t: &Tabs, b: &mut SepB) -> eyre::Result<()> {
    launch(ffn_k8, (&mut *dev, &b.h2, &t.wg, &t.wu, &t.wd, &s.ids, &mut b.eo)).await?;
    Ok(())
}
async fn sep_pf(dev: &mut Device, l: &LayerH, a: &mut AttnSlot, sh: &mut SharedSlot) -> eyre::Result<()> {
    launch(prefetch_layer, (&mut *dev, &l.wq, &l.wkv, &l.wo, &l.sg, &l.su, &l.sd, &l.sr, a.w(), sh.w())).await?;
    Ok(())
}
/// One unified launch. `l` supplies the layer-mode tensors (w_in/w_post/caches), `p` the prefetch source.
#[allow(clippy::too_many_arguments)]
async fn uni(
    dev: &mut Device, kk: usize, s: &Step, mode: &HbmScalar<u64>, l: &mut LayerH, p: Option<&LayerH>, t: &Tabs,
    a: &mut AttnSlot, sh: &mut SharedSlot, b: &mut UniB, yi: usize, yo: usize,
) -> eyre::Result<()> {
    let (y_in, y_out) = yio(&mut b.y, yi, yo);
    // prefetch source: `p`, or (unused in the other modes) the layer's own weights
    let LayerH { w_in, wq, wkv, wo, w_post, sg, su, sd, sr, kc, vc } = l;
    let pw = match p {
        Some(p) => p.pw(),
        None => PfW { wq, wkv, wo, wg: sg, wu: su, wd: sd, wr: sr },
    };
    kd!(kk, |fun| launch(fun, (
        &mut *dev, mode, y_in, &mut b.eo, &s.inp, &*w_in, &*w_post, &s.cos, &s.sin, &mut *kc, &mut *vc, &s.mask,
        pw, a.w(), sh.w(), t.et(), &s.ids, &mut b.out, y_out, &mut b.h2,
    )).await?, unified_res_c1, unified_res_c2, unified_res_c4, unified_res_c8, unified_res_c16);
    Ok(())
}

async fn rd_vh(dev: &mut Device, t: &HbmTensor<bf16, Ch, VH>) -> eyre::Result<Vec<f32>> {
    Ok(f(&t.to_host::<VH>(&mut dev.pdma).await?.into_vec()))
}
async fn rd_uvh(dev: &mut Device, t: &HbmTensor<bf16, Ch, UVH>) -> eyre::Result<Vec<f32>> {
    Ok(f(&t.to_host::<UVH>(&mut dev.pdma).await?.into_vec())[..H].to_vec())
}
async fn rd_cache(dev: &mut Device, l: &LayerH) -> eyre::Result<(Vec<f32>, Vec<f32>)> {
    Ok((
        f(&l.kc.to_host::<KvCache>(&mut dev.pdma).await?.into_vec()),
        f(&l.vc.to_host::<KvCache>(&mut dev.pdma).await?.into_vec()),
    ))
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let mode = std::env::args().nth(1).unwrap_or("check".into());
    let bench = mode == "bench";
    let kk = env("K", 1);
    let no_ref = std::env::var("NO_REF").is_ok();
    let pos = env("POS", (kk * T).min(1300) - 1 - if kk == 1 { 211 } else { 0 });
    assert!(pos < kk * T && (kk == 1 || pos >= kk * T / 2));
    let mut rng = SmallRng::seed_from_u64(7);
    let mut dev = Device::new(unified_res_c1.topology())?;
    let t0 = Instant::now();

    let y_in = rnd(&mut rng, H, -1.0, 1.0);
    let eo = rnd(&mut rng, K * H, -0.5, 0.5);
    let extra = rnd(&mut rng, H, -0.5, 0.5);
    let gates: Vec<f32> = (0..K).map(|k| 0.05 + 0.03 * k as f32 + 0.001234).collect();
    let kc0 = rnd(&mut rng, (pos + 1) * N * D, -2.0, 2.0);
    let vc0 = rnd(&mut rng, (pos + 1) * N * D, -1.0, 1.0);
    let (cos, sin) = cos_sin(pos);
    let ids: Vec<u64> = vec![3, 7, 0, 12, 23, 5, 9, 18];
    let packed = ids.iter().enumerate().map(|(k, &e)| e << (6 * k)).sum::<u64>() + (8u64 << 56);

    let s = Step {
        inp: up!(dev, PK, pack::pack_in_bf16(&extra, &gates)),
        off: up!(dev, m![1], vec![ah::kv_offset(pos)]),
        cos: up!(dev, VD, cos.clone()),
        sin: up!(dev, VD, ah::rope_sin_signed(&sin)),
        mask: up!(dev, Masks, ah::masks(pos + 1)),
        ids: HbmScalar::<u64>::from_host(packed, &mut dev.pdma).await?,
        mode_l: HbmScalar::<u64>::from_host(mode_layer(pos), &mut dev.pdma).await?,
        mode_f: HbmScalar::<u64>::from_host(MODE_FFN, &mut dev.pdma).await?,
        mode_pf: HbmScalar::<u64>::from_host(MODE_PREFETCH, &mut dev.pdma).await?,
    };
    // separate and unified copies of each layer (same rng seed), so caches evolve identically
    let mut r0 = SmallRng::seed_from_u64(100);
    let mut r1 = SmallRng::seed_from_u64(101);
    let mut l0s = layer_h(&mut dev, &mut r0, &kc0, &vc0, pos + 1).await?;
    let mut l1s = layer_h(&mut dev, &mut r1, &kc0, &vc0, pos + 1).await?;
    let mut r0 = SmallRng::seed_from_u64(100);
    let mut r1 = SmallRng::seed_from_u64(101);
    let mut l0u = layer_h(&mut dev, &mut r0, &kc0, &vc0, pos + 1).await?;
    let mut l1u = layer_h(&mut dev, &mut r1, &kc0, &vc0, pos + 1).await?;
    let t0f = tabs(&mut dev, &mut SmallRng::seed_from_u64(200)).await?;
    let t1f = tabs(&mut dev, &mut SmallRng::seed_from_u64(201)).await?;
    let z = || vec![bf16::from_f32(0.0); H];
    let mut uy = y_in.clone();
    uy.extend(rnd(&mut rng, H, -9.0, 9.0)); // junk slack row
    let mut sb = SepB {
        y: [up!(dev, VH, z()), up!(dev, VH, z()), up!(dev, VH, y_in.clone())],
        eo: up!(dev, KH, eo.clone()),
        out: up!(dev, PK, vec![-1f32; pack::PACK]),
        x: up!(dev, VH, z()),
        h2: up!(dev, VH, z()),
    };
    let mut ub = UniB {
        y: [up!(dev, UVH, [z(), z()].concat()), up!(dev, UVH, [z(), z()].concat()), up!(dev, UVH, uy)],
        eo: up!(dev, KH, eo.clone()),
        out: up!(dev, UPK, vec![-1f32; 2 * pack::PACK]),
        h2: up!(dev, UVH, [z(), z()].concat()),
    };
    // ONE resident slot (attention 288 KiB + shared/router 62 KiB per slice), shared by both paths
    let ta = Instant::now();
    let mut aslot = AttnSlot::alloc(&dev)?;
    let mut sslot = SharedSlot::alloc(&dev)?;
    eprintln!("resident slot allocated ({:.2}s); setup {:.1}s", ta.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64());

    // ---------------- check ----------------
    let reps = if bench { 1 } else { env("REPS", 1) };
    let mut ok = true;
    for rep in 0..reps {
        let mut sep = None;
        if !no_ref {
            let t0 = Instant::now();
            // separate: pf(L0) -> res(L0) -> ffn(T0) -> pf(L1) -> res(L1) -> ffn(T1)
            sep_pf(&mut dev, &l0s, &mut aslot, &mut sslot).await?;
            sep_layer(&mut dev, kk, &s, &mut l0s, &aslot, &sslot, &mut sb, 2, 0).await?;
            let h2_0s = rd_vh(&mut dev, &sb.h2).await?;
            let out0s = sb.out.to_host::<PK>(&mut dev.pdma).await?.into_vec();
            sep_ffn(&mut dev, &s, &t0f, &mut sb).await?;
            let eo0s = f(&sb.eo.to_host::<KH>(&mut dev.pdma).await?.into_vec());
            sep_pf(&mut dev, &l1s, &mut aslot, &mut sslot).await?;
            sep_layer(&mut dev, kk, &s, &mut l1s, &aslot, &sslot, &mut sb, 0, 1).await?;
            sep_ffn(&mut dev, &s, &t1f, &mut sb).await?;
            eprintln!("separate sequence {:.1}s", t0.elapsed().as_secs_f64());
            sep = Some((h2_0s, out0s, eo0s));
        }
        let t0 = Instant::now();
        // poison the slot with L1 so the unified prefetch(L0) must really overwrite it
        uni(&mut dev, kk, &s, &s.mode_pf, &mut l0u, Some(&l1u), &t0f, &mut aslot, &mut sslot, &mut ub, 2, 0).await?;
        uni(&mut dev, kk, &s, &s.mode_pf, &mut l0u, None, &t0f, &mut aslot, &mut sslot, &mut ub, 2, 0).await?;
        let eo_l0 = ub.eo.to_host::<KH>(&mut dev.pdma).await?.into_vec(); // expert_out seen by layer(L0)
        uni(&mut dev, kk, &s, &s.mode_l, &mut l0u, None, &t0f, &mut aslot, &mut sslot, &mut ub, 2, 0).await?;
        let h2_0u = rd_uvh(&mut dev, &ub.h2).await?;
        let out0u = ub.out.to_host::<UPK>(&mut dev.pdma).await?.into_vec()[..pack::PACK].to_vec();
        let y0u = rd_uvh(&mut dev, &ub.y[0]).await?;
        uni(&mut dev, kk, &s, &s.mode_f, &mut l0u, None, &t0f, &mut aslot, &mut sslot, &mut ub, 2, 0).await?;
        let eo0u = f(&ub.eo.to_host::<KH>(&mut dev.pdma).await?.into_vec());
        // residents must survive the ffn mode: rerun L0's layer mode into y[1] (the k/v row rewrite is
        // idempotent); the layer mode reads expert_out, so restore the one the first layer(L0) saw
        let eo_hold = ub.eo.to_host::<KH>(&mut dev.pdma).await?.into_vec();
        ub.eo = up!(dev, KH, eo_l0);
        uni(&mut dev, kk, &s, &s.mode_l, &mut l0u, None, &t0f, &mut aslot, &mut sslot, &mut ub, 2, 1).await?;
        let y0u_rerun = rd_uvh(&mut dev, &ub.y[1]).await?;
        let h2_0u_rerun = rd_uvh(&mut dev, &ub.h2).await?;
        ub.eo = up!(dev, KH, eo_hold);
        uni(&mut dev, kk, &s, &s.mode_pf, &mut l0u, Some(&l1u), &t0f, &mut aslot, &mut sslot, &mut ub, 2, 0).await?;
        uni(&mut dev, kk, &s, &s.mode_l, &mut l1u, None, &t1f, &mut aslot, &mut sslot, &mut ub, 0, 1).await?;
        uni(&mut dev, kk, &s, &s.mode_f, &mut l1u, None, &t1f, &mut aslot, &mut sslot, &mut ub, 0, 1).await?;
        eprintln!("unified sequence {:.1}s", t0.elapsed().as_secs_f64());

        println!("[rep {rep}] unified internal:");
        ok &= same("L0 layer rerun after ffn mode: y", &y0u_rerun, &y0u);
        ok &= same("L0 layer rerun after ffn mode: h2", &h2_0u_rerun, &h2_0u);
        let y1u = rd_uvh(&mut dev, &ub.y[1]).await?;
        let differs = y1u.iter().zip(&y0u).any(|(a, b)| a != b);
        println!("  L1 y differs from L0 y (prefetch mode really swapped weights): {differs}");
        ok &= differs;
        if let Some((h2_0s, out0s, eo0s)) = sep {
            println!("[rep {rep}] L0 layer:");
            ok &= same("h2", &h2_0u, &h2_0s);
            ok &= same("out(h2|logits)", &out0u, &out0s);
            ok &= same("y0", &y0u, &rd_vh(&mut dev, &sb.y[0]).await?);
            let (ks, vs) = rd_cache(&mut dev, &l0s).await?;
            let (ku, vu) = rd_cache(&mut dev, &l0u).await?;
            ok &= same("k_cache L0", &ku, &ks);
            ok &= same("v_cache L0", &vu, &vs);
            println!("[rep {rep}] L0 ffn:");
            ok &= same("expert_out", &eo0u, &eo0s);
            println!("[rep {rep}] L1 layer + ffn:");
            ok &= same("h2", &rd_uvh(&mut dev, &ub.h2).await?, &rd_vh(&mut dev, &sb.h2).await?);
            ok &= same("y1", &y1u, &rd_vh(&mut dev, &sb.y[1]).await?);
            let o1s = sb.out.to_host::<PK>(&mut dev.pdma).await?.into_vec();
            let o1u = ub.out.to_host::<UPK>(&mut dev.pdma).await?.into_vec()[..pack::PACK].to_vec();
            ok &= same("out(h2|logits)", &o1u, &o1s);
            let (ks, vs) = rd_cache(&mut dev, &l1s).await?;
            let (ku, vu) = rd_cache(&mut dev, &l1u).await?;
            ok &= same("k_cache L1", &ku, &ks);
            ok &= same("v_cache L1", &vu, &vs);
            ok &= same(
                "expert_out",
                &f(&ub.eo.to_host::<KH>(&mut dev.pdma).await?.into_vec()),
                &f(&sb.eo.to_host::<KH>(&mut dev.pdma).await?.into_vec()),
            );
            let row = ah::cache_index(0, pos, 0);
            let kc_init = f(&ah::to_cache_table(&kc0, pos + 1));
            let wrote = (0..D).any(|d| ks[row + d].to_bits() != kc_init[row + d].to_bits());
            println!("  k row at pos {pos} rewritten by the kernels: {wrote}");
            ok &= wrote;
        }
    }
    println!(
        "FKO {{\"kernel\":\"unified_res_c{kk}.vs_separate\",\"check\":\"{}\",\"max_abs_err\":0,\"outliers\":0,\"n\":1}}",
        if ok { "pass" } else { "fail" }
    );

    if bench {
        let iters = env("ITERS", 100);
        let attn_b = (2 * Q + 2 * KVP) * H * 2;
        let sh_b = (3 * F * H + R * H) * 2;
        let pf_b = attn_b + sh_b;
        let ffn_b = 8 * 3 * F * H * 2;
        if !no_ref {
            let lat = time_n!(iters, sep_pf(&mut dev, &l0s, &mut aslot, &mut sslot).await?);
            fko_metric("prefetch_layer (repeated)", "launch_us", pf_b, lat);
            let lat = time_n!(iters, sep_layer(&mut dev, kk, &s, &mut l0s, &aslot, &sslot, &mut sb, 0, 1).await?);
            fko_metric(&format!("layer_a_res_c{kk} (repeated)"), "launch_us", 0, lat);
            let lat = time_n!(iters, sep_ffn(&mut dev, &s, &t0f, &mut sb).await?);
            fko_metric("ffn_k8 (repeated)", "launch_us", ffn_b, lat);
        }
        let lat = time_n!(iters, uni(&mut dev, kk, &s, &s.mode_l, &mut l0u, None, &t0f, &mut aslot, &mut sslot, &mut ub, 0, 1).await?);
        fko_metric(&format!("unified_res_c{kk} mode(a) layer (repeated)"), "launch_us", 0, lat);
        let lat = time_n!(iters, uni(&mut dev, kk, &s, &s.mode_f, &mut l0u, None, &t0f, &mut aslot, &mut sslot, &mut ub, 0, 1).await?);
        fko_metric(&format!("unified_res_c{kk} mode(b) ffn (repeated)"), "launch_us", ffn_b, lat);
        let lat = time_n!(iters, uni(&mut dev, kk, &s, &s.mode_pf, &mut l0u, None, &t0f, &mut aslot, &mut sslot, &mut ub, 0, 1).await?);
        fko_metric(&format!("unified_res_c{kk} mode(c) prefetch (repeated)"), "launch_us", pf_b, lat);
        let lat = time_n!(iters, launch(prefetch_split, (&mut dev, l0u.pw(), aslot.w(), sslot.w())).await?);
        fko_metric("prefetch_split probe (unconditional 2-tile, repeated)", "launch_us", pf_b, lat);

        for round in 0..2 {
            if !no_ref {
                // separate programs: res(l) -> ffn_k8 -> prefetch_layer(l+1)
                let (mut tl, mut tf, mut tp, mut tt) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                sep_pf(&mut dev, &l0s, &mut aslot, &mut sslot).await?;
                for i in 0..iters + 4 {
                    let (l, nx, t, yi, yo) = if i % 2 == 0 { (&mut l0s, &l1s, &t0f, 1, 0) } else { (&mut l1s, &l0s, &t1f, 0, 1) };
                    let ta = Instant::now();
                    sep_layer(&mut dev, kk, &s, l, &aslot, &sslot, &mut sb, yi, yo).await?;
                    let tb = Instant::now();
                    sep_ffn(&mut dev, &s, t, &mut sb).await?;
                    let tc = Instant::now();
                    sep_pf(&mut dev, nx, &mut aslot, &mut sslot).await?;
                    let td = Instant::now();
                    if i >= 4 {
                        tl.push((tb - ta).as_secs_f64() * 1e6);
                        tf.push((tc - tb).as_secs_f64() * 1e6);
                        tp.push((td - tc).as_secs_f64() * 1e6);
                        tt.push((td - ta).as_secs_f64() * 1e6);
                    }
                }
                fko_metric(&format!("[r{round}] seq.separate.layer_a_res_c{kk}"), "us", 0, tl);
                fko_metric(&format!("[r{round}] seq.separate.ffn_k8"), "us", ffn_b, tf);
                fko_metric(&format!("[r{round}] seq.separate.prefetch_layer"), "us", pf_b, tp);
                fko_metric(&format!("[r{round}] seq.separate.total"), "us", 0, tt);
            }
            // unified program: a(l) -> b(T_l) -> c(l+1)
            let (mut tl, mut tf, mut tp, mut tt) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            uni(&mut dev, kk, &s, &s.mode_pf, &mut l0u, None, &t0f, &mut aslot, &mut sslot, &mut ub, 1, 0).await?;
            for i in 0..iters + 4 {
                let (l, nx, t, yi, yo) = if i % 2 == 0 { (&mut l0u, &l1u, &t0f, 1, 0) } else { (&mut l1u, &l0u, &t1f, 0, 1) };
                let ta = Instant::now();
                uni(&mut dev, kk, &s, &s.mode_l, l, None, t, &mut aslot, &mut sslot, &mut ub, yi, yo).await?;
                let tb = Instant::now();
                uni(&mut dev, kk, &s, &s.mode_f, l, None, t, &mut aslot, &mut sslot, &mut ub, yi, yo).await?;
                let tc = Instant::now();
                uni(&mut dev, kk, &s, &s.mode_pf, l, Some(nx), t, &mut aslot, &mut sslot, &mut ub, yi, yo).await?;
                let td = Instant::now();
                if i >= 4 {
                    tl.push((tb - ta).as_secs_f64() * 1e6);
                    tf.push((tc - tb).as_secs_f64() * 1e6);
                    tp.push((td - tc).as_secs_f64() * 1e6);
                    tt.push((td - ta).as_secs_f64() * 1e6);
                }
            }
            fko_metric(&format!("[r{round}] seq.unified_res.a_layer"), "us", 0, tl);
            fko_metric(&format!("[r{round}] seq.unified_res.b_ffn"), "us", ffn_b, tf);
            fko_metric(&format!("[r{round}] seq.unified_res.c_prefetch"), "us", pf_b, tp);
            fko_metric(&format!("[r{round}] seq.unified_res.total"), "us", 0, tt);
        }
    }
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
