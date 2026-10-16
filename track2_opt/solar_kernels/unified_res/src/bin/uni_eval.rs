//! `uni_eval <check|bench>` — unified chip-A program (`unified_a_cK`: mode = pos+1 -> layer_a_cK,
//! mode = 0 -> ffn_k8) vs the two separate programs. Plain `cargo run` = CPU emulation,
//! `cargo furiosa-opt run` = RNGD.
//!
//! Env: `K` = 1|2|4|8|16 (context chunks, default 1), `POS`, `ITERS`, `REPS` (NPU check repeats).
//!
//! check: two layers L0, L1 (different random weights), sequence
//!   layer(L0) -> ffn(h2_0) -> layer(L1, y_in = y_0, expert_out = ffn out) -> ffn(h2_1)
//!   run once with layer_a_cK / ffn_k8 and once with unified_a_cK only; every output
//!   (h2, y, packed out, k/v caches, ffn out) must be bit-identical.
//! bench: launch p50 of each program alone, and the per-layer alternating sequence
//!   (layer -> ffn, layers L0/L1 alternating) for separate vs unified programs.
use furiosa_opt_std::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use solar_unified_a::attn::gemv::{WKV, WO, WQ};
use solar_unified_a::attn::{D as KD, KvCache, Masks};
use solar_unified_a::axes::{H as AH, K as AK, Pk as AP, R as AR, U as AU};
use solar_unified_a::dense::shared::{WD, WG, WR};
use solar_unified_a::ffn::{WDT, WGT, ffn_k8};
use solar_unified_a::host::attn::{self as ah, D, H, N, P as KVP, Q, T};
use solar_unified_a::host::dense as dh;
use solar_unified_a::host::pack;
use solar_unified_a::layer::*;
use solar_unified_a::unified::*;
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
fn fko_metric(kernel: &str, metric: &str, mut v: Vec<f64>) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    println!(
        "FKO {{\"kernel\":\"{kernel}\",\"metric\":\"{metric}\",\"p50\":{:.2},\"p10\":{:.2},\"p90\":{:.2},\"min\":{:.2},\"n\":{n}}}",
        v[n / 2],
        v[n / 10],
        v[n * 9 / 10],
        v[0]
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

/// One layer's device tensors (weights + KV cache + expert tables).
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
    mode_l: HbmScalar<u64>, // pos + 1
    mode_f: HbmScalar<u64>, // 0
}

/// Separate-program buffers (engine today).
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

/// layer_a_cK for layer l: reads y[yi], writes y[yo].
async fn sep_layer(dev: &mut Device, kk: usize, s: &Step, l: &mut LayerH, b: &mut SepB, yi: usize, yo: usize) -> eyre::Result<()> {
    let [y0, y1, y2] = &mut b.y;
    let (y_in, y_out): (&HbmTensor<bf16, Ch, VH>, &mut HbmTensor<bf16, Ch, VH>) = match (yi, yo) {
        (2, 0) => (&*y2, y0),
        (0, 1) => (&*y0, y1),
        (1, 0) => (&*y1, y0),
        _ => panic!(),
    };
    kd!(kk, |fun| launch(fun, (
        &mut *dev, y_in, &b.eo, &s.inp, &l.w_in, &l.wq, &l.wkv, &s.off, &s.cos, &s.sin, &mut l.kc, &mut l.vc, &s.mask,
        &l.wo, &l.w_post, &l.sg, &l.su, &l.sd, &l.sr, &mut b.out, &mut b.x, y_out, &mut b.h2,
    )).await?, layer_a_c1, layer_a_c2, layer_a_c4, layer_a_c8, layer_a_c16);
    Ok(())
}
async fn sep_ffn(dev: &mut Device, s: &Step, t: &Tabs, b: &mut SepB) -> eyre::Result<()> {
    launch(ffn_k8, (&mut *dev, &b.h2, &t.wg, &t.wu, &t.wd, &s.ids, &mut b.eo)).await?;
    Ok(())
}
#[allow(clippy::too_many_arguments)]
async fn uni(dev: &mut Device, kk: usize, s: &Step, mode: &HbmScalar<u64>, l: &mut LayerH, t: &Tabs, b: &mut UniB, yi: usize, yo: usize) -> eyre::Result<()> {
    let [y0, y1, y2] = &mut b.y;
    let (y_in, y_out): (&HbmTensor<bf16, Ch, UVH>, &mut HbmTensor<bf16, Ch, UVH>) = match (yi, yo) {
        (2, 0) => (&*y2, y0),
        (0, 1) => (&*y0, y1),
        (1, 0) => (&*y1, y0),
        _ => panic!(),
    };
    let aw = AttnW { w_in: &l.w_in, wq: &l.wq, wkv: &l.wkv, wo: &l.wo, w_post: &l.w_post };
    let sw = SharedW { wg: &l.sg, wu: &l.su, wd: &l.sd, wr: &l.sr };
    kd!(kk, |fun| launch(fun, (
        &mut *dev, mode, y_in, &mut b.eo, &s.inp, &s.cos, &s.sin, &mut l.kc, &mut l.vc, &s.mask,
        aw, sw, t.et(), &s.ids, &mut b.out, y_out, &mut b.h2,
    )).await?, unified_a_c1, unified_a_c2, unified_a_c4, unified_a_c8, unified_a_c16);
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
    let pos = env("POS", (kk * T).min(1300) - 1 - if kk == 1 { 211 } else { 0 });
    assert!(pos < kk * T && (kk == 1 || pos >= kk * T / 2));
    let mut rng = SmallRng::seed_from_u64(7);
    let mut dev = Device::new(unified_a_c1.topology())?;
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
        mode_l: HbmScalar::<u64>::from_host(pos as u64 + 1, &mut dev.pdma).await?,
        mode_f: HbmScalar::<u64>::from_host(0, &mut dev.pdma).await?,
    };
    // separate and unified copies of each layer (identical weights: same rng seed), so caches evolve identically
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
    eprintln!("setup {:.1}s", t0.elapsed().as_secs_f64());

    // ---------------- check ----------------
    let reps = if bench { 1 } else { env("REPS", 1) };
    let mut ok = true;
    for rep in 0..reps {
        let t0 = Instant::now();
        // separate: L0 -> ffn(T0) -> L1 -> ffn(T1)
        sep_layer(&mut dev, kk, &s, &mut l0s, &mut sb, 2, 0).await?;
        let h2_0s = rd_vh(&mut dev, &sb.h2).await?;
        let out0s = sb.out.to_host::<PK>(&mut dev.pdma).await?.into_vec();
        sep_ffn(&mut dev, &s, &t0f, &mut sb).await?;
        let eo0s = f(&sb.eo.to_host::<KH>(&mut dev.pdma).await?.into_vec());
        sep_layer(&mut dev, kk, &s, &mut l1s, &mut sb, 0, 1).await?;
        sep_ffn(&mut dev, &s, &t1f, &mut sb).await?;
        eprintln!("separate sequence {:.1}s", t0.elapsed().as_secs_f64());
        let t0 = Instant::now();
        uni(&mut dev, kk, &s, &s.mode_l, &mut l0u, &t0f, &mut ub, 2, 0).await?;
        let h2_0u = rd_uvh(&mut dev, &ub.h2).await?;
        let out0u = ub.out.to_host::<UPK>(&mut dev.pdma).await?.into_vec()[..pack::PACK].to_vec();
        uni(&mut dev, kk, &s, &s.mode_f, &mut l0u, &t0f, &mut ub, 2, 0).await?; // ffn mode (y args unused)
        let eo0u = f(&ub.eo.to_host::<KH>(&mut dev.pdma).await?.into_vec());
        uni(&mut dev, kk, &s, &s.mode_l, &mut l1u, &t1f, &mut ub, 0, 1).await?;
        uni(&mut dev, kk, &s, &s.mode_f, &mut l1u, &t1f, &mut ub, 0, 1).await?;
        eprintln!("unified sequence {:.1}s", t0.elapsed().as_secs_f64());

        println!("[rep {rep}] L0 layer:");
        ok &= same("h2", &h2_0u, &h2_0s);
        ok &= same("out(h2|logits)", &out0u, &out0s);
        ok &= same("y0", &rd_uvh(&mut dev, &ub.y[0]).await?, &rd_vh(&mut dev, &sb.y[0]).await?);
        let (ks, vs) = rd_cache(&mut dev, &l0s).await?;
        let (ku, vu) = rd_cache(&mut dev, &l0u).await?;
        ok &= same("k_cache L0", &ku, &ks);
        ok &= same("v_cache L0", &vu, &vs);
        println!("[rep {rep}] L0 ffn:");
        ok &= same("expert_out", &eo0u, &eo0s);
        println!("[rep {rep}] L1 layer + ffn:");
        ok &= same("h2", &rd_uvh(&mut dev, &ub.h2).await?, &rd_vh(&mut dev, &sb.h2).await?);
        ok &= same("y1", &rd_uvh(&mut dev, &ub.y[1]).await?, &rd_vh(&mut dev, &sb.y[1]).await?);
        let o1s = sb.out.to_host::<PK>(&mut dev.pdma).await?.into_vec();
        let o1u = ub.out.to_host::<UPK>(&mut dev.pdma).await?.into_vec()[..pack::PACK].to_vec();
        ok &= same("out(h2|logits)", &o1u, &o1s);
        let (ks, vs) = rd_cache(&mut dev, &l1s).await?;
        let (ku, vu) = rd_cache(&mut dev, &l1u).await?;
        ok &= same("k_cache L1", &ku, &ks);
        ok &= same("v_cache L1", &vu, &vs);
        ok &= same("expert_out", &f(&ub.eo.to_host::<KH>(&mut dev.pdma).await?.into_vec()), &f(&sb.eo.to_host::<KH>(&mut dev.pdma).await?.into_vec()));
        // sanity: the new k row really was written (differs from the initial random cache row)
        let row = ah::cache_index(0, pos, 0);
        let kc_init = f(&ah::to_cache_table(&kc0, pos + 1));
        let wrote = (0..D).any(|d| ks[row + d].to_bits() != kc_init[row + d].to_bits());
        println!("  k row at pos {pos} rewritten by the kernels: {wrote}");
        ok &= wrote;
    }
    println!(
        "FKO {{\"kernel\":\"unified_a_c{kk}.vs_separate\",\"check\":\"{}\",\"max_abs_err\":0,\"outliers\":0,\"n\":1}}",
        if ok { "pass" } else { "fail" }
    );

    if bench {
        let iters = env("ITERS", 100);
        let lat = time_n!(iters, sep_layer(&mut dev, kk, &s, &mut l0s, &mut sb, 0, 1).await?);
        fko_metric(&format!("layer_a_c{kk} (repeated)"), "launch_us", lat);
        let lat = time_n!(iters, sep_ffn(&mut dev, &s, &t0f, &mut sb).await?);
        fko_metric("ffn_k8 (repeated)", "launch_us", lat);
        let lat = time_n!(iters, uni(&mut dev, kk, &s, &s.mode_l, &mut l0u, &t0f, &mut ub, 0, 1).await?);
        fko_metric(&format!("unified_a_c{kk} layer mode (repeated)"), "launch_us", lat);
        let lat = time_n!(iters, uni(&mut dev, kk, &s, &s.mode_f, &mut l0u, &t0f, &mut ub, 0, 1).await?);
        fko_metric(&format!("unified_a_c{kk} ffn mode (repeated)"), "launch_us", lat);

        for round in 0..2 {
            // separate programs, alternating (engine today)
            let (mut tl, mut tf, mut tt) = (Vec::new(), Vec::new(), Vec::new());
            for i in 0..iters + 4 {
                let (l, t, yi, yo) = if i % 2 == 0 { (&mut l0s, &t0f, 1, 0) } else { (&mut l1s, &t1f, 0, 1) };
                let ta = Instant::now();
                sep_layer(&mut dev, kk, &s, l, &mut sb, yi, yo).await?;
                let tb = Instant::now();
                sep_ffn(&mut dev, &s, t, &mut sb).await?;
                let tc = Instant::now();
                if i >= 4 {
                    tl.push((tb - ta).as_secs_f64() * 1e6);
                    tf.push((tc - tb).as_secs_f64() * 1e6);
                    tt.push((tc - ta).as_secs_f64() * 1e6);
                }
            }
            fko_metric(&format!("[r{round}] seq.separate.layer_a_c{kk}"), "us", tl);
            fko_metric(&format!("[r{round}] seq.separate.ffn_k8"), "us", tf);
            fko_metric(&format!("[r{round}] seq.separate.total"), "us", tt);
            // unified program, alternating modes
            let (mut tl, mut tf, mut tt) = (Vec::new(), Vec::new(), Vec::new());
            for i in 0..iters + 4 {
                let (l, t, yi, yo) = if i % 2 == 0 { (&mut l0u, &t0f, 1, 0) } else { (&mut l1u, &t1f, 0, 1) };
                let ta = Instant::now();
                uni(&mut dev, kk, &s, &s.mode_l, l, t, &mut ub, yi, yo).await?;
                let tb = Instant::now();
                uni(&mut dev, kk, &s, &s.mode_f, l, t, &mut ub, yi, yo).await?;
                let tc = Instant::now();
                if i >= 4 {
                    tl.push((tb - ta).as_secs_f64() * 1e6);
                    tf.push((tc - tb).as_secs_f64() * 1e6);
                    tt.push((tc - ta).as_secs_f64() * 1e6);
                }
            }
            fko_metric(&format!("[r{round}] seq.unified.layer_mode"), "us", tl);
            fko_metric(&format!("[r{round}] seq.unified.ffn_mode"), "us", tf);
            fko_metric(&format!("[r{round}] seq.unified.total"), "us", tt);
        }
    }
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
