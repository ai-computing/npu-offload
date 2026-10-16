//! `fko_eval layer <check|bench>` (env `K` = 1|2|4|8|16, `POS` = current position).
//! Plain `cargo run` = CPU emulation, `cargo furiosa-opt run` = RNGD.
//!
//! check: runs the fused `layer_a_cK` and the old sequence `combine -> attn_block_cK -> shared_router`
//! on the same device with identical inputs, compares them (expect bit-exact), and compares both
//! against a scalar f32 oracle.
//! bench: launch p50 of the fused kernel, and per-layer host step (upload + launch(es) + download)
//! for fused vs old.
use furiosa_opt_std::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use solar_layer_a::attn::gemv::{WKV, WO, WQ};
use solar_layer_a::attn::{D as KD, KvCache, Masks, attn_block_c1, attn_block_c2, attn_block_c4, attn_block_c8, attn_block_c16};
use solar_layer_a::axes::{H as AH, K as AK, Pk as AP, R as AR};
use solar_layer_a::dense::combine::combine;
use solar_layer_a::dense::shared::{WD, WG, WR, shared_router};
use solar_layer_a::host::attn::{self as ah, D, H, N, P as KVP, Q, T};
use solar_layer_a::host::dense as dh;
use solar_layer_a::host::pack;
use solar_layer_a::layer::{layer_a_c1, layer_a_c2, layer_a_c4, layer_a_c8, layer_a_c16};
use std::time::Instant;

type VH = m![AH];
type VD = m![KD];
type KH = m![AK, AH];
type KA = m![AK];
type RA = m![AR];
type PK = m![AP, AR];
const F: usize = 1280;
const R: usize = 128;
const K: usize = 8;

/// Minimal `tracing` subscriber collecting on-device profile spans (`FURIOSA_OPT_PROFILE=info`).
mod prof {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing::{Event, Metadata};

    #[derive(Default, Debug, Clone)]
    pub struct Span {
        pub name: String,
        pub cluster: i64,
        pub begin: i64,
        pub end: i64,
    }
    impl Visit for Span {
        fn record_debug(&mut self, f: &Field, v: &dyn std::fmt::Debug) {
            if f.name() == "name" {
                self.name = format!("{v:?}");
            }
        }
        fn record_str(&mut self, f: &Field, v: &str) {
            if f.name() == "name" {
                self.name = v.to_string();
            }
        }
        fn record_u64(&mut self, f: &Field, v: u64) {
            self.record_i64(f, v as i64)
        }
        fn record_i64(&mut self, f: &Field, v: i64) {
            match f.name() {
                "tid" => self.cluster = v,
                "begin_cycle" => self.begin = v,
                "end_cycle" => self.end = v,
                _ => {}
            }
        }
    }
    pub struct Prof {
        pub spans: Mutex<Vec<Span>>,
        n: AtomicU64,
    }
    impl Prof {
        pub fn new() -> Self {
            Prof { spans: Mutex::new(vec![]), n: AtomicU64::new(1) }
        }
    }
    impl tracing::Subscriber for Prof {
        fn enabled(&self, m: &Metadata<'_>) -> bool {
            m.target() == "span::npu"
        }
        fn new_span(&self, a: &Attributes<'_>) -> Id {
            if a.metadata().target() == "span::npu" {
                let mut s = Span::default();
                a.record(&mut s);
                self.spans.lock().unwrap().push(s);
            }
            Id::from_u64(self.n.fetch_add(1, Ordering::Relaxed))
        }
        fn record(&self, _: &Id, _: &Record<'_>) {}
        fn record_follows_from(&self, _: &Id, _: &Id) {}
        fn event(&self, _: &Event<'_>) {}
        fn enter(&self, _: &Id) {}
        fn exit(&self, _: &Id) {}
    }
}

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
/// Fused vs old composition: count bit-identical values.
fn same(name: &str, a: &[f32], b: &[f32]) -> bool {
    let n = a.iter().zip(b).filter(|(x, y)| x.to_bits() == y.to_bits()).count();
    let md = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    println!("fused vs old {name}: {n}/{} bit-identical, max |diff| = {md:e}", a.len());
    n == a.len()
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
        for _ in 0..10 {
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

macro_rules! run_layer {
    ($fused:ident, $old:ident, $k:expr, $bench:expr, $rng:expr) => {{
        let kk: usize = $k;
        let pos = env("POS", (kk * T).min(1300) - 1 - if kk == 1 { 211 } else { 0 });
        assert!(pos < kk * T);
        let rng: &mut SmallRng = $rng;
        let mut dev = Device::new($fused.topology())?;
        let t0 = Instant::now();
        let w = Wts {
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
        };
        // layer inputs
        let y_in = rnd(rng, H, -1.0, 1.0);
        let eo = rnd(rng, K * H, -0.5, 0.5);
        let extra = rnd(rng, H, -0.5, 0.5);
        let mut gates: Vec<f32> = (0..K).map(|k| 0.05 + 0.03 * k as f32 + 0.001234).collect();
        if std::env::var("GZERO").is_ok() { gates = vec![0.0; K]; }
        let mut extra = extra;
        if std::env::var("EZERO").is_ok() { extra = vec![bf16::from_f32(0.0); H]; }
        let kc0 = rnd(rng, (pos + 1) * N * D, -2.0, 2.0);
        let vc0 = rnd(rng, (pos + 1) * N * D, -1.0, 1.0);
        let (cos, sin) = cos_sin(pos);

        macro_rules! up {
            ($ty:ty, $v:expr) => {
                HostTensor::<_, $ty>::from_vec($v).to_hbm(&mut dev.pdma).await?
            };
        }
        let w_in_h = up!(VH, w.w_in.clone());
        let wq_h = up!(WQ, ah::perm_wq(&w.wq));
        let wkv_h = up!(WKV, ah::perm_wkv(&w.wk, &w.wv));
        let wo_h = up!(WO, ah::perm_wo(&w.wo));
        let wp_h = up!(VH, w.w_post.clone());
        let sg_h = up!(WG, dh::perm_wg(&w.sg));
        let su_h = up!(WG, dh::perm_wg(&w.su));
        let sd_h = up!(WD, dh::perm_wd(&w.sd));
        let wr_h = up!(WR, dh::perm_wr(&w.wr));
        let off_h = up!(m![1], vec![ah::kv_offset(pos)]);
        let cos_h = up!(VD, cos.clone());
        let sin_h = up!(VD, ah::rope_sin_signed(&sin));
        let mask_h = up!(Masks, ah::masks(pos + 1));
        let mut kc = up!(KvCache, ah::to_cache_table(&kc0, pos + 1));
        let mut vc = up!(KvCache, ah::to_cache_table(&vc0, pos + 1));
        let mut kc_old = up!(KvCache, ah::to_cache_table(&kc0, pos + 1));
        let mut vc_old = up!(KvCache, ah::to_cache_table(&vc0, pos + 1));
        let y_h = up!(VH, y_in.clone());
        let eo_h = up!(KH, eo.clone());
        let mut inp_h = up!(PK, pack::pack_in_bf16(&extra, &gates));
        let mut x_out = up!(VH, vec![bf16::from_f32(0.0); H]);
        let mut y_out = up!(VH, vec![bf16::from_f32(0.0); H]);
        let mut out_h = up!(PK, vec![-1f32; pack::PACK]);
        // old-sequence buffers
        let mut g_h = up!(KA, gates.clone());
        let mut ex_h = up!(VH, extra.clone());
        let mut xcur = up!(VH, vec![bf16::from_f32(0.0); H]);
        let mut x2_old = up!(VH, vec![bf16::from_f32(0.0); H]);
        let mut h2_old = up!(VH, vec![bf16::from_f32(0.0); H]);
        let mut y_old = up!(VH, vec![bf16::from_f32(0.0); H]);
        let mut lg_old = up!(RA, vec![0f32; R]);
        eprintln!("setup {:.1}s", t0.elapsed().as_secs_f64());

        let t0 = Instant::now();
        launch(
            $fused,
            (
                &mut dev, &y_h, &eo_h, &inp_h, &w_in_h, &wq_h, &wkv_h, &off_h, &cos_h, &sin_h, &mut kc, &mut vc, &mask_h,
                &wo_h, &wp_h, &sg_h, &su_h, &sd_h, &wr_h, &mut out_h, &mut x_out, &mut y_out,
            ),
        )
        .await?;
        eprintln!("fused launch {:.1}s", t0.elapsed().as_secs_f64());
        let t0 = Instant::now();
        launch(combine, (&mut dev, &y_h, &eo_h, &g_h, &ex_h, &mut xcur)).await?;
        launch(
            $old,
            (
                &mut dev, &xcur, &w_in_h, &wq_h, &wkv_h, &off_h, &cos_h, &sin_h, &mut kc_old, &mut vc_old, &mask_h, &wo_h,
                &wp_h, &mut x2_old, &mut h2_old,
            ),
        )
        .await?;
        launch(shared_router, (&mut dev, &h2_old, &x2_old, &sg_h, &su_h, &sd_h, &wr_h, &mut y_old, &mut lg_old)).await?;
        eprintln!("old sequence {:.1}s", t0.elapsed().as_secs_f64());

        let xg = f(&x_out.to_host::<VH>(&mut dev.pdma).await?.into_vec());
        let yg = f(&y_out.to_host::<VH>(&mut dev.pdma).await?.into_vec());
        let og: Vec<f32> = out_h.to_host::<PK>(&mut dev.pdma).await?.into_vec();
        let (h2g, lgg) = pack::unpack_out(&og);
        let h2g = f(&h2g);
        let xo = f(&xcur.to_host::<VH>(&mut dev.pdma).await?.into_vec());
        let yo = f(&y_old.to_host::<VH>(&mut dev.pdma).await?.into_vec());
        let h2o = f(&h2_old.to_host::<VH>(&mut dev.pdma).await?.into_vec());
        let lgo: Vec<f32> = lg_old.to_host::<RA>(&mut dev.pdma).await?.into_vec();
        let kt = f(&kc.to_host::<KvCache>(&mut dev.pdma).await?.into_vec());
        let kto = f(&kc_old.to_host::<KvCache>(&mut dev.pdma).await?.into_vec());
        let vt = f(&vc.to_host::<KvCache>(&mut dev.pdma).await?.into_vec());
        let vto = f(&vc_old.to_host::<KvCache>(&mut dev.pdma).await?.into_vec());
        let name = stringify!($fused);
        let mut ok = true;
        ok &= same("x", &xg, &xo);
        ok &= same("h2", &h2g, &h2o);
        ok &= same("y", &yg, &yo);
        ok &= same("logits", &lgg, &lgo);
        ok &= same("k_cache", &kt, &kto);
        ok &= same("v_cache", &vt, &vto);
        let h2_exact = og[..H].iter().all(|&v| r(v) == v);
        println!("out[0..4]={:?} out[4096..4100]={:?} out[4224-4..]={:?} nonzero={} | h2_old[0..4]={:?} lg_old[0..4]={:?}", &og[..4], &og[H..H + 4], &og[pack::PACK - 4..], og.iter().filter(|v| **v != 0.0).count(), &h2o[..4], &lgo[..4]);
        println!("packed h2 values exactly bf16-representable: {h2_exact}");
        ok &= h2_exact;
        println!("FKO {{\"kernel\":\"{name}.vs_old\",\"check\":\"{}\",\"max_abs_err\":0,\"outliers\":0,\"n\":1}}", if ok { "pass" } else { "fail" });

        // scalar oracle
        let t0 = Instant::now();
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
        eprintln!("oracle {:.1}s", t0.elapsed().as_secs_f64());
        if std::env::var("DBG").is_ok() {
            for h in [0usize, 1, 127, 128, 2048, 4095] {
                let ge: f32 = (0..K).map(|k| gates[k] * f32::from(eo[k * H + h])).sum();
                println!("h={h} fused={} old={} ref={} y={} extra={} sum_ge={ge}", xg[h], xo[h], x_r[h], f32::from(y_in[h]), f32::from(extra[h]));
            }
        }
        ok &= fko_check("old.x", &xo, &x_r, 0.01, 0.01);
        ok &= fko_check("old.h2", &h2o, &h2_r, 0.02, 0.01);
        let k_row: Vec<f32> = (0..N * D).map(|i| kt[ah::cache_index(i / D, pos, i % D)]).collect();
        ok &= fko_check(&format!("{name}.x"), &xg, &x_r, 0.01, 0.01);
        ok &= fko_check(&format!("{name}.k"), &k_row, &k_r, 0.02, 0.01);
        ok &= fko_check(&format!("{name}.h2"), &h2g, &h2_r, 0.02, 0.01);
        ok &= fko_check(&format!("{name}.y"), &yg, &y_r, 0.02, 0.011);
        ok &= fko_check(&format!("{name}.logits"), &lgg, &lg_r, 0.02, 0.011);

        if std::env::var("FURIOSA_OPT_PROFILE").is_ok() {
            let p = std::sync::Arc::new(prof::Prof::new());
            let d = tracing::Dispatch::from(p.clone());
            let _g = tracing::dispatcher::set_default(&d);
            launch(
                $fused,
                (
                    &mut dev, &y_h, &eo_h, &inp_h, &w_in_h, &wq_h, &wkv_h, &off_h, &cos_h, &sin_h, &mut kc, &mut vc, &mask_h,
                    &wo_h, &wp_h, &sg_h, &su_h, &sd_h, &wr_h, &mut out_h, &mut x_out, &mut y_out,
                ),
            )
            .await?;
            let mut v = p.spans.lock().unwrap().clone();
            v.sort_by_key(|s| (s.begin, s.cluster));
            let path = std::env::var("PROF_OUT").unwrap_or("/tmp/prof.txt".into());
            let txt: String = v.iter().map(|s| format!("{} {} {} {}\n", s.cluster, s.begin, s.end, s.name)).collect();
            std::fs::write(&path, txt)?;
            println!("profile: {} spans -> {path}", v.len());
        }
        if $bench {
            let iters = env("ITERS", 100);
            let wbytes = (2 * Q + 2 * KVP) * H * 2 + kk * 2 * T * N * D * 2 + (3 * F * H + R * H) * 2;
            let lat = time_n!(
                iters,
                launch(
                    $fused,
                    (
                        &mut dev, &y_h, &eo_h, &inp_h, &w_in_h, &wq_h, &wkv_h, &off_h, &cos_h, &sin_h, &mut kc, &mut vc,
                        &mask_h, &wo_h, &wp_h, &sg_h, &su_h, &sd_h, &wr_h, &mut out_h, &mut x_out, &mut y_out,
                    ),
                )
                .await?
            );
            fko_metric(name, "launch_us", wbytes, lat);
            // full host step, fused: upload pack, launch, download pack
            let pk = pack::pack_in_bf16(&extra, &gates);
            let mut out_host = HostTensor::<f32, PK>::zero();
            let lat = time_n!(iters, {
                HostTensor::<f32, PK>::from_vec(pk.clone()).to_hbm(&mut dev.pdma).output(&mut inp_h).await?;
                launch(
                    $fused,
                    (
                        &mut dev, &y_h, &eo_h, &inp_h, &w_in_h, &wq_h, &wkv_h, &off_h, &cos_h, &sin_h, &mut kc, &mut vc,
                        &mask_h, &wo_h, &wp_h, &sg_h, &su_h, &sd_h, &wr_h, &mut out_h, &mut x_out, &mut y_out,
                    ),
                )
                .await?;
                out_h.to_host::<PK>(&mut dev.pdma).output(&mut out_host).await?;
            });
            fko_metric(&format!("{name}.step"), "step_us", wbytes, lat);
            // old sequence, launches only
            let lat = time_n!(iters, launch(combine, (&mut dev, &y_h, &eo_h, &g_h, &ex_h, &mut xcur)).await?);
            fko_metric("old.combine", "us", 0, lat);
            let lat = time_n!(
                iters,
                launch(
                    $old,
                    (
                        &mut dev, &xcur, &w_in_h, &wq_h, &wkv_h, &off_h, &cos_h, &sin_h, &mut kc_old, &mut vc_old, &mask_h,
                        &wo_h, &wp_h, &mut x2_old, &mut h2_old,
                    ),
                )
                .await?
            );
            fko_metric(&format!("old.{}", stringify!($old)), "us", 0, lat);
            let lat = time_n!(iters, launch(shared_router, (&mut dev, &h2_old, &x2_old, &sg_h, &su_h, &sd_h, &wr_h, &mut y_old, &mut lg_old)).await?);
            fko_metric("old.shared_router", "us", 0, lat);
            // old full host step (as solar_engine chip.rs CombineA + LayerA)
            let mut lg_host = HostTensor::<f32, RA>::zero();
            let mut h2_host = HostTensor::<bf16, VH>::zero();
            let lat = time_n!(iters, {
                HostTensor::<f32, KA>::from_vec(gates.clone()).to_hbm(&mut dev.pdma).output(&mut g_h).await?;
                HostTensor::<bf16, VH>::from_vec(extra.clone()).to_hbm(&mut dev.pdma).output(&mut ex_h).await?;
                launch(combine, (&mut dev, &y_h, &eo_h, &g_h, &ex_h, &mut xcur)).await?;
                launch(
                    $old,
                    (
                        &mut dev, &xcur, &w_in_h, &wq_h, &wkv_h, &off_h, &cos_h, &sin_h, &mut kc_old, &mut vc_old, &mask_h,
                        &wo_h, &wp_h, &mut x2_old, &mut h2_old,
                    ),
                )
                .await?;
                launch(shared_router, (&mut dev, &h2_old, &x2_old, &sg_h, &su_h, &sd_h, &wr_h, &mut y_old, &mut lg_old)).await?;
                lg_old.to_host::<RA>(&mut dev.pdma).output(&mut lg_host).await?;
                h2_old.to_host::<VH>(&mut dev.pdma).output(&mut h2_host).await?;
            });
            fko_metric("old.step", "step_us", wbytes, lat);
        }
        ok
    }};
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> eyre::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (kernel, mode) = (args.get(1).map(String::as_str).unwrap_or("layer"), args.get(2).map(String::as_str).unwrap_or("check"));
    let bench = mode == "bench";
    let mut rng = SmallRng::seed_from_u64(0x1a7e);
    let ok = match kernel {
        "layer" => match env("K", 1) {
            1 => run_layer!(layer_a_c1, attn_block_c1, 1, bench, &mut rng),
            2 => run_layer!(layer_a_c2, attn_block_c2, 2, bench, &mut rng),
            4 => run_layer!(layer_a_c4, attn_block_c4, 4, bench, &mut rng),
            8 => run_layer!(layer_a_c8, attn_block_c8, 8, bench, &mut rng),
            16 => run_layer!(layer_a_c16, attn_block_c16, 16, bench, &mut rng),
            k => eyre::bail!("K={k} not in 1|2|4|8|16"),
        },
        "probe" => {
            use solar_layer_a::attn::{Qc as AQc, RW as ARW};
            use solar_layer_a::probe::{probe_floor, probe_q};
            let mut dev = Device::new(probe_q.topology())?;
            let x = HostTensor::<bf16, VH>::from_vec(rnd(&mut rng, H, -1.0, 1.0)).to_hbm(&mut dev.pdma).await?;
            let wi = HostTensor::<bf16, VH>::from_vec(rnd(&mut rng, H, 0.5, 1.5)).to_hbm(&mut dev.pdma).await?;
            let wq = HostTensor::<bf16, WQ>::from_vec(rnd(&mut rng, Q * H, -0.03, 0.03)).to_hbm(&mut dev.pdma).await?;
            let mut qo = HostTensor::<bf16, m![AQc, ARW]>::zero().to_hbm(&mut dev.pdma).await?;
            let mut y = HostTensor::<bf16, VH>::zero().to_hbm(&mut dev.pdma).await?;
            fko_metric("probe_floor", "us", 0, time_n!(100, launch(probe_floor, (&mut dev, &x, &mut y)).await?));
            fko_metric("probe_q", "us", Q * H * 2, time_n!(100, launch(probe_q, (&mut dev, &x, &wi, &wq, &mut qo)).await?));
            {
                use solar_layer_a::probe::{QA, WQ2, probe_q2};
                let wq2 = HostTensor::<bf16, WQ2>::from_vec(rnd(&mut rng, Q * H, -0.03, 0.03)).to_hbm(&mut dev.pdma).await?;
                let mut q2 = HostTensor::<bf16, m![QA]>::zero().to_hbm(&mut dev.pdma).await?;
                fko_metric("probe_q2", "us", Q * H * 2, time_n!(100, launch(probe_q2, (&mut dev, &x, &wq2, &mut q2)).await?));
            }
            true
        }
        _ => eyre::bail!("unknown kernel {kernel:?}; expected layer"),
    };
    std::process::exit(if ok { 0 } else { 1 });
}
