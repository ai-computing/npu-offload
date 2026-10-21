//! `eval <block|ffn|lm|all> [check|bench]`: each kernel of ${CRATE} against a host oracle (HF semantics) on random
//! weights. Plain `cargo run` = CPU emulation, `cargo furiosa-opt run` = RNGD. Prints one `CHECK {json}` line per
//! output and `BENCH {json}` per timed kernel; exits 1 when a check fails.
//! Env: `CTX` (block: context length incl. the current token; default per launch size).
use furiosa_opt_std::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use ${CRATE}::axes::{H as AH, K as AK};
use ${CRATE}::ffn::{WDT, WGT, ffn_k};
use ${CRATE}::host::{self, *};
use ${CRATE}::lm::{LOut, WLM, lm_head};
use ${CRATE}::run::{self, LayerDev, Scratch, VE, VH, tb, ub};

type KH = m![AK, AH];

fn rnd(rng: &mut SmallRng, n: usize, lo: f32, hi: f32) -> Vec<u16> {
    (0..n).map(|_| f2bf(rng.random_range(lo..hi))).collect()
}
fn fv(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| bf2f(b)).collect()
}
fn env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn check(name: &str, got: &[f32], exp: &[f32], rtol: f32, atol: f32) -> bool {
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
        "CHECK {{\"kernel\":\"{name}\",\"check\":\"{}\",\"max_abs_err\":{max_err},\"outliers\":{bad},\"n\":{},\"rms_ref\":{rms}}}",
        if pass { "pass" } else { "fail" },
        got.len()
    );
    pass
}

fn bench_line(name: &str, bytes: usize, mut v: Vec<f64>) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let p50 = v[n / 2];
    println!(
        "BENCH {{\"kernel\":\"{name}\",\"p50_us\":{p50:.2},\"p90_us\":{:.2},\"min_us\":{:.2},\"n\":{n},\"bytes\":{bytes},\"gbps_p50\":{:.1}}}",
        v[n * 9 / 10],
        v[0],
        bytes as f64 / p50 / 1e3
    );
}

struct RandW {
    w_in: Vec<u16>,
    wq: Vec<u16>,
    wk: Vec<u16>,
    wv: Vec<u16>,
    qn: Vec<u16>,
    kn: Vec<u16>,
    wo: Vec<u16>,
    w_post: Vec<u16>,
    wr: Vec<u16>,
}
impl RandW {
    fn new(rng: &mut SmallRng) -> Self {
        Self {
            w_in: rnd(rng, H, 0.5, 1.5),
            wq: rnd(rng, Q * H, -0.03, 0.03),
            wk: rnd(rng, P * H, -0.03, 0.03),
            wv: rnd(rng, P * H, -0.03, 0.03),
            qn: rnd(rng, D, 0.5, 1.5),
            kn: rnd(rng, D, 0.5, 1.5),
            wo: rnd(rng, H * Q, -0.02, 0.02),
            w_post: rnd(rng, H, 0.5, 1.5),
            wr: rnd(rng, NE * H, -0.05, 0.05),
        }
    }
    fn view(&self) -> BlockW<'_> {
        BlockW {
            w_in: &self.w_in,
            wq: &self.wq,
            wk: &self.wk,
            wv: &self.wv,
            qn: &self.qn,
            kn: &self.kn,
            wo: &self.wo,
            w_post: &self.w_post,
            wr: &self.wr,
        }
    }
}

/// logical cache [ctx, N, D] -> KvCache table
fn to_table(x: &[u16], ctx: usize) -> Vec<u16> {
    let mut o = vec![0u16; N * CM * T * D];
    for t in 0..ctx {
        for n in 0..N {
            o[cache_index(n, t, 0)..][..D].copy_from_slice(&x[(t * N + n) * D..][..D]);
        }
    }
    o
}

async fn block(rng: &mut SmallRng, bench: bool) -> eyre::Result<bool> {
    let mut ok = true;
    for &c in CHUNKS {
        let ctx = env("CTX", (c * T).min(1300) - if c == 1 { 211 } else { 0 });
        if host::chunks_for(ctx) != Some(c) {
            continue;
        }
        let pos = ctx - 1;
        let mut dev = Device::new(${CRATE}::ffn::ffn_k.topology())?;
        let w = RandW::new(rng);
        let x = rnd(rng, H, -1.0, 1.0);
        // logical cache [ctx, N, D]; row `pos` holds stale data the kernel must overwrite
        let mut kc0 = rnd(rng, ctx * N * D, -2.0, 2.0);
        let mut vc0 = rnd(rng, ctx * N * D, -1.0, 1.0);
        let mut l = LayerDev::upload(&mut dev, &w.view()).await?;
        l.kc = HostTensor::<bf16, ${CRATE}::attn::KvCache>::from_vec(tb(&to_table(&kc0, ctx))).to_hbm(&mut dev.pdma).await?;
        l.vc = HostTensor::<bf16, ${CRATE}::attn::KvCache>::from_vec(tb(&to_table(&vc0, ctx))).to_hbm(&mut dev.pdma).await?;
        let mut s = Scratch::new(&mut dev).await?;
        s.set_pos(&mut dev, pos).await?;
        HostTensor::<bf16, VH>::from_vec(tb(&x)).to_hbm(&mut dev.pdma).output(&mut s.x).await?;
        run::attn_launch(&mut dev, ctx, &mut l, &mut s).await?;

        let (cos, sin) = cos_sin(pos);
        let (q_r, k_r, v_r) = qkv_ref(&w.view(), &fv(&x), &cos, &sin);
        kc0[pos * N * D..][..N * D].copy_from_slice(&k_r.iter().map(|&v| f2bf(v)).collect::<Vec<_>>());
        vc0[pos * N * D..][..N * D].copy_from_slice(&v_r.iter().map(|&v| f2bf(v)).collect::<Vec<_>>());
        let a = attention_ref(&q_r, &kc0, &vc0, ctx);
        let (x2_r, h2_r, lg_r) = block_out_ref(&w.view(), &fv(&x), &a);

        let x2 = fv(&ub(&s.x2.to_host::<VH>(&mut dev.pdma).await?.into_vec()));
        let h2 = fv(&ub(&s.h2.to_host::<VH>(&mut dev.pdma).await?.into_vec()));
        let lg = fv(&ub(&s.lg.to_host::<VE>(&mut dev.pdma).await?.into_vec()));
        let kt = fv(&ub(&l.kc.to_host::<${CRATE}::attn::KvCache>(&mut dev.pdma).await?.into_vec()));
        let vt = fv(&ub(&l.vc.to_host::<${CRATE}::attn::KvCache>(&mut dev.pdma).await?.into_vec()));
        let row = |t: &[f32]| -> Vec<f32> { (0..N * D).map(|i| t[cache_index(i / D, pos, i % D)]).collect() };
        let name = format!("attn_block_c{c}(ctx={ctx})");
        ok &= check(&format!("{name}.k"), &row(&kt), &k_r, 0.02, 0.02);
        ok &= check(&format!("{name}.v"), &row(&vt), &v_r, 0.02, 0.01);
        ok &= check(&format!("{name}.x2"), &x2, &x2_r, 0.02, 0.01);
        ok &= check(&format!("{name}.h2"), &h2, &h2_r, 0.02, 0.02);
        ok &= check(&format!("{name}.logits"), &lg, &lg_r, 0.03, 0.02);
        if bench {
            let mut lat = Vec::new();
            for i in 0..60 {
                let t = Instant::now();
                run::attn_launch(&mut dev, ctx, &mut l, &mut s).await?;
                if i >= 10 {
                    lat.push(t.elapsed().as_secs_f64() * 1e6);
                }
            }
            bench_line(&format!("attn_block_c{c}"), (Q + 2 * P + Q) * H * 2 + NE * H * 2 + c * 2 * T * N * D * 2, lat);
        }
    }
    Ok(ok)
}

async fn ffn(rng: &mut SmallRng, bench: bool) -> eyre::Result<bool> {
    let mut dev = Device::new(ffn_k.topology())?;
    // table of E_T random experts (device layouts) + HF copies of the K used ones
    let slots: Vec<usize> = (0..K).map(|k| (k * 7 + 3) % E_T).collect();
    let (mut tg, mut tu, mut td) = (Vec::new(), Vec::new(), Vec::new());
    let mut used = Vec::new();
    for e in 0..E_T {
        let g = rnd(rng, F * H, -0.03, 0.03);
        let u = rnd(rng, F * H, -0.03, 0.03);
        let d = rnd(rng, H * F, -0.03, 0.03);
        tg.extend(perm_wg(&g));
        tu.extend(perm_wg(&u));
        td.extend(perm_wd(&d));
        if slots.contains(&e) {
            used.push((e, g, u, d));
        }
    }
    let x = rnd(rng, H, -1.0, 1.0);
    let g_h = HostTensor::<bf16, WGT>::from_vec(tb(&tg)).to_hbm(&mut dev.pdma).await?;
    let u_h = HostTensor::<bf16, WGT>::from_vec(tb(&tu)).to_hbm(&mut dev.pdma).await?;
    let d_h = HostTensor::<bf16, WDT>::from_vec(tb(&td)).to_hbm(&mut dev.pdma).await?;
    let x_h = HostTensor::<bf16, VH>::from_vec(tb(&x)).to_hbm(&mut dev.pdma).await?;
    let ids = HbmScalar::<u64>::from_host(pack_slots(&slots), &mut dev.pdma).await?;
    let mut out = HostTensor::<bf16, KH>::zero().to_hbm(&mut dev.pdma).await?;
    launch(ffn_k, (&mut dev, &x_h, &g_h, &u_h, &d_h, &ids, &mut out)).await?;
    let got = fv(&ub(&out.to_host::<KH>(&mut dev.pdma).await?.into_vec()));
    let mut ok = true;
    for (k, &s) in slots.iter().enumerate() {
        let (_, g, u, d) = used.iter().find(|t| t.0 == s).unwrap();
        let exp = expert_ref(g, u, d, &fv(&x));
        ok &= check(&format!("ffn_k.slot{s}"), &got[k * H..(k + 1) * H], &exp, 0.03, 0.01);
    }
    if bench {
        let mut lat = Vec::new();
        for i in 0..60 {
            let t = Instant::now();
            launch(ffn_k, (&mut dev, &x_h, &g_h, &u_h, &d_h, &ids, &mut out)).await?;
            if i >= 10 {
                lat.push(t.elapsed().as_secs_f64() * 1e6);
            }
        }
        bench_line("ffn_k8", K * 3 * F * H * 2, lat);
    }
    Ok(ok)
}

async fn lm(rng: &mut SmallRng, bench: bool) -> eyre::Result<bool> {
    let mut dev = Device::new(lm_head.topology())?;
    let w = rnd(rng, VOCAB * H, -0.03, 0.03);
    let x = rnd(rng, H, -1.0, 1.0);
    let w_h = HostTensor::<bf16, WLM>::from_vec(tb(&perm_lm(&w))).to_hbm(&mut dev.pdma).await?;
    let x_h = HostTensor::<bf16, VH>::from_vec(tb(&x)).to_hbm(&mut dev.pdma).await?;
    let mut out = HostTensor::<bf16, LOut>::zero().to_hbm(&mut dev.pdma).await?;
    launch(lm_head, (&mut dev, &x_h, &w_h, &mut out)).await?;
    let got = fv(&ub(&out.to_host::<LOut>(&mut dev.pdma).await?.into_vec()));
    let xf = fv(&x);
    // sampled rows (full oracle is slow on the emulator host)
    let rows: Vec<usize> = (0..4096).map(|i| (i * 7919 + 13) % VOCAB).chain(VOCAB - 8..VOCAB).collect();
    let exp: Vec<f32> = rows.iter().map(|&i| r(w[i * H..(i + 1) * H].iter().zip(&xf).map(|(&a, &b)| bf2f(a) * b).sum())).collect();
    let g: Vec<f32> = rows.iter().map(|&i| got[i]).collect();
    let ok = check("lm_head", &g, &exp, 0.02, 0.01);
    if bench {
        let mut lat = Vec::new();
        for i in 0..30 {
            let t = Instant::now();
            launch(lm_head, (&mut dev, &x_h, &w_h, &mut out)).await?;
            if i >= 5 {
                lat.push(t.elapsed().as_secs_f64() * 1e6);
            }
        }
        bench_line("lm_head", LNC * LF * H * 2, lat);
    }
    Ok(ok)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> eyre::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let what = args.get(1).map(String::as_str).unwrap_or("all");
    let bench = args.get(2).map(String::as_str) == Some("bench");
    let mut rng = SmallRng::seed_from_u64(0x0e7a);
    let mut ok = true;
    if what == "block" || what == "all" {
        ok &= block(&mut rng, bench).await?;
    }
    if what == "ffn" || what == "all" {
        ok &= ffn(&mut rng, bench).await?;
    }
    if what == "lm" || what == "all" {
        ok &= lm(&mut rng, bench).await?;
    }
    println!("RESULT {}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
