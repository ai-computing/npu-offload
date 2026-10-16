//! `eval check [n...]`  : CPU (cargo run) or NPU (cargo furiosa-opt run) correctness of experts_b
//!                        vs scalar oracle and vs the old ffn_k8 + combine sequence (bitwise).
//! `eval bench`         : NPU only. Old per-layer sequence vs fused, swap primitive, overlap test.
use furiosa_opt_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use solar_experts_b::axes::{H as AH, K as AK};
use solar_experts_b::combine::combine;
use solar_experts_b::experts_b::{experts_b, swap_slot};
use solar_experts_b::ffn32::{WD, WDT, WG, WGT, ffn_k8};
use std::time::Instant;

#[path = "../host_util.rs"]
#[allow(dead_code)]
mod host_util;
use host_util::*;

type HA = m![AH];
type KA = m![AK];
type KH = m![AK, AH];

const E2: usize = 32;
const SLOT_ELEMS: usize = F * H;
const IDS: [usize; 8] = [5, 17, 0, 31, 9, 22, 3, 12];

fn pack(slots: &[usize]) -> u64 {
    let mut p = (slots.len() as u64) << 56;
    for (k, &s) in slots.iter().enumerate() {
        p += (s as u64) << (6 * k);
    }
    p
}
fn r(v: f32) -> f32 { f32::from(bf16::from_f32(v)) }

struct Model {
    // row-major per expert, only for slots in IDS (others zero)
    g: Vec<Vec<bf16>>,
    u: Vec<Vec<bf16>>,
    d: Vec<Vec<bf16>>,
}

fn build(rng: &mut SmallRng) -> (Model, Vec<bf16>, Vec<bf16>, Vec<bf16>) {
    let mut m = Model { g: vec![vec![]; E2], u: vec![vec![]; E2], d: vec![vec![]; E2] };
    let (mut tg, mut tu, mut td) = (Vec::with_capacity(E2 * SLOT_ELEMS), Vec::with_capacity(E2 * SLOT_ELEMS), Vec::with_capacity(E2 * SLOT_ELEMS));
    for e in 0..E2 {
        if IDS.contains(&e) {
            m.g[e] = rand_bf16(rng, F * H, 0.03);
            m.u[e] = rand_bf16(rng, F * H, 0.03);
            m.d[e] = rand_bf16(rng, H * F, 0.03);
            tg.extend(perm_wg(&m.g[e]));
            tu.extend(perm_wg(&m.u[e]));
            td.extend(perm_wd(&m.d[e]));
        } else {
            tg.extend(std::iter::repeat_n(bf16::from_f32(0.0), SLOT_ELEMS));
            tu.extend(std::iter::repeat_n(bf16::from_f32(0.0), SLOT_ELEMS));
            td.extend(std::iter::repeat_n(bf16::from_f32(0.0), SLOT_ELEMS));
        }
    }
    (m, tg, tu, td)
}

fn gates_for(n: usize) -> [f32; 8] {
    let mut g = [0f32; 8];
    for k in 0..n {
        g[k] = 0.31 - 0.027 * k as f32 + 0.000123;
    }
    g
}

fn oracle(m: &Model, x: &[bf16], slots: &[usize], gates: &[f32; 8]) -> Vec<f32> {
    let mut acc = vec![0f32; H];
    for (k, &e) in slots.iter().enumerate() {
        let y = expert_ref(x, &m.g[e], &m.u[e], &m.d[e]);
        for h in 0..H {
            acc[h] += gates[k] * r(y[h]);
        }
    }
    acc.iter().map(|&v| r(v)).collect()
}

fn pct(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[v.len() * 9 / 10], v[0])
}

#[cfg(backend = "npu")]
mod npu {
    use super::*;
    use furiosa_opt_rt::Buffer;
    use furiosa_opt_std::runtime::{Buffers, DeviceSend};
    use solar_experts_b::axes::{Chip, PK as APK};

    pub type PA = m![APK];

    /// The device allocation behind a placed HBM tensor (or the window of a view).
    pub fn buf_of<T: DeviceSend>(t: &T) -> Buffer {
        let mut b = Buffers::new();
        t.bind(&mut b).expect("unplaced");
        b[0].clone()
    }

    /// Packed per-layer upload: bytes [0, 2H) = x bf16[H], [2H, 2H+32) = gates f32[8].
    pub fn pack_bytes(x: &[bf16], gates: &[f32; 8]) -> Vec<u8> {
        let mut v = Vec::with_capacity(2 * H + 32);
        for b in x { v.extend_from_slice(&b.to_bits().to_le_bytes()); }
        for g in gates { v.extend_from_slice(&g.to_le_bytes()); }
        v
    }

    /// One allocation; x / gates are aliasing HbmTensors over it.
    pub struct Packed {
        pub all: HbmTensor<bf16, Chip, PA>,
        pub x: HbmTensor<bf16, Chip, HA>,
        pub gates: HbmTensor<f32, Chip, KA>,
    }
    pub async fn packed(dev: &mut Device) -> eyre::Result<Packed> {
        let all = HostTensor::<bf16, PA>::zero().to_hbm(&mut dev.pdma).await?;
        let b = buf_of(&all);
        let x: HbmTensor<bf16, Chip, HA> = b.slice(0..2 * H).into();
        let gates: HbmTensor<f32, Chip, KA> = b.slice(2 * H..2 * H + 32).into();
        Ok(Packed { all, x, gates })
    }

    /// HbmTensor aliasing table slot `e` (one expert, WG layout).
    pub fn slot_wg(t: &HbmTensor<bf16, Chip, WGT>, e: usize) -> HbmTensor<bf16, Chip, WG> {
        buf_of(t).slice(e * SLOT_ELEMS * 2..(e + 1) * SLOT_ELEMS * 2).into()
    }
    pub fn slot_wd(t: &HbmTensor<bf16, Chip, WDT>, e: usize) -> HbmTensor<bf16, Chip, WD> {
        buf_of(t).slice(e * SLOT_ELEMS * 2..(e + 1) * SLOT_ELEMS * 2).into()
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> eyre::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("check");
    let ns: Vec<usize> = if args.len() > 2 { args[2..].iter().map(|s| s.parse().unwrap()).collect() } else { vec![1, 2, 4, 8] };
    let mut rng = SmallRng::seed_from_u64(11);
    let t0 = Instant::now();
    let (m, tg, tu, td) = build(&mut rng);
    let x = rand_bf16(&mut rng, H, 1.0);
    let mut dev = Device::new(experts_b.topology())?;
    let wg_t = HostTensor::<bf16, WGT>::from_vec(tg).to_hbm(&mut dev.pdma).await?;
    let wu_t = HostTensor::<bf16, WGT>::from_vec(tu).to_hbm(&mut dev.pdma).await?;
    let wd_t = HostTensor::<bf16, WDT>::from_vec(td).to_hbm(&mut dev.pdma).await?;
    let zeros = HostTensor::<bf16, KH>::zero().to_hbm(&mut dev.pdma).await?;
    let zh = HostTensor::<bf16, HA>::zero().to_hbm(&mut dev.pdma).await?;
    let x_hbm = HostTensor::<bf16, HA>::from_vec(x.clone()).to_hbm(&mut dev.pdma).await?;
    let mut ids = HbmScalar::<u64>::from_host(0, &mut dev.pdma).await?;
    let mut g_hbm = HostTensor::<f32, KA>::zero().to_hbm(&mut dev.pdma).await?;
    let mut out = HostTensor::<bf16, HA>::zero().to_hbm(&mut dev.pdma).await?;
    let mut rows = HostTensor::<bf16, KH>::zero().to_hbm(&mut dev.pdma).await?;
    let mut xn = HostTensor::<bf16, HA>::zero().to_hbm(&mut dev.pdma).await?;
    eprintln!("setup {:.1}s", t0.elapsed().as_secs_f64());

    let mut all_ok = true;
    // ---- correctness: fused vs oracle and vs old sequence (bitwise)
    for &n in &ns {
        let slots = &IDS[..n];
        let gates = gates_for(n);
        ids.write(pack(slots), &mut dev.pdma).await?;
        HostTensor::<f32, KA>::from_vec(gates.to_vec()).to_hbm(&mut dev.pdma).output(&mut g_hbm).await?;
        let t = Instant::now();
        launch(experts_b, (&mut dev, &x_hbm, &g_hbm, &wg_t, &wu_t, &wd_t, &ids, &zeros, &mut out)).await?;
        let tf = t.elapsed().as_secs_f64();
        let got: Vec<bf16> = out.to_host::<HA>(&mut dev.pdma).await?.into_vec();
        launch(ffn_k8, (&mut dev, &x_hbm, &wg_t, &wu_t, &wd_t, &ids, &mut rows)).await?;
        launch(combine, (&mut dev, &zh, &rows, &g_hbm, &zh, &mut xn)).await?;
        let old: Vec<bf16> = xn.to_host::<HA>(&mut dev.pdma).await?.into_vec();
        let exp = oracle(&m, &x, slots, &gates);
        let ok = check(&format!("experts_b n={n} vs oracle"), &got, &exp);
        let same = got.iter().zip(&old).filter(|(a, b)| f32::from(**a) == f32::from(**b)).count();
        let samebits = got.iter().zip(&old).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        println!("experts_b n={n}: == old(ffn_k8+combine) {same}/{H} (bitwise {samebits}/{H}); fused launch {:.3}s", tf);
        all_ok &= ok && same == H;
    }

    #[cfg(backend = "npu")]
    if mode == "bench" {
        bench(&mut dev, &m, &x, &wg_t, &wu_t, &wd_t, &zeros, &zh, &mut ids, &mut g_hbm, &mut rows, &mut xn, &mut rng).await?;
    }
    let _ = (&mut rng, mode);
    println!("RESULT {}", if all_ok { "PASS" } else { "FAIL" });
    if !all_ok { std::process::exit(1); }
    Ok(())
}

#[cfg(backend = "npu")]
#[allow(clippy::too_many_arguments)]
async fn bench(
    dev: &mut Device,
    m: &Model,
    x: &[bf16],
    wg_t: &HbmTensor<bf16, solar_experts_b::axes::Chip, WGT>,
    wu_t: &HbmTensor<bf16, solar_experts_b::axes::Chip, WGT>,
    wd_t: &HbmTensor<bf16, solar_experts_b::axes::Chip, WDT>,
    zeros: &HbmTensor<bf16, solar_experts_b::axes::Chip, KH>,
    zh: &HbmTensor<bf16, solar_experts_b::axes::Chip, HA>,
    ids: &mut HbmScalar<u64>,
    g_hbm: &mut HbmTensor<f32, solar_experts_b::axes::Chip, KA>,
    rows: &mut HbmTensor<bf16, solar_experts_b::axes::Chip, KH>,
    xn: &mut HbmTensor<bf16, solar_experts_b::axes::Chip, HA>,
    rng: &mut SmallRng,
) -> eyre::Result<()> {
    use npu::*;
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut h2 = HostTensor::<bf16, HA>::zero().to_hbm(&mut dev.pdma).await?;
    let mut pk = packed(dev).await?;
    let mut out = HostTensor::<bf16, HA>::zero().to_hbm(&mut dev.pdma).await?;
    let mut host_out = HostTensor::<bf16, HA>::zero();
    let xbits: Vec<u16> = x.iter().map(|v| v.to_bits()).collect();

    // ---- packed-upload correctness (aliasing tensors) on NPU
    {
        let gates = gates_for(8);
        HostTensor::<bf16, PA>::from_buf(pack_bytes(x, &gates)).to_hbm(&mut dev.pdma).output(&mut pk.all).await?;
        ids.write(pack(&IDS), &mut dev.pdma).await?;
        launch(experts_b, (&mut *dev, &pk.x, &pk.gates, wg_t, wu_t, wd_t, &*ids, zeros, &mut out)).await?;
        let got: Vec<bf16> = out.to_host::<HA>(&mut dev.pdma).await?.into_vec();
        let exp = oracle(m, x, &IDS, &gates);
        let ok = check("experts_b packed-upload n=8 vs oracle", &got, &exp);
        println!("packed upload path: {}", if ok { "PASS" } else { "FAIL" });
    }

    for &n in &[1usize, 2, 4, 8] {
        let slots = &IDS[..n];
        let gates = gates_for(n);
        // old sequence, timed per stage
        let (mut up, mut ffn, mut comb, mut dl, mut tot) = (vec![], vec![], vec![], vec![], vec![]);
        for i in 0..iters + 10 {
            let t0 = Instant::now();
            HostTensor::<bf16, HA>::from_vec(xbits.iter().map(|&b| bf16::from_bits(b)).collect::<Vec<_>>()).to_hbm(&mut dev.pdma).output(&mut h2).await?;
            ids.write(pack(slots), &mut dev.pdma).await?;
            HostTensor::<f32, KA>::from_vec(gates.to_vec()).to_hbm(&mut dev.pdma).output(&mut *g_hbm).await?;
            let t1 = Instant::now();
            launch(ffn_k8, (&mut *dev, &h2, wg_t, wu_t, wd_t, &*ids, &mut *rows)).await?;
            let t2 = Instant::now();
            launch(combine, (&mut *dev, zh, &*rows, &*g_hbm, zh, &mut *xn)).await?;
            let t3 = Instant::now();
            let o: Vec<u16> = xn.to_host::<HA>(&mut dev.pdma).await?.into_vec().iter().map(|v| v.to_bits()).collect();
            let t4 = Instant::now();
            std::hint::black_box(o);
            if i >= 10 {
                up.push((t1 - t0).as_secs_f64() * 1e6);
                ffn.push((t2 - t1).as_secs_f64() * 1e6);
                comb.push((t3 - t2).as_secs_f64() * 1e6);
                dl.push((t4 - t3).as_secs_f64() * 1e6);
                tot.push((t4 - t0).as_secs_f64() * 1e6);
            }
        }
        println!(
            "OLD n={n}: total p50 {:.1}us (p90 {:.1}) | uploads(3) {:.1} | ffn_k8 {:.1} | combine {:.1} | download {:.1}",
            pct(tot.clone()).0, pct(tot).1, pct(up).0, pct(ffn).0, pct(comb).0, pct(dl).0
        );
        // fused
        let (mut up, mut ids_w, mut kern, mut dl, mut tot) = (vec![], vec![], vec![], vec![], vec![]);
        for i in 0..iters + 10 {
            let t0 = Instant::now();
            let bytes = {
                let xv: Vec<bf16> = xbits.iter().map(|&b| bf16::from_bits(b)).collect();
                pack_bytes(&xv, &gates)
            };
            HostTensor::<bf16, PA>::from_buf(bytes).to_hbm(&mut dev.pdma).output(&mut pk.all).await?;
            let t1 = Instant::now();
            ids.write(pack(slots), &mut dev.pdma).await?;
            let t2 = Instant::now();
            launch(experts_b, (&mut *dev, &pk.x, &pk.gates, wg_t, wu_t, wd_t, &*ids, zeros, &mut out)).await?;
            let t3 = Instant::now();
            out.to_host::<HA>(&mut dev.pdma).output(&mut host_out).await?;
            let o: Vec<u16> = host_out.clone().into_vec().iter().map(|v| v.to_bits()).collect();
            let t4 = Instant::now();
            std::hint::black_box(o);
            if i >= 10 {
                up.push((t1 - t0).as_secs_f64() * 1e6);
                ids_w.push((t2 - t1).as_secs_f64() * 1e6);
                kern.push((t3 - t2).as_secs_f64() * 1e6);
                dl.push((t4 - t3).as_secs_f64() * 1e6);
                tot.push((t4 - t0).as_secs_f64() * 1e6);
            }
        }
        println!(
            "NEW n={n}: total p50 {:.1}us (p90 {:.1}) | packed upload {:.1} | ids write {:.1} | experts_b {:.1} | download {:.1}",
            pct(tot.clone()).0, pct(tot).1, pct(up).0, pct(ids_w).0, pct(kern).0, pct(dl).0
        );
    }

    // ---- per-transfer micro costs
    {
        let gates = gates_for(8);
        let xv: Vec<bf16> = x.to_vec();
        let (mut a, mut b, mut c, mut d, mut e) = (vec![], vec![], vec![], vec![], vec![]);
        for _ in 0..iters {
            let t = Instant::now();
            HostTensor::<bf16, HA>::from_vec(xv.clone()).to_hbm(&mut dev.pdma).output(&mut h2).await?;
            a.push(t.elapsed().as_secs_f64() * 1e6);
            let t = Instant::now();
            HostTensor::<f32, KA>::from_vec(gates.to_vec()).to_hbm(&mut dev.pdma).output(&mut *g_hbm).await?;
            b.push(t.elapsed().as_secs_f64() * 1e6);
            let t = Instant::now();
            ids.write(pack(&IDS), &mut dev.pdma).await?;
            c.push(t.elapsed().as_secs_f64() * 1e6);
            let t = Instant::now();
            HostTensor::<bf16, PA>::from_buf(pack_bytes(x, &gates)).to_hbm(&mut dev.pdma).output(&mut pk.all).await?;
            d.push(t.elapsed().as_secs_f64() * 1e6);
            let t = Instant::now();
            out.to_host::<HA>(&mut dev.pdma).output(&mut host_out).await?;
            e.push(t.elapsed().as_secs_f64() * 1e6);
        }
        println!(
            "XFER p50: h2 bf16[H] {:.1}us | gates f32[8] {:.1}us | ids scalar {:.1}us | packed h2+gates {:.1}us | download bf16[H] into host tensor {:.1}us",
            pct(a).0, pct(b).0, pct(c).0, pct(d).0, pct(e).0
        );
    }

    // ---- swap primitive
    let swap_iters = 30;
    let e_new = 30usize; // a slot not in IDS[..4]
    let ng = rand_bf16(rng, F * H, 0.03);
    let nu = rand_bf16(rng, F * H, 0.03);
    let nd = rand_bf16(rng, H * F, 0.03);
    let hg = HostTensor::<bf16, WG>::from_vec(perm_wg(&ng));
    let hu = HostTensor::<bf16, WG>::from_vec(perm_wg(&nu));
    let hd = HostTensor::<bf16, WD>::from_vec(perm_wd(&nd));
    let hg_np = HostTensor::<bf16, WG>::from_vec(perm_wg(&ng));
    let mut sg = slot_wg(wg_t, e_new);
    let mut su = slot_wg(wu_t, e_new);
    let mut sd = slot_wd(wd_t, e_new);
    // A) direct PCIe write host -> table slot
    let mut lat = vec![];
    for _ in 0..swap_iters {
        let t = Instant::now();
        hg.to_hbm(&mut dev.pdma).output(&mut sg).await?;
        hu.to_hbm(&mut dev.pdma).output(&mut su).await?;
        hd.to_hbm(&mut dev.pdma).output(&mut sd).await?;
        lat.push(t.elapsed().as_secs_f64() * 1e6);
    }
    let (p50, p90, mn) = pct(lat);
    println!("SWAP-A direct host->slot (3x10.5MB): p50 {p50:.0}us p90 {p90:.0} min {mn:.0} -> {:.1} GB/s", 3.0 * (SLOT_ELEMS * 2) as f64 / p50 / 1e3);
    let mut lat = vec![];
    for _ in 0..swap_iters {
        let t = Instant::now();
        hg_np.to_hbm(&mut dev.pdma).output(&mut sg).await?;
        lat.push(t.elapsed().as_secs_f64() * 1e6);
    }
    let (p50, _, _) = pct(lat);
    println!("SWAP-A one matrix: p50 {p50:.0}us -> {:.1} GB/s", (SLOT_ELEMS * 2) as f64 / p50 / 1e3);
    // verify slot e_new now holds the new expert
    {
        let slots = [e_new, IDS[1]];
        let gates = gates_for(2);
        HostTensor::<bf16, PA>::from_buf(pack_bytes(x, &gates)).to_hbm(&mut dev.pdma).output(&mut pk.all).await?;
        ids.write(pack(&slots), &mut dev.pdma).await?;
        launch(experts_b, (&mut *dev, &pk.x, &pk.gates, wg_t, wu_t, wd_t, &*ids, zeros, &mut out)).await?;
        let got: Vec<bf16> = out.to_host::<HA>(&mut dev.pdma).await?.into_vec();
        let mut mm = Model { g: m.g.clone(), u: m.u.clone(), d: m.d.clone() };
        mm.g[e_new] = ng.clone(); mm.u[e_new] = nu.clone(); mm.d[e_new] = nd.clone();
        let exp = oracle(&mm, x, &slots, &gates);
        println!("SWAP-A verify: {}", if check("after swap slot 30", &got, &exp) { "PASS" } else { "FAIL" });
        // neighbours untouched?
        let slots = [IDS[3], IDS[0]]; // 31 and 5
        ids.write(pack(&slots), &mut dev.pdma).await?;
        launch(experts_b, (&mut *dev, &pk.x, &pk.gates, wg_t, wu_t, wd_t, &*ids, zeros, &mut out)).await?;
        let got: Vec<bf16> = out.to_host::<HA>(&mut dev.pdma).await?.into_vec();
        let exp = oracle(m, x, &slots, &gates);
        println!("SWAP-A neighbours (31,5): {}", if check("neighbours", &got, &exp) { "PASS" } else { "FAIL" });
    }
    // B) staged: upload to staging HBM, then device copy into runtime-selected slot
    let mut stg = HostTensor::<bf16, WG>::zero().to_hbm(&mut dev.pdma).await?;
    let mut stu = HostTensor::<bf16, WG>::zero().to_hbm(&mut dev.pdma).await?;
    let mut std_ = HostTensor::<bf16, WD>::zero().to_hbm(&mut dev.pdma).await?;
    hg.to_hbm(&mut dev.pdma).output(&mut stg).await?;
    hu.to_hbm(&mut dev.pdma).output(&mut stu).await?;
    hd.to_hbm(&mut dev.pdma).output(&mut std_).await?;
    let mut slot = HbmScalar::<usize>::from_host(29, &mut dev.pdma).await?;
    // swap_slot mutates the tables; tables are borrowed immutably elsewhere, so alias them mutably via their buffers
    let mut tg_m: HbmTensor<bf16, solar_experts_b::axes::Chip, WGT> = buf_of(wg_t).into();
    let mut tu_m: HbmTensor<bf16, solar_experts_b::axes::Chip, WGT> = buf_of(wu_t).into();
    let mut td_m: HbmTensor<bf16, solar_experts_b::axes::Chip, WDT> = buf_of(wd_t).into();
    let mut lat = vec![];
    for i in 0..swap_iters {
        slot.write(if i % 2 == 0 { 29 } else { 30 }, &mut dev.pdma).await?;
        let t = Instant::now();
        launch(swap_slot, (&mut *dev, &stg, &stu, &std_, &slot, &mut tg_m, &mut tu_m, &mut td_m)).await?;
        lat.push(t.elapsed().as_secs_f64() * 1e6);
    }
    let (p50, p90, mn) = pct(lat);
    println!("SWAP-B swap_slot launch (HBM->HBM 31.5MB): p50 {p50:.0}us p90 {p90:.0} min {mn:.0} -> {:.0} GB/s copy", 3.0 * (SLOT_ELEMS * 2) as f64 / p50 / 1e3);
    {
        let slots = [29usize, 30];
        let gates = gates_for(2);
        HostTensor::<bf16, PA>::from_buf(pack_bytes(x, &gates)).to_hbm(&mut dev.pdma).output(&mut pk.all).await?;
        ids.write(pack(&slots), &mut dev.pdma).await?;
        launch(experts_b, (&mut *dev, &pk.x, &pk.gates, wg_t, wu_t, wd_t, &*ids, zeros, &mut out)).await?;
        let got: Vec<bf16> = out.to_host::<HA>(&mut dev.pdma).await?.into_vec();
        let mut mm = Model { g: m.g.clone(), u: m.u.clone(), d: m.d.clone() };
        for e in [29, 30] { mm.g[e] = ng.clone(); mm.u[e] = nu.clone(); mm.d[e] = nd.clone(); }
        let exp = oracle(&mm, x, &slots, &gates);
        println!("SWAP-B verify: {}", if check("after swap_slot 29,30", &got, &exp) { "PASS" } else { "FAIL" });
    }

    // ---- overlap: PCIe slot write concurrently with an experts_b launch (n=8, slots disjoint from 30)
    {
        let gates = gates_for(8);
        HostTensor::<bf16, PA>::from_buf(pack_bytes(x, &gates)).to_hbm(&mut dev.pdma).output(&mut pk.all).await?;
        ids.write(pack(&IDS), &mut dev.pdma).await?;
        // second handle to the same PCIe DMA context (same Arc'd device); never dropped
        let mut pdma2 = std::mem::ManuallyDrop::new(unsafe { std::ptr::read(&dev.pdma) });
        let (mut a, mut b, mut c, mut d) = (vec![], vec![], vec![], vec![]);
        for _ in 0..swap_iters {
            let t = Instant::now();
            launch(experts_b, (&mut *dev, &pk.x, &pk.gates, wg_t, wu_t, wd_t, &*ids, zeros, &mut out)).await?;
            a.push(t.elapsed().as_secs_f64() * 1e6);
            let t = Instant::now();
            hg.to_hbm(&mut *pdma2).output(&mut sg).await?;
            b.push(t.elapsed().as_secs_f64() * 1e6);
            let t = Instant::now();
            let (r1, r2) = tokio::join!(
                launch(experts_b, (&mut *dev, &pk.x, &pk.gates, wg_t, wu_t, wd_t, &*ids, zeros, &mut out)),
                hg.to_hbm(&mut *pdma2).output(&mut sg)
            );
            r1?;
            r2?;
            c.push(t.elapsed().as_secs_f64() * 1e6);
            // also: one launch + a full 3-matrix swap in parallel
            let t = Instant::now();
            let (r1, r2) = tokio::join!(
                launch(experts_b, (&mut *dev, &pk.x, &pk.gates, wg_t, wu_t, wd_t, &*ids, zeros, &mut out)),
                async {
                    hg.to_hbm(&mut *pdma2).output(&mut sg).await?;
                    hu.to_hbm(&mut *pdma2).output(&mut su).await?;
                    hd.to_hbm(&mut *pdma2).output(&mut sd).await
                }
            );
            r1?;
            r2?;
            d.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let got: Vec<bf16> = out.to_host::<HA>(&mut dev.pdma).await?.into_vec();
        let exp = oracle(m, x, &IDS, &gates);
        let ok = check("experts_b n=8 during concurrent swap", &got, &exp);
        println!(
            "OVERLAP: launch alone p50 {:.0}us | slot write (1 matrix) alone {:.0}us | join(launch, 1 matrix) {:.0}us | join(launch, 3 matrices) {:.0}us | result {}",
            pct(a).0, pct(b).0, pct(c).0, pct(d).0, if ok { "PASS" } else { "FAIL" }
        );
        // two OS threads, each with its own runtime: thread S does a full 3-matrix slot write,
        // thread L launches experts_b back-to-back until S is done.
        let done = std::sync::atomic::AtomicBool::new(false);
        let pk_ref = &pk;
        let out_ref = &mut out;
        let ids_ref = &*ids;
        let pd = &mut *pdma2;
        let (sg_r, su_r, sd_r) = (&mut sg, &mut su, &mut sd);
        let (hg_r, hu_r, hd_r) = (&hg, &hu, &hd);
        let dev_r = &mut *dev;
        let (swap_us, lats) = std::thread::scope(|s| {
            let done = &done;
            let l = s.spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
                rt.block_on(async {
                    let mut lat = vec![];
                    while !done.load(std::sync::atomic::Ordering::Acquire) {
                        let t = Instant::now();
                        launch(experts_b, (&mut *dev_r, &pk_ref.x, &pk_ref.gates, wg_t, wu_t, wd_t, ids_ref, zeros, &mut *out_ref)).await.unwrap();
                        lat.push(t.elapsed().as_secs_f64() * 1e6);
                    }
                    lat
                })
            });
            std::thread::sleep(std::time::Duration::from_millis(2));
            let sw = s.spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
                let t = Instant::now();
                rt.block_on(async {
                    hg_r.to_hbm(&mut *pd).output(sg_r).await.unwrap();
                    hu_r.to_hbm(&mut *pd).output(su_r).await.unwrap();
                    hd_r.to_hbm(&mut *pd).output(sd_r).await.unwrap();
                });
                let us = t.elapsed().as_secs_f64() * 1e6;
                done.store(true, std::sync::atomic::Ordering::Release);
                us
            });
            let us = sw.join().unwrap();
            (us, l.join().unwrap())
        });
        let n = lats.len();
        println!(
            "OVERLAP-THREADS: 3-matrix slot write {swap_us:.0}us while {n} experts_b launches ran (launch p50 {:.0}us, sum {:.0}us)",
            pct(lats.clone()).0,
            lats.iter().sum::<f64>()
        );
    }
    Ok(())
}
