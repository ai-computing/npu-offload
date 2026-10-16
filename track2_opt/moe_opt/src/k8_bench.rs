use furiosa_opt_std::prelude::*;
use moe_opt::kernel::ffn::{WDT, WGT, ffn_k8, ffn_k8u};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::time::Instant;
#[path = "host_util.rs"]
mod host_util;
use host_util::*;
type HA = m![4096];
type KH = m![8, 4096];
const E: usize = 64;

fn expert_weights(e: usize) -> (Vec<bf16>, Vec<bf16>, Vec<bf16>) {
    let mut rng = SmallRng::seed_from_u64(1000 + e as u64);
    (rand_bf16(&mut rng, F * H, 0.03), rand_bf16(&mut rng, F * H, 0.03), rand_bf16(&mut rng, H * F, 0.03))
}

fn pack(ids: &[usize]) -> u64 {
    let mut p = (ids.len() as u64) << 56;
    for (k, &e) in ids.iter().enumerate() { p += (e as u64) << (6 * k); }
    p
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> eyre::Result<()> {
    let mut device = Device::new(ffn_k8.topology())?;
    let t0 = Instant::now();
    let (mut tg, mut tu, mut td) = (Vec::with_capacity(E * F * H), Vec::with_capacity(E * F * H), Vec::with_capacity(E * F * H));
    for e in 0..E {
        let (g, u, d) = expert_weights(e);
        tg.extend(perm_wg(&g)); tu.extend(perm_wg(&u)); td.extend(perm_wd(&d));
    }
    println!("generated {E} experts in {:.1}s", t0.elapsed().as_secs_f64());
    let t0 = Instant::now();
    let tg_hbm = HostTensor::<bf16, WGT>::from_vec(tg).to_hbm(&mut device.pdma).await?;
    let tu_hbm = HostTensor::<bf16, WGT>::from_vec(tu).to_hbm(&mut device.pdma).await?;
    let td_hbm = HostTensor::<bf16, WDT>::from_vec(td).to_hbm(&mut device.pdma).await?;
    println!("uploaded tables ({:.2} GB) in {:.1}s", (3 * E * F * H * 2) as f64 / 1e9, t0.elapsed().as_secs_f64());

    let mut rng = SmallRng::seed_from_u64(7);
    let x = rand_bf16(&mut rng, H, 1.0);
    let mut x_hbm = HostTensor::<bf16, HA>::from_vec(x.clone()).to_hbm(&mut device.pdma).await?;
    let mut out = HostTensor::<bf16, KH>::zero().to_hbm(&mut device.pdma).await?;
    let mut ids_hbm = HbmScalar::<u64>::from_host(0, &mut device.pdma).await?;

    // correctness: two different id sets, full and partial active
    for ids in [vec![3usize, 17, 42, 63, 0, 8, 29, 51], vec![60, 5, 33, 12]] {
        ids_hbm.write(pack(&ids), &mut device.pdma).await?;
        // poison output so skipped rows are visible
        HostTensor::<bf16, KH>::from_vec(vec![bf16::from_f32(-7.0); 8 * H]).to_hbm(&mut device.pdma).output(&mut out).await?;
        launch(ffn_k8, (&mut device, &x_hbm, &tg_hbm, &tu_hbm, &td_hbm, &ids_hbm, &mut out)).await?;
        let got: Vec<bf16> = out.to_host::<KH>(&mut device.pdma).await?.into_vec();
        let mut all_ok = true;
        for (k, &e) in ids.iter().enumerate() {
            let (g, u, d) = expert_weights(e);
            let exp = expert_ref(&x, &g, &u, &d);
            all_ok &= check(&format!("  ids={ids:?} k={k} expert={e}"), &got[k * H..(k + 1) * H], &exp);
        }
        let untouched = got[ids.len() * H..].iter().all(|v| f32::from(*v) == -7.0);
        println!("ids={ids:?}: all experts match = {all_ok}; inactive rows untouched = {untouched}");
    }

    let bytes = 8 * 3 * F * H * 2;
    let id_sets: Vec<Vec<usize>> = (0..16).map(|i| (0..8).map(|k| (i * 7 + k * 9) % E).collect()).collect();
    ids_hbm.write(pack(&id_sets[0]), &mut device.pdma).await?;
    for _ in 0..10 { launch(ffn_k8, (&mut device, &x_hbm, &tg_hbm, &tu_hbm, &td_hbm, &ids_hbm, &mut out)).await?; }
    // (a) device-resident: launch only, ids fixed
    let mut lat = vec![];
    for _ in 0..100 { let t = Instant::now(); launch(ffn_k8, (&mut device, &x_hbm, &tg_hbm, &tu_hbm, &td_hbm, &ids_hbm, &mut out)).await?; lat.push(t.elapsed().as_secs_f64() * 1e6); }
    report("K=8 launch only (ids fixed)", bytes, lat);
    // (b) ids rewritten each call + launch
    let mut lat = vec![];
    for i in 0..100 {
        let t = Instant::now();
        ids_hbm.write(pack(&id_sets[i % 16]), &mut device.pdma).await?;
        launch(ffn_k8, (&mut device, &x_hbm, &tg_hbm, &tu_hbm, &td_hbm, &ids_hbm, &mut out)).await?;
        lat.push(t.elapsed().as_secs_f64() * 1e6);
    }
    report("K=8 ids write + launch", bytes, lat);
    // (c) full host round trip: x upload + ids write + launch + output download
    let mut lat = vec![];
    let xh = HostTensor::<bf16, HA>::from_vec(x.clone());
    for i in 0..100 {
        let t = Instant::now();
        xh.to_hbm(&mut device.pdma).output(&mut x_hbm).await?;
        ids_hbm.write(pack(&id_sets[i % 16]), &mut device.pdma).await?;
        launch(ffn_k8, (&mut device, &x_hbm, &tg_hbm, &tu_hbm, &td_hbm, &ids_hbm, &mut out)).await?;
        let _o: Vec<bf16> = out.to_host::<KH>(&mut device.pdma).await?.into_vec();
        lat.push(t.elapsed().as_secs_f64() * 1e6);
    }
    report("K=8 host roundtrip (x up, ids, launch, out down)", bytes, lat);
    // (d) small pieces
    let mut lat = vec![];
    for i in 0..200 { let t = Instant::now(); ids_hbm.write(pack(&id_sets[i % 16]), &mut device.pdma).await?; lat.push(t.elapsed().as_secs_f64() * 1e6); }
    report("  HbmScalar write alone", 8, lat);
    let mut lat = vec![];
    for _ in 0..200 { let t = Instant::now(); xh.to_hbm(&mut device.pdma).output(&mut x_hbm).await?; lat.push(t.elapsed().as_secs_f64() * 1e6); }
    report("  x upload (8KB) alone", 8192, lat);
    let mut lat = vec![];
    for _ in 0..200 { let t = Instant::now(); let _o: Vec<bf16> = out.to_host::<KH>(&mut device.pdma).await?.into_vec(); lat.push(t.elapsed().as_secs_f64() * 1e6); }
    report("  out download (64KB) alone", 65536, lat);
    // unguarded variant
    {
        let ids = vec![3usize, 17, 42, 63, 0, 8, 29, 51];
        ids_hbm.write(pack(&ids), &mut device.pdma).await?;
        launch(ffn_k8u, (&mut device, &x_hbm, &tg_hbm, &tu_hbm, &td_hbm, &ids_hbm, &mut out)).await?;
        let got: Vec<bf16> = out.to_host::<KH>(&mut device.pdma).await?.into_vec();
        let mut all_ok = true;
        for (k, &e) in ids.iter().enumerate() {
            let (g, u, d) = expert_weights(e);
            all_ok &= check(&format!("  k8u k={k} expert={e}"), &got[k * H..(k + 1) * H], &expert_ref(&x, &g, &u, &d));
        }
        println!("ffn_k8u all experts match = {all_ok}");
        for _ in 0..10 { launch(ffn_k8u, (&mut device, &x_hbm, &tg_hbm, &tu_hbm, &td_hbm, &ids_hbm, &mut out)).await?; }
        let mut lat = vec![];
        for _ in 0..100 { let t = Instant::now(); launch(ffn_k8u, (&mut device, &x_hbm, &tg_hbm, &tu_hbm, &td_hbm, &ids_hbm, &mut out)).await?; lat.push(t.elapsed().as_secs_f64() * 1e6); }
        report("K=8 UNGUARDED launch only", bytes, lat);
        let mut lat = vec![];
        for i in 0..100 {
            let t = Instant::now();
            ids_hbm.write(pack(&id_sets[i % 16]), &mut device.pdma).await?;
            launch(ffn_k8u, (&mut device, &x_hbm, &tg_hbm, &tu_hbm, &td_hbm, &ids_hbm, &mut out)).await?;
            lat.push(t.elapsed().as_secs_f64() * 1e6);
        }
        report("K=8 UNGUARDED ids write + launch", bytes, lat);
    }
    // (e) active=4
    ids_hbm.write(pack(&id_sets[0][..4]), &mut device.pdma).await?;
    let mut lat = vec![];
    for _ in 0..100 { let t = Instant::now(); launch(ffn_k8, (&mut device, &x_hbm, &tg_hbm, &tu_hbm, &td_hbm, &ids_hbm, &mut out)).await?; lat.push(t.elapsed().as_secs_f64() * 1e6); }
    report("K=4 active (guarded) launch only", bytes / 2, lat);
    Ok(())
}
