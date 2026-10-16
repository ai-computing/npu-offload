//! expert_srv: furiosa-opt hot-expert server for the Solar hybrid runtime.
//!
//! Owns one RNGD chip (select with FURIOSA_OPT_NPUS / FURIOSA_VISIBLE_CHIPS). Keeps per-layer hot-expert
//! tables (E = 32 experts/layer, Solar dims) resident in HBM and serves top-8 expert FFN requests from a
//! Python client over a shared-memory mailbox (spin-wait on both sides, no syscalls on the hot path).
//!
//! Usage: expert_srv <shm_path> <weights_dir> <n_layers>
//!   weights_dir/layer_<l>.bin : E experts of raw row-major bf16, per expert [Wg F*H][Wu F*H][Wd H*F]
//!
//! Mailbox layout (little endian):
//!   0  u64 req_seq   (client increments after writing a request)
//!   8  u64 resp_seq  (server sets = req_seq when done)
//!   16 u32 op        (1 = ffn_k8u, 3 = quit)
//!   20 u32 layer
//!   24 u64 packed ids (count << 56 | e_k << 6k)
//!   32 u32 status    (0 ok, nonzero error)
//!   36 u32 ready     (server sets 1 once tables are resident)
//!   40 f64 last_us   (server-side latency of the last request)
//!   64      x   : H bf16
//!   16384   out : K*H bf16 (row k = expert k output; combine is done by the client)
use expert_srv::ffn::{WDT, WGT, ffn_k8u};
use furiosa_opt_std::prelude::*;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;
#[path = "host_util.rs"]
#[allow(dead_code)]
mod host_util;
use host_util::{F, H, perm_wd, perm_wg};

const E: usize = 32;
const K: usize = 8;
const OFF_X: usize = 64;
const OFF_OUT: usize = 16384;
pub const SHM_BYTES: usize = OFF_OUT + K * H * 2 + 4096;
type HA = m![4096];
type KH = m![8, 4096];

fn read_bf16(path: &str) -> eyre::Result<Vec<bf16>> {
    let bytes = std::fs::read(path)?;
    eyre::ensure!(bytes.len() % 2 == 0, "odd size {path}");
    Ok(bytes.chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> eyre::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    eyre::ensure!(args.len() == 4, "usage: expert_srv <shm_path> <weights_dir> <n_layers>");
    let (shm_path, wdir, n_layers) = (&args[1], &args[2], args[3].parse::<usize>()?);

    let file = std::fs::OpenOptions::new().read(true).write(true).open(shm_path)?;
    let mut mm = unsafe { memmap2::MmapMut::map_mut(&file)? };
    eyre::ensure!(mm.len() >= SHM_BYTES, "shm too small: {} < {}", mm.len(), SHM_BYTES);
    let base = mm.as_mut_ptr();
    let req = unsafe { &*(base as *const AtomicU64) };
    let resp = unsafe { &*(base.add(8) as *const AtomicU64) };
    let op = unsafe { &*(base.add(16) as *const AtomicU32) };
    let layer_f = unsafe { &*(base.add(20) as *const AtomicU32) };
    let packed_f = unsafe { &*(base.add(24) as *const AtomicU64) };
    let status = unsafe { &*(base.add(32) as *const AtomicU32) };
    let ready = unsafe { &*(base.add(36) as *const AtomicU32) };
    let last_us = base.wrapping_add(40) as *mut f64;

    let mut device = Device::new(ffn_k8u.topology())?;
    let t0 = Instant::now();
    let mut tables = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let raw = read_bf16(&format!("{wdir}/layer_{l}.bin"))?;
        let per = 3 * F * H;
        eyre::ensure!(raw.len() == E * per, "layer {l}: {} elems, want {}", raw.len(), E * per);
        let (mut tg, mut tu, mut td) = (Vec::with_capacity(E * F * H), Vec::with_capacity(E * F * H), Vec::with_capacity(E * F * H));
        for e in 0..E {
            let w = &raw[e * per..(e + 1) * per];
            tg.extend(perm_wg(&w[..F * H]));
            tu.extend(perm_wg(&w[F * H..2 * F * H]));
            td.extend(perm_wd(&w[2 * F * H..]));
        }
        let g = HostTensor::<bf16, WGT>::from_vec(tg).to_hbm(&mut device.pdma).await?;
        let u = HostTensor::<bf16, WGT>::from_vec(tu).to_hbm(&mut device.pdma).await?;
        let d = HostTensor::<bf16, WDT>::from_vec(td).to_hbm(&mut device.pdma).await?;
        tables.push((g, u, d));
        if l % 8 == 7 { eprintln!("[expert_srv] layer {l} resident ({:.0}s)", t0.elapsed().as_secs_f64()); }
    }
    let mut x_hbm = HostTensor::<bf16, HA>::zero().to_hbm(&mut device.pdma).await?;
    let mut out = HostTensor::<bf16, KH>::zero().to_hbm(&mut device.pdma).await?;
    let mut ids_hbm = HbmScalar::<u64>::from_host(0, &mut device.pdma).await?;
    // warm the kernel once per layer table shape (same shape for all layers)
    launch(ffn_k8u, (&mut device, &x_hbm, &tables[0].0, &tables[0].1, &tables[0].2, &ids_hbm, &mut out)).await?;
    eprintln!("[expert_srv] ready: {n_layers} layers x {E} experts in {:.0}s", t0.elapsed().as_secs_f64());
    resp.store(req.load(Ordering::Acquire), Ordering::Release);
    ready.store(1, Ordering::Release);

    let mut last = req.load(Ordering::Acquire);
    loop {
        let mut cur = req.load(Ordering::Acquire);
        let mut spins: u64 = 0;
        while cur == last {
            std::hint::spin_loop();
            spins += 1;
            if spins > 1 << 22 { std::thread::yield_now(); spins = 0; }
            cur = req.load(Ordering::Acquire);
        }
        last = cur;
        let t = Instant::now();
        let code = op.load(Ordering::Acquire);
        let st: u32 = match code {
            1 => {
                let l = layer_f.load(Ordering::Acquire) as usize;
                if l >= n_layers { 2 } else {
                    let xs = unsafe { std::slice::from_raw_parts(base.add(OFF_X) as *const bf16, H) }.to_vec();
                    HostTensor::<bf16, HA>::from_vec(xs).to_hbm(&mut device.pdma).output(&mut x_hbm).await?;
                    ids_hbm.write(packed_f.load(Ordering::Acquire), &mut device.pdma).await?;
                    let (g, u, d) = &tables[l];
                    launch(ffn_k8u, (&mut device, &x_hbm, g, u, d, &ids_hbm, &mut out)).await?;
                    let o: Vec<bf16> = out.to_host::<KH>(&mut device.pdma).await?.into_vec();
                    unsafe { std::ptr::copy_nonoverlapping(o.as_ptr(), base.add(OFF_OUT) as *mut bf16, K * H) };
                    0
                }
            }
            3 => { status.store(0, Ordering::Release); resp.store(cur, Ordering::Release); break; }
            _ => 1,
        };
        unsafe { *last_us = t.elapsed().as_secs_f64() * 1e6 };
        status.store(st, Ordering::Release);
        resp.store(cur, Ordering::Release);
    }
    eprintln!("[expert_srv] quit");
    Ok(())
}
