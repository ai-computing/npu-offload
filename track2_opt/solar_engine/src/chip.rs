//! One RNGD chip driven by its own OS thread + current-thread tokio runtime (two chips run truly in
//! parallel this way; a single runtime serializes launches). Requests go over a channel.
//!
//! Chip A ("dense"): shared expert + router, hot-expert table, combine, lm_head.
//! Chip B ("experts"): hot-expert table; combines its own experts on device and returns one [H] vector.
use furiosa_opt_std::prelude::*;
use solar_engine::axes::{Chip as ChipT, H, K, R, W};
use solar_engine::dense::{combine::combine, lm_head::{WLM, lm_head_am}, shared::{WD as SWD, WG as SWG, WR, shared_router}};
use solar_engine::ffn::{WDT, WGT, ffn_k8};
use solar_engine::ffn32;
use solar_engine::experts_b::experts_b;
use solar_engine::axes::{Pack, PK};
use furiosa_opt_rt::Buffer;
use furiosa_opt_std::runtime::{Buffers, DeviceSend};

/// The device allocation behind a placed HBM tensor.
fn buf_of<T: DeviceSend>(t: &T) -> Buffer {
    let mut b = Buffers::new();
    t.bind(&mut b).expect("unplaced");
    b[0].clone()
}
use solar_engine::layer::{layer_a_c1, layer_a_c2, layer_a_c4, layer_a_c8, layer_a_c16};
use solar_engine::unified::{AttnW, ExpertT, SharedW, unified_a_c1, unified_a_c2, unified_a_c4, unified_a_c8, unified_a_c16};
use solar_engine::axes::{Pk, U, WH};
use solar_engine::dense::lm_half::{WLMH, lm_head_am_h};
use solar_engine::layer::{AttnSlot, SharedSlot};
use solar_engine::unified_res::{ExpertT as ExpertTR, MODE_FFN, MODE_PREFETCH, PfW, mode_layer,
    unified_res_c1, unified_res_c2, unified_res_c4, unified_res_c8, unified_res_c16};
use solar_engine::attn::{D, KvCache, Masks, attn_block_c1, attn_block_c2, attn_block_c4, attn_block_c8, attn_block_c16};
use solar_engine::attn::gemv::{WKV, WO, WQ};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::Instant;

#[path = "host_util.rs"]
#[allow(dead_code)]
mod host_util;
#[path = "attn_host.rs"]
#[allow(dead_code)]
mod attn_host;
#[path = "dense_host_util.rs"]
#[allow(dead_code)]
mod dense_host_util;

pub const SLOTS: usize = 24; // chip A experts per layer table (axes E)
pub const SLOTS_B: usize = 32; // chip B experts per layer table (axes E2)
const HN: usize = 4096;
const FN: usize = 1280;
const VN: usize = 196608;

pub type Ex = (&'static [u16], &'static [u16], &'static [u16]);
pub struct DenseLayer {
    pub sg: &'static [u16], pub su: &'static [u16], pub sd: &'static [u16], pub wr: &'static [u16],
    pub w_in: &'static [u16], pub wq: &'static [u16], pub wk: &'static [u16], pub wv: &'static [u16],
    pub wo: &'static [u16], pub w_post: &'static [u16],
}
pub struct DenseW { pub layers: Vec<DenseLayer>, pub lm: &'static [u16], pub w_final: &'static [u16] }

type R1<T> = Sender<eyre::Result<T>>;
pub enum Cmd {
    /// per token: embedding row -> xcur, rope cos/sin (raw HF, sin sign-folded here), position
    Token { emb: Vec<u16>, cos: Vec<u16>, sin: Vec<u16>, pos: usize, s: usize },
    /// attention block + shared/router on xcur; replies (router logits f32[128], h2 bf16[H])
    LayerA { layer: usize, reply: R1<(Vec<f32>, Vec<u16>)> },
    SharedRouter { layer: usize, h2: Vec<u16>, x2: Vec<u16>, reply: R1<Vec<f32>> },
    ExpertsA { layer: usize, slots: Vec<usize> },
    /// fused layer step: x = y_prev + sum g·out_A + extra (packed inp), attention block, shared+router;
    /// replies (router logits f32[128], h2 bf16[H]) from ONE packed download
    LayerA2 { layer: usize, inp: Vec<f32>, reply: R1<(Vec<f32>, Vec<u16>)>, s: usize },
    /// copy layer `layer`'s attention + shared/router weights HBM -> resident DM (runs in the chip-idle window)
    Prefetch { layer: usize },
    /// final: x = y_last + sum g·out_A + extra, then lm_head argmax
    FinalLm { inp: Vec<f32>, reply: R1<usize>, s: usize },
    /// ExpertsA with an explicit h2 upload (fused path keeps h2 only in the packed download)
    ExpertsA2 { layer: usize, h2: Vec<u16>, slots: Vec<usize>, s: usize },
    /// xcur = y + sum_k g_k out_k + extra  (stays in HBM; no reply)
    CombineA { gates: [f32; 8], extra: Vec<u16> },
    /// lm_head on xcur -> argmax token
    LmHead { reply: R1<usize> },
    Stats { reply: R1<String> },
    /// chip B (LM_ON_B=1): lm_head on an uploaded final x -> argmax token
    LmHeadX { x: Vec<u16>, reply: R1<usize> },
    /// LM_SPLIT=1, chip B: its half of the head (rows WH..W) on an uploaded x -> (bf16 max, row)
    LmHalfX { x: Vec<u16>, reply: R1<(f32, usize)> },
    /// LM_SPLIT=1, chip A: final combine -> x to `xreply`, then its half of the head (rows 0..WH) -> `reply`
    FinalSplit { inp: Vec<f32>, xreply: R1<Vec<u16>>, reply: R1<(f32, usize)>, s: usize },
    /// chip A (LM_ON_B=1): final combine only, returns x (keeps the resident slots)
    FinalX { inp: Vec<f32>, reply: R1<Vec<u16>>, s: usize },
    ExpertsB { layer: usize, h2: Vec<u16>, slots: Vec<usize>, gates: [f32; 8], reply: R1<Vec<u16>>, t0: Instant },
    Quit,
}

pub struct Chip { pub tx: Sender<Cmd>, handle: Option<JoinHandle<eyre::Result<()>>> }

impl Chip {
    pub fn spawn(tag: usize, tables: Vec<Vec<Ex>>, dense: Option<DenseW>, ready: Sender<usize>) -> Chip {
        let (tx, rx) = channel::<Cmd>();
        let handle = std::thread::spawn(move || -> eyre::Result<()> {
            // chip-driver threads (and the runtime helper threads they spawn) share cores 0..CHIP_CORES-1;
            // the CPU-expert OpenMP team lives on the remaining cores. (2 cores per chip was too few.)
            let cc: usize = std::env::var("CHIP_CORES").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
            if cc > 0 { crate::host::pin_current(0, cc - 1); }
            let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
            let r = rt.block_on(worker(tag, tables, dense, rx, &ready));
            if let Err(e) = &r { eprintln!("[chip{tag}] ERROR {e:?}"); ready.send(usize::MAX).ok(); }
            r
        });
        Chip { tx, handle: Some(handle) }
    }
    pub fn send(&self, c: Cmd) { self.tx.send(c).expect("chip thread gone"); }
}

impl Drop for Chip {
    fn drop(&mut self) {
        let _ = self.tx.send(Cmd::Quit);
        if let Some(h) = self.handle.take() { let _ = h.join(); }
    }
}

pub fn rx<T>() -> (R1<T>, Receiver<eyre::Result<T>>) { channel() }

/// Receive with a short busy-spin before blocking: thread wake-up latency (~tens of µs) is on the
/// per-layer critical path; spinning briefly avoids it. SPIN_US (default 2000) bounds the spin.
pub fn spin_recv<T>(r: &Receiver<T>) -> Option<T> {
    use std::sync::mpsc::TryRecvError;
    static LIM: std::sync::OnceLock<u128> = std::sync::OnceLock::new();
    let lim = *LIM.get_or_init(|| std::env::var("SPIN_US").ok().and_then(|s| s.parse().ok()).unwrap_or(2000));
    let serve = mmio::is_server();
    let t = Instant::now();
    loop {
        match r.try_recv() {
            Ok(v) => return Some(v),
            Err(TryRecvError::Disconnected) => return None,
            Err(TryRecvError::Empty) => {
                if serve {
                    // the main thread (and its idle OpenMP team) serves the chips' small device reads
                    mmio::serve();
                } else if t.elapsed().as_micros() > lim { return r.recv().ok(); }
                std::hint::spin_loop();
            }
        }
    }
}

/// Small device->host reads served by the main thread: a chip thread posts (src in the BAR window, dst, len) and
/// spins; the main thread, which is waiting for exactly that result, copies with its OpenMP team (parallel 64-byte
/// uncached loads overlap on PCIe: 8 KB in ~10 us vs ~50 us for a DMA).
pub mod mmio {
    use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering::*};
    struct Job { st: AtomicU8, src: AtomicUsize, dst: AtomicUsize, len: AtomicUsize }
    const FREE: u8 = 0; const CLAIM: u8 = 1; const POSTED: u8 = 2; const DONE: u8 = 3;
    static JOBS: [Job; 4] = [const { Job { st: AtomicU8::new(FREE), src: AtomicUsize::new(0), dst: AtomicUsize::new(0), len: AtomicUsize::new(0) } }; 4];
    static ON: AtomicBool = AtomicBool::new(false);
    std::thread_local! { static SERVER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
    /// call on the main thread once it only waits through `spin_recv` (after init)
    pub fn start_serving() { SERVER.with(|s| s.set(true)); ON.store(true, Release); }
    pub fn is_server() -> bool { SERVER.with(|s| s.get()) }
    pub fn serve() {
        for j in JOBS.iter() {
            if j.st.load(Acquire) == POSTED {
                unsafe { crate::host::ck_mmio_read(j.src.load(Relaxed) as *const u8, j.dst.load(Relaxed) as *mut u8, j.len.load(Relaxed)) };
                j.st.store(DONE, Release);
            }
        }
    }
    pub unsafe extern "C" fn hook(src: *const u8, dst: *mut u8, len: usize) -> bool {
        if !ON.load(Acquire) || is_server() { return false; }
        loop {
            for j in JOBS.iter() {
                if j.st.compare_exchange(FREE, CLAIM, AcqRel, Relaxed).is_ok() {
                    j.src.store(src as usize, Relaxed); j.dst.store(dst as usize, Relaxed); j.len.store(len, Relaxed);
                    j.st.store(POSTED, Release);
                    while j.st.load(Acquire) != DONE { std::hint::spin_loop(); }
                    j.st.store(FREE, Release);
                    return true;
                }
            }
            std::hint::spin_loop();
        }
    }
}

const WHN: usize = 98304;
/// argmax over per-512-row keys (see lm_head_am): (bf16 max value, row); ties -> lowest row
fn best_key(k: &[f32]) -> (f32, usize) {
    let mut best = (f32::MIN, usize::MAX);
    for (s, &kv) in k.iter().enumerate() {
        let bits = kv.to_bits();
        let val = f32::from_bits(bits & 0xFFFF0000);
        let row = s * 512 + 511 - (bits & 0x1FF) as usize;
        if val > best.0 || (val == best.0 && row < best.1) { best = (val, row); }
    }
    best
}
fn bf(v: &[u16]) -> Vec<bf16> { v.iter().map(|&b| bf16::from_bits(b)).collect() }
fn pack(slots: &[usize]) -> u64 {
    let mut p = (slots.len() as u64) << 56;
    for (k, &s) in slots.iter().enumerate() { p += (s as u64) << (6 * k); }
    p
}

async fn worker(tag: usize, tables: Vec<Vec<Ex>>, dense: Option<DenseW>, rx: Receiver<Cmd>, ready: &Sender<usize>) -> eyre::Result<()> {
    let mut dev = Device::new(ffn_k8.topology())?;
    let t0 = Instant::now();
    let mut tabs = Vec::with_capacity(tables.len());
    let mut tabs32 = Vec::with_capacity(tables.len());
    let big = tables.first().map(|t| t.len() == SLOTS_B).unwrap_or(false);
    // permute all layer tables in parallel (host CPU), then upload sequentially
    let t_perm = Instant::now();
    let permuted: Vec<(Vec<bf16>, Vec<bf16>, Vec<bf16>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = tables.iter().map(|tl| sc.spawn(move || {
            assert!(tl.len() == if big { SLOTS_B } else { SLOTS });
            let n = tl.len();
            let (mut tg, mut tu, mut td) = (Vec::with_capacity(n * FN * HN), Vec::with_capacity(n * FN * HN), Vec::with_capacity(n * FN * HN));
            for (g, u, d) in tl {
                tg.extend(host_util::perm_wg(&bf(g))); tu.extend(host_util::perm_wg(&bf(u))); td.extend(host_util::perm_wd(&bf(d)));
            }
            (tg, tu, td)
        })).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    eprintln!("[chip{tag}] permuted tables in {:.0}s", t_perm.elapsed().as_secs_f64());
    for (tg, tu, td) in permuted {
        if big {
            tabs32.push((HostTensor::<bf16, ffn32::WGT>::from_vec(tg).to_hbm(&mut dev.pdma).await?,
                         HostTensor::<bf16, ffn32::WGT>::from_vec(tu).to_hbm(&mut dev.pdma).await?,
                         HostTensor::<bf16, ffn32::WDT>::from_vec(td).to_hbm(&mut dev.pdma).await?));
        } else {
            tabs.push((HostTensor::<bf16, WGT>::from_vec(tg).to_hbm(&mut dev.pdma).await?,
                       HostTensor::<bf16, WGT>::from_vec(tu).to_hbm(&mut dev.pdma).await?,
                       HostTensor::<bf16, WDT>::from_vec(td).to_hbm(&mut dev.pdma).await?));
        }
    }
    // dense weights (chip A)
    let mut dl = vec![];
    let mut attw = vec![];
    let mut kvc = vec![];
    let mut lmw = None;
    let mut lmwh = None;
    let lm_split = std::env::var("LM_SPLIT").map(|v| v == "1").unwrap_or(false);
    if let Some(dw) = &dense {
        for d in dw.layers.iter() {
            dl.push((HostTensor::<bf16, SWG>::from_vec(dense_host_util::perm_wg(&bf(d.sg))).to_hbm(&mut dev.pdma).await?,
                     HostTensor::<bf16, SWG>::from_vec(dense_host_util::perm_wg(&bf(d.su))).to_hbm(&mut dev.pdma).await?,
                     HostTensor::<bf16, SWD>::from_vec(dense_host_util::perm_wd(&bf(d.sd))).to_hbm(&mut dev.pdma).await?,
                     HostTensor::<bf16, WR>::from_vec(dense_host_util::perm_wr(&bf(d.wr))).to_hbm(&mut dev.pdma).await?));
        }
        for d in &dw.layers {
            attw.push((HostTensor::<bf16, m![H]>::from_vec(bf(d.w_in)).to_hbm(&mut dev.pdma).await?,
                       HostTensor::<bf16, WQ>::from_vec(attn_host::perm_wq(&bf(d.wq))).to_hbm(&mut dev.pdma).await?,
                       HostTensor::<bf16, WKV>::from_vec(attn_host::perm_wkv(&bf(d.wk), &bf(d.wv))).to_hbm(&mut dev.pdma).await?,
                       HostTensor::<bf16, WO>::from_vec(attn_host::perm_wo(&bf(d.wo))).to_hbm(&mut dev.pdma).await?,
                       HostTensor::<bf16, m![H]>::from_vec(bf(d.w_post)).to_hbm(&mut dev.pdma).await?));
            kvc.push((HostTensor::<bf16, KvCache>::zero().to_hbm(&mut dev.pdma).await?,
                      HostTensor::<bf16, KvCache>::zero().to_hbm(&mut dev.pdma).await?));
        }
        let iota: Vec<f32> = (0..512u32).map(|j| f32::from_bits(511 - j)).collect();
        if lm_split {
            let rows = if tag == 0 { &dw.lm[..WHN * HN] } else { &dw.lm[WHN * HN..] };
            lmwh = Some((HostTensor::<bf16, WLMH>::from_vec(dense_host_util::perm_wlm(&bf(rows))).to_hbm(&mut dev.pdma).await?,
                         HostTensor::<bf16, m![H]>::from_vec(bf(dw.w_final)).to_hbm(&mut dev.pdma).await?,
                         HostTensor::<f32, m![WH % 512]>::from_vec(iota.clone()).to_hbm(&mut dev.pdma).await?));
        } else {
        lmw = Some((HostTensor::<bf16, WLM>::from_vec(dense_host_util::perm_wlm(&bf(dw.lm))).to_hbm(&mut dev.pdma).await?,
                    HostTensor::<bf16, m![H]>::from_vec(bf(dw.w_final)).to_hbm(&mut dev.pdma).await?,
                    HostTensor::<f32, m![W % 512]>::from_vec(iota).to_hbm(&mut dev.pdma).await?));
        }
    }
    let mut lmlh = HostTensor::<bf16, m![WH]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut keysh = HostTensor::<f32, m![WH / 512]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut h2 = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut x2 = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut y = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut dev.pdma).await?;
    let zero = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut extra = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut xn = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut logits = HostTensor::<f32, m![R]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut out = HostTensor::<bf16, m![K, H]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut gates = HostTensor::<f32, m![K]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut ids = HbmScalar::<u64>::from_host(0, &mut dev.pdma).await?;
    let zeros_kh = HostTensor::<bf16, m![K, H]>::zero().to_hbm(&mut dev.pdma).await?;
    // chip B packed upload: one allocation, h2 (bf16[H]) and gates (f32[8]) alias slices of it
    let mut packed_all: HbmTensor<bf16, ChipT, m![PK]> = HostTensor::<bf16, m![PK]>::zero().to_hbm(&mut dev.pdma).await?;
    let (pk_x, pk_g): (HbmTensor<bf16, ChipT, m![H]>, HbmTensor<f32, ChipT, m![K]>) = {
        let b = buf_of(&packed_all);
        (b.slice(0..2 * HN).into(), b.slice(2 * HN..2 * HN + 32).into())
    };
    let mut inp_t = HostTensor::<f32, Pack>::zero().to_hbm(&mut dev.pdma).await?;
    let mut outp_t = HostTensor::<f32, Pack>::zero().to_hbm(&mut dev.pdma).await?;
    let mut ybuf = [HostTensor::<bf16, m![H]>::zero().to_hbm(&mut dev.pdma).await?, HostTensor::<bf16, m![H]>::zero().to_hbm(&mut dev.pdma).await?];
    let mut ylast = 0usize; // index of ybuf holding the latest y_out
    // resident DM weight slot (one layer) for the compute-only layer_a_res kernels (chip A only)
    // unified chip-A program buffers ([U=2, ...]: row 0 used) and the two mode scalars
    let full_dense = dense.as_ref().map(|d| !d.layers.is_empty()).unwrap_or(false);
    let unified = full_dense && std::env::var("NO_UNIFIED").is_err();
    let mut uy = [HostTensor::<bf16, m![U, H]>::zero().to_hbm(&mut dev.pdma).await?, HostTensor::<bf16, m![U, H]>::zero().to_hbm(&mut dev.pdma).await?];
    let uy0 = HostTensor::<bf16, m![U, H]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut uh2 = HostTensor::<bf16, m![U, H]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut uouts = HostTensor::<f32, m![U, Pk, R]>::zero().to_hbm(&mut dev.pdma).await?;
    let uouts_row0: HbmTensor<f32, ChipT, Pack> = buf_of(&uouts).slice(0..33 * 128 * 4).into();
    let mut uy_row0: [HbmTensor<bf16, ChipT, m![H]>; 2] = [buf_of(&uy[0]).slice(0..2 * HN).into(), buf_of(&uy[1]).slice(0..2 * HN).into()];
    let mut mode_l = HbmScalar::<u64>::from_host(1, &mut dev.pdma).await?;
    let mode_f = HbmScalar::<u64>::from_host(0, &mut dev.pdma).await?;
    // URES=1: one chip-A program with layer(resident weights) | ffn | prefetch modes
    let ures = full_dense && std::env::var("URES").map(|v| v == "1").unwrap_or(false);
    let mode_ffn = HbmScalar::<u64>::from_host(MODE_FFN, &mut dev.pdma).await?;
    let mode_pf = HbmScalar::<u64>::from_host(MODE_PREFETCH, &mut dev.pdma).await?;
    let mut ra = if ures { Some(AttnSlot::alloc(&dev)?) } else { None };
    let mut rs = if ures { Some(SharedSlot::alloc(&dev)?) } else { None };
    let mut res_layer: Option<usize> = None;
    let mut lml = HostTensor::<bf16, m![W]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut xcur = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut cos_t = HostTensor::<bf16, m![D]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut sin_t = HostTensor::<bf16, m![D]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut kvoff = HostTensor::<i32, m![1]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut masks_t = HostTensor::<f32, Masks>::zero().to_hbm(&mut dev.pdma).await?;
    let mut ctx = 0usize;
    let mut keys = HostTensor::<f32, m![W / 512]>::zero().to_hbm(&mut dev.pdma).await?;
    // ---- optional second decode stream (NSTREAM=2): its own KV caches + per-stream buffers, swapped in on demand ----
    let nstream: usize = if full_dense { std::env::var("NSTREAM").ok().and_then(|v| v.parse().ok()).unwrap_or(1) } else { 1 };
    let mut cur_s = 0usize;
    let mut o_kvc = vec![];
    if nstream > 1 { for _ in 0..kvc.len() { o_kvc.push((HostTensor::<bf16, KvCache>::zero().to_hbm(&mut dev.pdma).await?, HostTensor::<bf16, KvCache>::zero().to_hbm(&mut dev.pdma).await?)); } }
    let mut o_out = HostTensor::<bf16, m![K, H]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut o_uy = [HostTensor::<bf16, m![U, H]>::zero().to_hbm(&mut dev.pdma).await?, HostTensor::<bf16, m![U, H]>::zero().to_hbm(&mut dev.pdma).await?];
    let mut o_uy_row0: [HbmTensor<bf16, ChipT, m![H]>; 2] = [buf_of(&o_uy[0]).slice(0..2 * HN).into(), buf_of(&o_uy[1]).slice(0..2 * HN).into()];
    let mut o_uh2 = HostTensor::<bf16, m![U, H]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut o_ylast = 0usize;
    let mut o_mode_l = HbmScalar::<u64>::from_host(1, &mut dev.pdma).await?;
    let mut o_cos = HostTensor::<bf16, m![D]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut o_sin = HostTensor::<bf16, m![D]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut o_masks = HostTensor::<f32, Masks>::zero().to_hbm(&mut dev.pdma).await?;
    let mut o_kvoff = HostTensor::<i32, m![1]>::zero().to_hbm(&mut dev.pdma).await?;
    let mut o_ctx = 0usize;
    // launch the unified_res program: $mode scalar, current layer $l (KV/w_in/tables), prefetch source layer $pl
    macro_rules! ures_launch { ($mode:expr, $l:expr, $pl:expr, $yin:expr, $yout:expr) => {{
        let (kc, vc) = &mut kvc[$l];
        ures_launch!($mode, $l, $pl, $yin, $yout, kc, vc)
    }};
    ($mode:expr, $l:expr, $pl:expr, $yin:expr, $yout:expr, $kc:expr, $vc:expr) => {{
        let (w_in, _, _, _, w_post) = &attw[$l];
        let (kc, vc) = ($kc, $vc);
        let (_, pwq, pwkv, pwo, _) = &attw[$pl];
        let (pg, pu, pd, pr) = &dl[$pl];
        let pw = PfW { wq: pwq, wkv: pwkv, wo: pwo, wg: pg, wu: pu, wd: pd, wr: pr };
        let (tg, tu, td) = &tabs[$l];
        let et = ExpertTR { wg_t: tg, wu_t: tu, wd_t: td };
        let (a, sh) = (ra.as_mut().unwrap(), rs.as_mut().unwrap());
        let chunks = ctx.div_ceil(512).next_power_of_two();
        macro_rules! gr { ($f:ident) => { launch($f, (&mut dev, $mode, $yin, &mut out, &inp_t, w_in, w_post, &cos_t, &sin_t, kc, vc, &masks_t, pw, a.w(), sh.w(), et, &ids, &mut uouts, $yout, &mut uh2)).await? } }
        match chunks { 1 => gr!(unified_res_c1), 2 => gr!(unified_res_c2), 4 => gr!(unified_res_c4), 8 => gr!(unified_res_c8), 16 => gr!(unified_res_c16),
                       _ => eyre::bail!("context {ctx} exceeds 8192") };
    }} }
    macro_rules! use_stream { ($s:expr) => { if $s != cur_s {
        assert!(nstream > 1, "stream {} requested but NSTREAM=1", $s);
        std::mem::swap(&mut kvc, &mut o_kvc); std::mem::swap(&mut out, &mut o_out); std::mem::swap(&mut uy, &mut o_uy);
        std::mem::swap(&mut uy_row0, &mut o_uy_row0); std::mem::swap(&mut uh2, &mut o_uh2); std::mem::swap(&mut ylast, &mut o_ylast);
        std::mem::swap(&mut mode_l, &mut o_mode_l); std::mem::swap(&mut cos_t, &mut o_cos); std::mem::swap(&mut sin_t, &mut o_sin);
        std::mem::swap(&mut masks_t, &mut o_masks); std::mem::swap(&mut kvoff, &mut o_kvoff); std::mem::swap(&mut ctx, &mut o_ctx);
        cur_s = $s;
    } } }
    eprintln!("[chip{tag}] resident: {} expert tables (x{}){} in {:.0}s", tabs.len() + tabs32.len(), if big { SLOTS_B } else { SLOTS },
              if dense.is_some() { " + dense + lm_head" } else { "" }, t0.elapsed().as_secs_f64());
    ready.send(tag).ok();
    let mut st: std::collections::BTreeMap<&'static str, (f64, usize)> = Default::default();
    let mut idle = 0f64;
    macro_rules! tm { ($k:expr, $t:expr) => {{ let e = st.entry($k).or_insert((0.0, 0)); e.0 += $t.elapsed().as_secs_f64(); e.1 += 1; Instant::now() }} }
    loop {
        let ti = Instant::now();
        let Some(cmd) = spin_recv(&rx) else { break };
        idle += ti.elapsed().as_secs_f64();
        let tc = Instant::now();
        match cmd {
            Cmd::Quit => break,
            Cmd::Stats { reply } => {
                let mut v: Vec<String> = st.iter().map(|(k, (t, n))| format!("{k}={:.0}us x{n}", t * 1e6 / (*n).max(1) as f64)).collect();
                v.push(format!("idle={:.2}s", idle));
                st.clear(); idle = 0.0;
                let _ = reply.send(Ok(v.join(" ")));
            }
            Cmd::SharedRouter { layer, h2: hh, x2: xx, reply } => {
                let r: eyre::Result<Vec<f32>> = async {
                    HostTensor::<bf16, m![H]>::from_vec(bf(&hh)).to_hbm(&mut dev.pdma).output(&mut h2).await?;
                    HostTensor::<bf16, m![H]>::from_vec(bf(&xx)).to_hbm(&mut dev.pdma).output(&mut x2).await?;
                    let (g, u, d, wr) = &dl[layer];
                    launch(shared_router, (&mut dev, &h2, &x2, g, u, d, wr, &mut y, &mut logits)).await?;
                    Ok(logits.to_host::<m![R]>(&mut dev.pdma).await?.into_vec())
                }.await;
                let _ = reply.send(r);
            }
            Cmd::ExpertsA { layer, slots } => {
                ids.write(pack(&slots), &mut dev.pdma).await?;
                let t1 = tm!("EA.ids", tc);
                if big { let (g, u, d) = &tabs32[layer]; launch(ffn32::ffn_k8, (&mut dev, &h2, g, u, d, &ids, &mut out)).await?; }
                else { let (g, u, d) = &tabs[layer]; launch(ffn_k8, (&mut dev, &h2, g, u, d, &ids, &mut out)).await?; }
                tm!(if slots.len() <= 2 { "EA.ffn<=2" } else if slots.len() <= 4 { "EA.ffn<=4" } else { "EA.ffn>4" }, t1);
            }
            Cmd::Token { emb, cos, sin, pos, s } => {
                use_stream!(s);
                HostTensor::<bf16, m![H]>::from_vec(bf(&emb)).to_hbm(&mut dev.pdma).output(&mut xcur).await?;
                HostTensor::<bf16, m![D]>::from_vec(bf(&cos)).to_hbm(&mut dev.pdma).output(&mut cos_t).await?;
                HostTensor::<bf16, m![D]>::from_vec(attn_host::rope_sin_signed(&bf(&sin))).to_hbm(&mut dev.pdma).output(&mut sin_t).await?;
                HostTensor::<i32, m![1]>::from_vec(vec![attn_host::kv_offset(pos)]).to_hbm(&mut dev.pdma).output(&mut kvoff).await?;
                ctx = pos + 1;
                if ures { mode_l.write(mode_layer(pos), &mut dev.pdma).await?; }
                else if unified { mode_l.write((pos + 1) as u64, &mut dev.pdma).await?; }
                HostTensor::<f32, Masks>::from_vec(attn_host::masks(ctx)).to_hbm(&mut dev.pdma).output(&mut masks_t).await?;
            }
            Cmd::LayerA { layer, reply } => {
                let r: eyre::Result<(Vec<f32>, Vec<u16>)> = async {
                    let (w_in, wq, wkv, wo, w_post) = &attw[layer];
                    let (kc, vc) = &mut kvc[layer];
                    let chunks = ctx.div_ceil(512).next_power_of_two();
                    match chunks {
                        1 => launch(attn_block_c1, (&mut dev, &xcur, w_in, wq, wkv, &kvoff, &cos_t, &sin_t, kc, vc, &masks_t, wo, w_post, &mut x2, &mut h2)).await?,
                        2 => launch(attn_block_c2, (&mut dev, &xcur, w_in, wq, wkv, &kvoff, &cos_t, &sin_t, kc, vc, &masks_t, wo, w_post, &mut x2, &mut h2)).await?,
                        4 => launch(attn_block_c4, (&mut dev, &xcur, w_in, wq, wkv, &kvoff, &cos_t, &sin_t, kc, vc, &masks_t, wo, w_post, &mut x2, &mut h2)).await?,
                        8 => launch(attn_block_c8, (&mut dev, &xcur, w_in, wq, wkv, &kvoff, &cos_t, &sin_t, kc, vc, &masks_t, wo, w_post, &mut x2, &mut h2)).await?,
                        16 => launch(attn_block_c16, (&mut dev, &xcur, w_in, wq, wkv, &kvoff, &cos_t, &sin_t, kc, vc, &masks_t, wo, w_post, &mut x2, &mut h2)).await?,
                        _ => eyre::bail!("context {ctx} exceeds 8192"),
                    };
                    let t1 = tm!("A.attn", tc);
                    let (g, u, d, wr) = &dl[layer];
                    launch(shared_router, (&mut dev, &h2, &x2, g, u, d, wr, &mut y, &mut logits)).await?;
                    let t2 = tm!("A.shared", t1);
                    let lg: Vec<f32> = logits.to_host::<m![R]>(&mut dev.pdma).await?.into_vec();
                    let t3 = tm!("A.dl_logits", t2);
                    let hv: Vec<u16> = h2.to_host::<m![H]>(&mut dev.pdma).await?.into_vec().iter().map(|v| v.to_bits()).collect();
                    tm!("A.dl_h2", t3);
                    Ok((lg, hv))
                }.await;
                let _ = reply.send(r);
            }
            Cmd::Prefetch { layer } => {
                if ures && res_layer != Some(layer) {
                    let [ua, _] = &mut uy;
                    let r: eyre::Result<()> = async { ures_launch!(&mode_pf, layer, layer, &uy0, ua); Ok(()) }.await;
                    r?;
                    res_layer = Some(layer);
                    tm!("P.prefetch", tc);
                }
            }
            Cmd::LayerA2 { layer, inp, reply, s } => {
                use_stream!(s);
                let r: eyre::Result<(Vec<f32>, Vec<u16>)> = async {
                    HostTensor::<f32, Pack>::from_vec(inp).to_hbm(&mut dev.pdma).output(&mut inp_t).await?;
                    let t1 = tm!("L.up", tc);
                    let (w_in, wq, wkv, wo, w_post) = &attw[layer];
                    let (kc, vc) = &mut kvc[layer];
                    let (g, u, d, wr) = &dl[layer];
                    let (yi, yo) = if layer == 0 { (None, 0usize) } else { (Some(ylast), 1 - ylast) };
                    let [ya, yb] = &mut ybuf;
                    let (y_in, y_out): (&HbmTensor<bf16, ChipT, m![H]>, &mut HbmTensor<bf16, ChipT, m![H]>) = match (yi, yo) {
                        (None, _) => (&zero, ya),
                        (Some(0), _) => (&*ya, yb),
                        (Some(_), _) => (&*yb, ya),
                    };
                    let chunks = ctx.div_ceil(512).next_power_of_two();
                    if ures {
                        if res_layer != Some(layer) {   // not prefetched in time (layer 0 of a token, or first call)
                            let [ua, _] = &mut uy;
                            ures_launch!(&mode_pf, layer, layer, &uy0, ua, &mut *kc, &mut *vc);
                            res_layer = Some(layer);
                        }
                        let ys = if yi.is_none() { None } else { Some(ylast) };
                        let [ua, ub] = &mut uy;
                        let (y_in_u, y_out_u): (&HbmTensor<bf16, ChipT, m![U, H]>, &mut HbmTensor<bf16, ChipT, m![U, H]>) = match ys {
                            None => (&uy0, ua), Some(0) => (&*ua, ub), Some(_) => (&*ub, ua) };
                        ures_launch!(&mode_l, layer, layer, y_in_u, y_out_u, &mut *kc, &mut *vc);
                        let _ = (wq, wkv, wo, g, u, d, wr, w_in, w_post);
                    } else if unified {
                        let ys = if yi.is_none() { None } else { Some(ylast) };
                        let [ua, ub] = &mut uy;
                        let (y_in_u, y_out_u): (&HbmTensor<bf16, ChipT, m![U, H]>, &mut HbmTensor<bf16, ChipT, m![U, H]>) = match ys {
                            None => (&uy0, ua), Some(0) => (&*ua, ub), Some(_) => (&*ub, ua) };
                        let aw = AttnW { w_in, wq, wkv, wo, w_post };
                        let sw = SharedW { wg: g, wu: u, wd: d, wr };
                        let (tg, tu, td) = &tabs[layer];
                        let et = ExpertT { wg_t: tg, wu_t: tu, wd_t: td };
                        macro_rules! gu { ($f:ident) => { launch($f, (&mut dev, &mode_l, y_in_u, &mut out, &inp_t, &cos_t, &sin_t, kc, vc, &masks_t, aw, sw, et, &ids, &mut uouts, y_out_u, &mut uh2)).await? } }
                        match chunks { 1 => gu!(unified_a_c1), 2 => gu!(unified_a_c2), 4 => gu!(unified_a_c4), 8 => gu!(unified_a_c8), 16 => gu!(unified_a_c16),
                                       _ => eyre::bail!("context {ctx} exceeds 8192") };
                                        } else {
                        macro_rules! go { ($f:ident) => { launch($f, (&mut dev, y_in, &out, &inp_t, w_in, wq, wkv, &kvoff, &cos_t, &sin_t, kc, vc, &masks_t, wo, w_post, g, u, d, wr, &mut outp_t, &mut xcur, y_out, &mut h2)).await? } }
                        match chunks { 1 => go!(layer_a_c1), 2 => go!(layer_a_c2), 4 => go!(layer_a_c4), 8 => go!(layer_a_c8), 16 => go!(layer_a_c16),
                                       _ => eyre::bail!("context {ctx} exceeds 8192") };
                    }
                    ylast = yo;
                    let t2 = tm!("L.fused", t1);
                    let v: Vec<f32> = if unified { uouts_row0.to_host::<Pack>(&mut dev.pdma).await?.into_vec() }
                                      else { outp_t.to_host::<Pack>(&mut dev.pdma).await?.into_vec() };
                    tm!("L.dl", t2);
                    Ok((v[HN..HN + 128].to_vec(), v[..HN].iter().map(|&f| bf16::from_f32(f).to_bits()).collect()))
                }.await;
                let _ = reply.send(r);
            }
            Cmd::FinalLm { inp, reply, s } => {
                use_stream!(s);
                // lm_head's DM scratch does not fit next to the resident slots: release them for the head
                // (unless the head runs on chip B: LM_ON_B=1, then FinalLm only combines and returns x)
                let lm_on_b = std::env::var("LM_ON_B").map(|v| v == "1").unwrap_or(false);
                if ures && !lm_on_b { ra = None; rs = None; res_layer = None; }
                let r: eyre::Result<usize> = async {
                    // final combine with the standalone kernel: x = y_last + sum g·out_A + extra (bf16 extra)
                    let gv: Vec<f32> = inp[HN..HN + 8].to_vec();
                    let ex: Vec<bf16> = inp[..HN].iter().map(|&f| bf16::from_f32(f)).collect();
                    HostTensor::<f32, m![K]>::from_vec(gv).to_hbm(&mut dev.pdma).output(&mut gates).await?;
                    HostTensor::<bf16, m![H]>::from_vec(ex).to_hbm(&mut dev.pdma).output(&mut extra).await?;
                    let yl = if unified { &uy_row0[ylast] } else { &ybuf[ylast] };
                    launch(combine, (&mut dev, yl, &out, &gates, &extra, &mut xcur)).await?;
                    let (wlm, wf, iota) = lmw.as_ref().unwrap();
                    launch(lm_head_am, (&mut dev, &xcur, wf, wlm, iota, &mut lml, &mut keys)).await?;
                    let k: Vec<f32> = keys.to_host::<m![W / 512]>(&mut dev.pdma).await?.into_vec();
                    let mut best = (f32::MIN, usize::MAX);
                    for (s, &kv) in k.iter().enumerate() {
                        let bits = kv.to_bits();
                        let val = f32::from_bits(bits & 0xFFFF0000);
                        let row = s * 512 + 511 - (bits & 0x1FF) as usize;
                        if val > best.0 || (val == best.0 && row < best.1) { best = (val, row); }
                    }
                    Ok(best.1)
                }.await;
                let _ = reply.send(r);
                if ures {
                    ra = Some(AttnSlot::alloc(&dev)?); rs = Some(SharedSlot::alloc(&dev)?);
                    let [ua, _] = &mut uy;
                    let rr: eyre::Result<()> = async { ures_launch!(&mode_pf, 0, 0, &uy0, ua); Ok(()) }.await;
                    rr?;
                    res_layer = Some(0);
                }
            }
            Cmd::FinalX { inp, reply, s } => {
                use_stream!(s);
                let r: eyre::Result<Vec<u16>> = async {
                    let gv: Vec<f32> = inp[HN..HN + 8].to_vec();
                    let ex: Vec<bf16> = inp[..HN].iter().map(|&f| bf16::from_f32(f)).collect();
                    HostTensor::<f32, m![K]>::from_vec(gv).to_hbm(&mut dev.pdma).output(&mut gates).await?;
                    HostTensor::<bf16, m![H]>::from_vec(ex).to_hbm(&mut dev.pdma).output(&mut extra).await?;
                    let yl = if unified { &uy_row0[ylast] } else { &ybuf[ylast] };
                    launch(combine, (&mut dev, yl, &out, &gates, &extra, &mut xcur)).await?;
                    Ok(xcur.to_host::<m![H]>(&mut dev.pdma).await?.into_vec().iter().map(|v| v.to_bits()).collect())
                }.await;
                let _ = reply.send(r);
                // next token starts at layer 0: prefetch it now (off the critical path while chip B runs lm_head)
                if ures && res_layer != Some(0) {
                    let [ua, _] = &mut uy;
                    let rr: eyre::Result<()> = async { ures_launch!(&mode_pf, 0, 0, &uy0, ua); Ok(()) }.await;
                    rr?;
                    res_layer = Some(0);
                }
            }
            Cmd::ExpertsA2 { layer, h2: _hh, slots, s } => {
                use_stream!(s);
                // h2 is already resident: layer_a writes it to the `h2` HBM tensor
                ids.write(pack(&slots), &mut dev.pdma).await?;
                let t1 = tm!("EA.up", tc);
                if ures {
                    let [ua, _] = &mut uy;
                    ures_launch!(&mode_ffn, layer, layer, &uy0, ua);
                } else if unified {
                    let (w_in, wq, wkv, wo, w_post) = &attw[layer];
                    let (kc, vc) = &mut kvc[layer];
                    let (g, u, d, wr) = &dl[layer];
                    let (tg, tu, td) = &tabs[layer];
                    let aw = AttnW { w_in, wq, wkv, wo, w_post };
                    let sw = SharedW { wg: g, wu: u, wd: d, wr };
                    let et = ExpertT { wg_t: tg, wu_t: tu, wd_t: td };
                    let [ua, _] = &mut uy;
                    let chunks = ctx.div_ceil(512).next_power_of_two();
                    // same program as the layer launch (mode 0 = ffn_k8 body on h2s row 0); y/out rows untouched
                    macro_rules! gf { ($f:ident) => { launch($f, (&mut dev, &mode_f, &uy0, &mut out, &inp_t, &cos_t, &sin_t, kc, vc, &masks_t, aw, sw, et, &ids, &mut uouts, ua, &mut uh2)).await? } }
                    match chunks { 1 => gf!(unified_a_c1), 2 => gf!(unified_a_c2), 4 => gf!(unified_a_c4), 8 => gf!(unified_a_c8), _ => gf!(unified_a_c16) };
                } else {
                    let (g, u, d) = &tabs[layer];
                    launch(ffn_k8, (&mut dev, &h2, g, u, d, &ids, &mut out)).await?;
                }
                tm!("EA.ffn", t1);
            }
            Cmd::CombineA { gates: gv, extra: ex } => {
                HostTensor::<f32, m![K]>::from_vec(gv.to_vec()).to_hbm(&mut dev.pdma).output(&mut gates).await?;
                HostTensor::<bf16, m![H]>::from_vec(bf(&ex)).to_hbm(&mut dev.pdma).output(&mut extra).await?;
                let t1 = tm!("C.upload", tc);
                launch(combine, (&mut dev, &y, &out, &gates, &extra, &mut xcur)).await?;
                tm!("C.combine", t1);
            }
            Cmd::LmHead { reply } => {
                let r: eyre::Result<usize> = async {
                    let (wlm, wf, iota) = lmw.as_ref().unwrap();
                    launch(lm_head_am, (&mut dev, &xcur, wf, wlm, iota, &mut lml, &mut keys)).await?;
                    let k: Vec<f32> = keys.to_host::<m![W / 512]>(&mut dev.pdma).await?.into_vec();
                    let mut best = (f32::MIN, usize::MAX);
                    for (s, &kv) in k.iter().enumerate() {
                        let bits = kv.to_bits();
                        let val = f32::from_bits(bits & 0xFFFF0000);
                        let row = s * 512 + 511 - (bits & 0x1FF) as usize;
                        if val > best.0 || (val == best.0 && row < best.1) { best = (val, row); }
                    }
                    Ok(best.1)
                }.await;
                let _ = reply.send(r);
            }
            Cmd::LmHalfX { x, reply } => {
                let r: eyre::Result<(f32, usize)> = async {
                    let (wlm, wf, iota) = lmwh.as_ref().ok_or_else(|| eyre::eyre!("no half lm_head on this chip"))?;
                    HostTensor::<bf16, m![H]>::from_vec(bf(&x)).to_hbm(&mut dev.pdma).output(&mut xn).await?;
                    launch(lm_head_am_h, (&mut dev, &xn, wf, wlm, iota, &mut lmlh, &mut keysh)).await?;
                    let k: Vec<f32> = keysh.to_host::<m![WH / 512]>(&mut dev.pdma).await?.into_vec();
                    let (v, row) = best_key(&k);
                    Ok((v, row + WHN))
                }.await;
                let _ = reply.send(r);
            }
            Cmd::FinalSplit { inp, xreply, reply, s } => {
                use_stream!(s);
                let rx: eyre::Result<Vec<u16>> = async {
                    let gv: Vec<f32> = inp[HN..HN + 8].to_vec();
                    let ex: Vec<bf16> = inp[..HN].iter().map(|&f| bf16::from_f32(f)).collect();
                    HostTensor::<f32, m![K]>::from_vec(gv).to_hbm(&mut dev.pdma).output(&mut gates).await?;
                    HostTensor::<bf16, m![H]>::from_vec(ex).to_hbm(&mut dev.pdma).output(&mut extra).await?;
                    let yl = if unified { &uy_row0[ylast] } else { &ybuf[ylast] };
                    launch(combine, (&mut dev, yl, &out, &gates, &extra, &mut xcur)).await?;
                    Ok(xcur.to_host::<m![H]>(&mut dev.pdma).await?.into_vec().iter().map(|v| v.to_bits()).collect())
                }.await;
                let _ = xreply.send(rx);
                // the head's DM scratch does not fit next to the resident slots: release them for the head
                if ures { ra = None; rs = None; res_layer = None; }
                let r: eyre::Result<(f32, usize)> = async {
                    let (wlm, wf, iota) = lmwh.as_ref().ok_or_else(|| eyre::eyre!("no half lm_head on this chip"))?;
                    launch(lm_head_am_h, (&mut dev, &xcur, wf, wlm, iota, &mut lmlh, &mut keysh)).await?;
                    let k: Vec<f32> = keysh.to_host::<m![WH / 512]>(&mut dev.pdma).await?.into_vec();
                    Ok(best_key(&k))
                }.await;
                let _ = reply.send(r);
                if ures {
                    ra = Some(AttnSlot::alloc(&dev)?); rs = Some(SharedSlot::alloc(&dev)?);
                    let [ua, _] = &mut uy;
                    let rr: eyre::Result<()> = async { ures_launch!(&mode_pf, 0, 0, &uy0, ua); Ok(()) }.await;
                    rr?;
                    res_layer = Some(0);
                }
            }
            Cmd::LmHeadX { x, reply } => {
                let r: eyre::Result<usize> = async {
                    let (wlm, wf, iota) = lmw.as_ref().ok_or_else(|| eyre::eyre!("no lm_head weights on this chip"))?;
                    HostTensor::<bf16, m![H]>::from_vec(bf(&x)).to_hbm(&mut dev.pdma).output(&mut xn).await?;
                    launch(lm_head_am, (&mut dev, &xn, wf, wlm, iota, &mut lml, &mut keys)).await?;
                    let k: Vec<f32> = keys.to_host::<m![W / 512]>(&mut dev.pdma).await?.into_vec();
                    let mut best = (f32::MIN, usize::MAX);
                    for (s, &kv) in k.iter().enumerate() {
                        let bits = kv.to_bits();
                        let val = f32::from_bits(bits & 0xFFFF0000);
                        let row = s * 512 + 511 - (bits & 0x1FF) as usize;
                        if val > best.0 || (val == best.0 && row < best.1) { best = (val, row); }
                    }
                    Ok(best.1)
                }.await;
                let _ = reply.send(r);
            }
            Cmd::ExpertsB { layer, h2: hh, slots, gates: gv, reply, t0: tsend } => {
                tm!("B.wake", tsend);
                let r: eyre::Result<Vec<u16>> = async {
                    let fused = big && std::env::var("NO_FUSED_B").is_err();
                    if fused {
                        // ONE upload: h2 bits then the 8 gates as raw f32 bytes (2 bf16 slots each)
                        let mut v: Vec<bf16> = Vec::with_capacity(HN + 16);
                        v.extend(hh.iter().map(|&b| bf16::from_bits(b)));
                        for g in gv { let bits = g.to_bits(); v.push(bf16::from_bits(bits as u16)); v.push(bf16::from_bits((bits >> 16) as u16)); }
                        HostTensor::<bf16, m![PK]>::from_vec(v).to_hbm(&mut dev.pdma).output(&mut packed_all).await?;
                    } else {
                        HostTensor::<bf16, m![H]>::from_vec(bf(&hh)).to_hbm(&mut dev.pdma).output(&mut h2).await?;
                        HostTensor::<f32, m![K]>::from_vec(gv.to_vec()).to_hbm(&mut dev.pdma).output(&mut gates).await?;
                    }
                    ids.write(pack(&slots), &mut dev.pdma).await?;
                    let t1 = tm!("B.upload", tc);
                    let t3 = if fused {
                        let (g, u, d) = &tabs32[layer];
                        launch(experts_b, (&mut dev, &pk_x, &pk_g, g, u, d, &ids, &zeros_kh, &mut xn)).await?;
                        tm!("B.fused", t1)
                    } else {
                        if big { let (g, u, d) = &tabs32[layer]; launch(ffn32::ffn_k8, (&mut dev, &h2, g, u, d, &ids, &mut out)).await?; }
                        else { let (g, u, d) = &tabs[layer]; launch(ffn_k8, (&mut dev, &h2, g, u, d, &ids, &mut out)).await?; }
                        let t2 = tm!("B.ffn", t1);
                        launch(combine, (&mut dev, &zero, &out, &gates, &zero, &mut xn)).await?;
                        tm!("B.combine", t2)
                    };
                    let o = xn.to_host::<m![H]>(&mut dev.pdma).await?.into_vec().iter().map(|v| v.to_bits()).collect();
                    tm!("B.dl", t3);
                    Ok(o)
                }.await;
                let _ = reply.send(r);
                tm!("B.rtt", tsend);
            }
        }
    }
    let _ = (VN, &mut x2);
    Ok(())
}
