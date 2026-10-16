//! Host-side model: safetensors mmap weights, bf16 helpers, CPU kernels (FFI), reference ops.
use memmap2::Mmap;
use std::collections::HashMap;

pub const H: usize = 4096;
pub const NQ: usize = 64;
pub const NKV: usize = 8;
pub const D: usize = 128;
pub const QD: usize = NQ * D; // 8192
pub const KVD: usize = NKV * D; // 1024
pub const F: usize = 1280;
pub const E: usize = 128;
pub const TOPK: usize = 8;
pub const L: usize = 48;
pub const VOCAB: usize = 196608;
pub const EPS: f32 = 1e-5;

unsafe extern "C" {
    fn ck_set_threads(n: i32);
    pub fn ck_mmio_read(src: *const u8, dst: *mut u8, len: usize);
    pub fn ck_mmio_set_threads(n: i32);
    fn ck_gemv(w: *const u16, x: *const u16, y: *mut f32, m: i32, k: i32);
    fn ck_moe2(e: i32, g: *const *const u16, u: *const *const u16, d: *const *const u16, w: *const f32,
               x: *const u16, out: *mut f32, h: i32, f: i32);
    fn ck_place_expert(g: *const u16, u: *const u16, d: *const u16, dst: *mut u16, h: i32, f: i32);
    fn ck_moe3(e: i32, base: *const *const u16, w: *const f32, x: *const u16, out: *mut f32, h: i32, f: i32);
    fn ck_read_bw(n: usize, iters: i32) -> f64;
    fn ck_place_expert_q8(g: *const u16, u: *const u16, d: *const u16, q: *mut i8, sc: *mut f32, rs: *mut i32, h: i32, f: i32);
    fn ck_moe_q8(e: i32, q: *const *mut i8, sc: *const *mut f32, rs: *const *mut i32, w: *const f32, x: *const u16,
                 out: *mut f32, h: i32, f: i32);
    fn ck_moe_q8w(e: i32, q: *const *mut i8, sc: *const *mut f32, w: *const f32, x: *const u16, out: *mut f32, h: i32, f: i32);
    fn ck_moe_q8x(e: i32, q: *const *mut i8, sc: *const *mut f32, rs: *const *mut i32, w: *const f32, x: *const u16,
                  out: *mut f32, h: i32, f: i32);
    fn ck_touch_experts(e: i32, base: *const *const u16, h: i32, f: i32) -> f32;
    fn ck_moe1(e: i32, g: *const *const u16, u: *const *const u16, d: *const *const u16, w: *const f32,
               x: *const u16, out: *mut f32, h: i32, f: i32);
}

pub fn set_threads(n: usize) { unsafe { ck_set_threads(n as i32) } }

#[inline]
pub fn f2bf(f: f32) -> u16 {
    let u = f.to_bits();
    let u = u.wrapping_add(0x7FFF + ((u >> 16) & 1));
    (u >> 16) as u16
}
#[inline]
pub fn bf2f(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }
pub fn round_bf(v: &mut [f32]) { for x in v.iter_mut() { *x = bf2f(f2bf(*x)); } }
pub fn to_bf(v: &[f32]) -> Vec<u16> { v.iter().map(|&x| f2bf(x)).collect() }

pub fn gemv(w: &[u16], x: &[u16], m: usize, k: usize) -> Vec<f32> {
    assert_eq!(w.len(), m * k);
    assert_eq!(x.len(), k);
    let mut y = vec![0f32; m];
    unsafe { ck_gemv(w.as_ptr(), x.as_ptr(), y.as_mut_ptr(), m as i32, k as i32) };
    y
}

/// out += sum_e w_e * expert_e(x)  (x: bf16 [H])
pub fn moe_cpu(experts: &[(&[u16], &[u16], &[u16])], w: &[f32], x: &[u16], out: &mut [f32]) {
    if experts.is_empty() { return; }
    let g: Vec<*const u16> = experts.iter().map(|e| e.0.as_ptr()).collect();
    let u: Vec<*const u16> = experts.iter().map(|e| e.1.as_ptr()).collect();
    let d: Vec<*const u16> = experts.iter().map(|e| e.2.as_ptr()).collect();
    let f = if std::env::var("MOE_V1").is_ok() { ck_moe1 } else { ck_moe2 };
    unsafe { f(experts.len() as i32, g.as_ptr(), u.as_ptr(), d.as_ptr(), w.as_ptr(), x.as_ptr(),
                     out.as_mut_ptr(), H as i32, F as i32) };
}

/// HF SolarOpenRMSNorm: fp32 normalize, cast to bf16, multiply by bf16 weight -> bf16.
pub fn rmsnorm(x: &[f32], w: &[u16]) -> Vec<u16> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + EPS).sqrt();
    x.iter().zip(w).map(|(&v, &wb)| f2bf(bf2f(f2bf(v * r)) * bf2f(wb))).collect()
}

pub struct Weights {
    tensors: HashMap<String, (&'static [u8], Vec<usize>)>,
}

impl Weights {
    pub fn open(dir: &str) -> eyre::Result<Self> {
        let mut tensors = HashMap::new();
        let mut files: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false)).collect();
        files.sort();
        for p in files {
            let f = std::fs::File::open(&p)?;
            let mm: &'static Mmap = Box::leak(Box::new(unsafe { Mmap::map(&f)? }));
            let st = safetensors::SafeTensors::deserialize(&mm[..])?;
            for (name, view) in st.tensors() {
                let data: &'static [u8] = unsafe { std::slice::from_raw_parts(view.data().as_ptr(), view.data().len()) };
                tensors.insert(name, (data, view.shape().to_vec()));
            }
        }
        Ok(Self { tensors })
    }
    pub fn raw(&self, name: &str) -> &'static [u8] {
        self.tensors.get(name).unwrap_or_else(|| panic!("missing tensor {name}")).0
    }
    /// bf16 tensor as u16 slice (zero copy).
    pub fn bf(&self, name: &str) -> &'static [u16] {
        let r = self.raw(name);
        unsafe { std::slice::from_raw_parts(r.as_ptr() as *const u16, r.len() / 2) }
    }
    pub fn f32v(&self, name: &str) -> Vec<f32> {
        let r = self.raw(name);
        r.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
    pub fn layer(&self, l: usize, n: &str) -> &'static [u16] { self.bf(&format!("model.layers.{l}.{n}")) }
    pub fn expert(&self, l: usize, e: usize) -> (&'static [u16], &'static [u16], &'static [u16]) {
        let p = format!("model.layers.{l}.mlp.experts.{e}.");
        (self.bf(&(p.clone() + "gate_proj.weight")), self.bf(&(p.clone() + "up_proj.weight")),
         self.bf(&(p + "down_proj.weight")))
    }
}

/// sigmoid router + correction bias + top-k + normalize (n_group = 1). Returns (ids, weights).
pub fn route(logits: &[f32], bias: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let s: Vec<f32> = logits.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
    let mut idx: Vec<usize> = (0..E).collect();
    idx.sort_by(|&a, &b| (s[b] + bias[b]).partial_cmp(&(s[a] + bias[a])).unwrap());
    idx.truncate(TOPK);
    let w: Vec<f32> = idx.iter().map(|&i| s[i]).collect();
    let tot: f32 = w.iter().sum::<f32>() + 1e-20;
    (idx, w.iter().map(|v| v / tot).collect())
}

pub fn read_bf16_file(path: &str) -> eyre::Result<Vec<u16>> {
    let b = std::fs::read(path)?;
    Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

/// Pin the calling thread to cores [lo, hi].
pub fn pin_current(lo: usize, hi: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for c in lo..=hi { libc::CPU_SET(c, &mut set); }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

/// Allocate + NUMA-place one expert (pages first-touched by the OpenMP thread that will read them).
pub fn place_expert(g: &[u16], u: &[u16], d: &[u16]) -> Box<[u16]> {
    let n = 3 * F * H;
    let mut v: Vec<u16> = Vec::with_capacity(n);
    unsafe {
        ck_place_expert(g.as_ptr(), u.as_ptr(), d.as_ptr(), v.as_mut_ptr(), H as i32, F as i32);
        v.set_len(n);
    }
    v.into_boxed_slice()
}

/// out += sum_e w_e * expert_e(x) for NUMA-placed experts (ck_place_expert layout).
pub fn moe_cpu_placed(bases: &[*const u16], w: &[f32], x: &[u16], out: &mut [f32]) {
    if bases.is_empty() { return; }
    unsafe { ck_moe3(bases.len() as i32, bases.as_ptr(), w.as_ptr(), x.as_ptr(), out.as_mut_ptr(), H as i32, F as i32) };
}

pub fn read_bw(n_per_thread: usize, iters: i32) -> f64 { unsafe { ck_read_bw(n_per_thread, iters) } }

/// int8 (per-row symmetric) copy of one expert, NUMA-placed like place_expert. Approximate.
pub struct Q8 { pub q: Box<[i8]>, pub sc: Box<[f32]>, pub rs: Box<[i32]> }
unsafe impl Send for Q8 {}
unsafe impl Sync for Q8 {}

pub fn place_expert_q8(g: &[u16], u: &[u16], d: &[u16]) -> Q8 {
    let rows = 2 * F + H;
    let nq = 2 * F * H + H * F;
    let mut q: Vec<i8> = Vec::with_capacity(nq);
    let mut sc = vec![0f32; rows];
    let mut rs = vec![0i32; rows];
    unsafe {
        ck_place_expert_q8(g.as_ptr(), u.as_ptr(), d.as_ptr(), q.as_mut_ptr(), sc.as_mut_ptr(), rs.as_mut_ptr(), H as i32, F as i32);
        q.set_len(nq);
    }
    Q8 { q: q.into_boxed_slice(), sc: sc.into_boxed_slice(), rs: rs.into_boxed_slice() }
}

pub fn moe_cpu_q8(ex: &[&Q8], w: &[f32], x: &[u16], out: &mut [f32]) {
    if ex.is_empty() { return; }
    let q: Vec<*mut i8> = ex.iter().map(|e| e.q.as_ptr() as *mut i8).collect();
    let sc: Vec<*mut f32> = ex.iter().map(|e| e.sc.as_ptr() as *mut f32).collect();
    let rs: Vec<*mut i32> = ex.iter().map(|e| e.rs.as_ptr() as *mut i32).collect();
    if std::env::var("Q8_MODE").map(|v| v == "w").unwrap_or(false) {
        let _ = &rs;
        unsafe { ck_moe_q8w(ex.len() as i32, q.as_ptr(), sc.as_ptr(), w.as_ptr(), x.as_ptr(), out.as_mut_ptr(), H as i32, F as i32) };
    } else if std::env::var("Q8_MODE").map(|v| v != "x16").unwrap_or(false) || std::env::var("Q8_ACT").is_ok() {
        unsafe { ck_moe_q8(ex.len() as i32, q.as_ptr(), sc.as_ptr(), rs.as_ptr(), w.as_ptr(), x.as_ptr(), out.as_mut_ptr(), H as i32, F as i32) };
    } else {
        // default: int8 weights x 16-bit activations (two VNNI passes) — weight-quantization error only
        unsafe { ck_moe_q8x(ex.len() as i32, q.as_ptr(), sc.as_ptr(), rs.as_ptr(), w.as_ptr(), x.as_ptr(), out.as_mut_ptr(), H as i32, F as i32) };
    }
}

/// Pull NUMA-placed cold experts into the reading threads' caches (exact; see ck_touch_experts).
pub fn touch_experts(bases: &[*const u16]) -> f32 {
    if bases.is_empty() { return 0.0; }
    unsafe { ck_touch_experts(bases.len() as i32, bases.as_ptr(), H as i32, F as i32) }
}
