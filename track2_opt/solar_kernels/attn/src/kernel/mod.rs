//! Solar-Open-100B decode attention block on one RNGD chip (8 PEs).
//!
//! KV cache per layer: `[N, Cm * T, D]` bf16 = `Cm` chunks of T=512 tokens per kv head,
//! zero-initialized once by the host. Masks: `[Cm, T]` f32, 0 = valid, -1e30 = invalid (additive).
//!
//! Entry points (one decode step, one layer):
//!   * `attn_block_c{1,2,4,8,16}`: the whole block in one launch for contexts <= 512 * K tokens:
//!     rmsnorm -> q/k/v GEMV -> RoPE -> cache write -> attention over chunks 0..K -> o_proj ->
//!     residual -> post rmsnorm.
//!   * split path (any context): `attn_in` -> `attn_first` (chunk 0) -> `attn_next` (chunks
//!     1..n-1) -> `attn_out`.
use furiosa_opt_std::prelude::*;

pub(crate) mod attention;
pub(crate) mod epilogue;
pub mod gemv;
pub(crate) mod rmsnorm;
pub(crate) mod rope;

axes![
    H = 4096,
    Q = 8192,
    P = 1024,
    N = 8,
    G = 8,
    D = 128,
    T = 512,
    // GEMV chunking: Wq rows = Qc x R, [Wk; Wv] rows = R (= KV2 x N x D), Wo rows = Hc x Hr.
    R = 2048,
    Qc = 4,
    KV2 = 2,
    Hr = 1024,
    Hc = 4,
    // interleave axis for two-operand vector ops
    Ip = 2,
    // KV cache table: max chunks of T tokens per layer
    Cm = 16,
];

pub type Chip = m![1];

/// Single active cluster (cluster 1 is padding).
pub(crate) type C1 = m![1 # 2];
/// One kv head per 32-slice group.
pub(crate) type SliceN32 = m![N, 1 # 32];

use gemv::*;

/// One layer's K or V cache: `[N, Cm * T, D]` (head-major; chunk c = tokens c*512 .. c*512+511).
pub type KvCache = m![N, Cm, T, D];
/// Additive masks for every chunk of the cache table.
pub type Masks = m![Cm, T];

/// Writes one token row (`[N, D]`, one head per SliceN32 slice) into the cache at byte offset
/// `kv_offset` (= pos * D * 2) of each head's `[Cm * T, D]` plane.
fn store_kv(
    device: &mut Device,
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    val: DmTensor<bf16, Chip, C1, SliceN32, m![D]>,
    cache: &mut HbmTensor<bf16, Chip, KvCache>,
) {
    cache
        .view_mut()
        .scatter::<m![Cm, T], m![N, 1 # 32, D]>()
        .by_byte_offsets::<m![1]>(kv_offset.view())
        .from_dm(&mut device.tdma, val);
}

/// `v / rms` (the deferred rmsnorm factor).
fn scale_inv(
    device: &mut Device,
    v: &DmTensor<bf16, Chip, C1, SliceN32, m![D]>,
    rms_vrf: &VrfTensor<f32, Chip, C1, SliceN32, m![1 # 8]>,
) -> DmTensor<bf16, Chip, C1, SliceN32, m![D]> {
    device
        .main
        .begin(v.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![D / 4], m![D % 4]>()
        .vector_fp_div(rms_vrf)
        .vector_widen_concat::<m![D / 8], m![D % 8]>()
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit_trim::<m![D % 8]>()
        .commit()
}

/// Shared front half: rmsnorm, q/k/v GEMVs, RoPE, cache write. Returns q (after RoPE) with one
/// kv-head group per `SliceN32` slice.
#[expect(clippy::too_many_arguments)]
fn qkv_rope(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    w_in: &HbmTensor<bf16, Chip, m![H]>,
    wq: &HbmTensor<bf16, Chip, WQ>,
    wkv: &HbmTensor<bf16, Chip, WKV>,
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![D]>,
    sin: &HbmTensor<bf16, Chip, m![D]>,
    k_cache: &mut HbmTensor<bf16, Chip, KvCache>,
    v_cache: &mut HbmTensor<bf16, Chip, KvCache>,
) -> DmTensor<bf16, Chip, C1, SliceN32, m![G, D]> {
    let wkv: DmTensor<bf16, Chip, RC, RS, RElem> = wkv.to_dm(&mut device.tdma);
    // rms computed redundantly on each of the 8 head slices (x broadcast), no DM->DM hop
    let xs: DmTensor<bf16, Chip, C1, SliceN32, m![H]> = x.to_dm(&mut device.tdma);
    let rms: DmTensor<f32, Chip, C1, SliceN32, m![1 # 8]> = rmsnorm::rms_s1(device, &xs);
    let rms_vrf: VrfTensor<f32, Chip, C1, SliceN32, m![1 # 8]> =
        device.sub.begin(rms.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let u = rmsnorm::u_trf(device, x, w_in);

    let mut kv: DmTensor<bf16, Chip, RC, RSO, m![R % 512]> = DmTensor::new();
    gemv_r(device, &wkv, &u, kv.view_mut());
    let kv_hbm: HbmTensor<bf16, Chip, m![R]> = kv.to_hbm(&mut device.tdma);
    let kv3: HbmTensor<bf16, Chip, m![KV2, N, D]> = unsafe { kv_hbm.reshape() };
    let k: DmTensor<bf16, Chip, C1, SliceN32, m![D]> =
        kv3.view().tile::<m![KV2], 1, m![1 # 2, N, D]>(0).to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, C1, SliceN32, m![D]> =
        kv3.view().tile::<m![KV2], 1, m![1 # 2, N, D]>(1).to_dm(&mut device.tdma);
    let v = scale_inv(device, &v, &rms_vrf);
    store_kv(device, kv_offset, v, v_cache);

    let q = proj_q(device, wq, &u);
    // q row = qc * 2048 + c * 1024 + s * 512 + e  ==  n * 1024 + g * 128 + d
    let q: DmTensor<bf16, Chip, m![N % 2], m![G / 4, 1 # 128], m![N / 2, G % 4, D]> = unsafe { q.reshape() };
    let q: DmTensor<bf16, Chip, C1, SliceN32, m![G, D]> = q.to_dm(&mut device.tdma);

    let (q, k) = rope::apply_rope(device, &q, &k, cos, sin, &rms_vrf);
    store_kv(device, kv_offset, k, k_cache);
    q
}

/// rmsnorm -> q/k/v projections -> RoPE -> KV-cache write.
///
/// * `x`, `w_in`: `[H]` bf16.
/// * `wq`: [`WQ`] (host: `host::perm_wq`), `wkv`: [`WKV`] (host: `host::perm_wkv`).
/// * `kv_offset`: `pos * D * 2` (byte offset of the token inside each head's `[Cm * T, D]` plane).
/// * `cos`: `[D]`, `sin`: `[D]` sign-folded (`host::rope_sin_signed`), bf16, for this position.
/// * `k_cache`, `v_cache`: [`KvCache`] tables.
/// * `q_out`: `[N, G, D]` (= q head `n * G + g`), after RoPE.
#[device(chip = 1, pe = 8)]
#[expect(clippy::too_many_arguments)]
pub fn attn_in(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    w_in: &HbmTensor<bf16, Chip, m![H]>,
    wq: &HbmTensor<bf16, Chip, WQ>,
    wkv: &HbmTensor<bf16, Chip, WKV>,
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![D]>,
    sin: &HbmTensor<bf16, Chip, m![D]>,
    k_cache: &mut HbmTensor<bf16, Chip, KvCache>,
    v_cache: &mut HbmTensor<bf16, Chip, KvCache>,
    q_out: &mut HbmTensor<bf16, Chip, m![N, G, D]>,
) {
    let q = qkv_rope(device, x, w_in, wq, wkv, kv_offset, cos, sin, k_cache, v_cache);
    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
}

/// Attention over chunk `chunk` of the cache, initializing the running state:
/// `max`, `sum`: `[N, G]` f32, `out`: `[N, G, D]` bf16 (unnormalized).
#[device(chip = 1, pe = 8)]
#[expect(clippy::too_many_arguments)]
pub fn attn_first(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![N, G, D]>,
    k_cache: &HbmTensor<bf16, Chip, KvCache>,
    v_cache: &HbmTensor<bf16, Chip, KvCache>,
    masks: &HbmTensor<f32, Chip, Masks>,
    chunk: &HbmScalar<usize>,
    max_hbm: &mut HbmTensor<f32, Chip, m![N, G]>,
    sum_hbm: &mut HbmTensor<f32, Chip, m![N, G]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![N, G, D]>,
) {
    let c = chunk.to_spm(&mut device.tdma);
    let q = attention::q_trf_from_hbm(device, q);
    let tbl = attention::mask_table(device, masks);
    let mask = attention::mask_vrf(device, &tbl, c);
    let (max, sum, out) = attention::first_dm(device, &q, kv_chunk(k_cache, c), kv_chunk(v_cache, c), &mask);
    max.view().to_hbm_view(&mut device.tdma, max_hbm.view_mut());
    sum.view().to_hbm_view(&mut device.tdma, sum_hbm.view_mut());
    out.view().to_hbm_view(&mut device.tdma, out_hbm.view_mut());
}

/// Attention over chunk `chunk`, folded into the running (online-softmax) state in HBM.
#[device(chip = 1, pe = 8)]
#[expect(clippy::too_many_arguments)]
pub fn attn_next(
    device: &mut Device,
    q: &HbmTensor<bf16, Chip, m![N, G, D]>,
    k_cache: &HbmTensor<bf16, Chip, KvCache>,
    v_cache: &HbmTensor<bf16, Chip, KvCache>,
    masks: &HbmTensor<f32, Chip, Masks>,
    chunk: &HbmScalar<usize>,
    max_hbm: &mut HbmTensor<f32, Chip, m![N, G]>,
    sum_hbm: &mut HbmTensor<f32, Chip, m![N, G]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![N, G, D]>,
) {
    let c = chunk.to_spm(&mut device.tdma);
    let q = attention::q_trf_from_hbm(device, q);
    let tbl = attention::mask_table(device, masks);
    let mask = attention::mask_vrf(device, &tbl, c);
    let (old_max, old_sum, old_out) = attention::load_state(device, max_hbm, sum_hbm, out_hbm);
    let (max, sum, out) = attention::next_dm(
        device,
        &q,
        kv_chunk(k_cache, c),
        kv_chunk(v_cache, c),
        &mask,
        &old_max,
        &old_sum,
        &old_out,
    );
    max.view().to_hbm_view(&mut device.tdma, max_hbm.view_mut());
    sum.view().to_hbm_view(&mut device.tdma, sum_hbm.view_mut());
    out.view().to_hbm_view(&mut device.tdma, out_hbm.view_mut());
}

fn kv_chunk(cache: &HbmTensor<bf16, Chip, KvCache>, c: usize) -> HbmTensorView<'_, bf16, Chip, m![N, 1 # 16, T, D]> {
    cache.view().tile::<m![Cm], 1, m![N, 1 # 16, T, D]>(c)
}

/// out/sum -> o_proj -> x2 = x + o -> h2 = rmsnorm(x2, w_post).
///
/// * `attn`: `[N, G, D]` unnormalized attention output, `sum`: `[N, G]` softmax denominators.
/// * `wo`: [`WO`] (host: `host::perm_wo`).
#[device(chip = 1, pe = 8)]
#[expect(clippy::too_many_arguments)]
pub fn attn_out(
    device: &mut Device,
    attn: &HbmTensor<bf16, Chip, m![N, G, D]>,
    sum: &HbmTensor<f32, Chip, m![N, G]>,
    x: &HbmTensor<bf16, Chip, m![H]>,
    wo: &HbmTensor<bf16, Chip, WO>,
    w_post: &HbmTensor<bf16, Chip, m![H]>,
    x2_out: &mut HbmTensor<bf16, Chip, m![H]>,
    h2_out: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let sum: attention::MaxSum = sum.to_dm(&mut device.tdma);
    let out: attention::Out = attn.to_dm(&mut device.tdma);
    let a = attention::norm_dm(device, &out, &sum);
    o_proj_epilogue(device, a, x, wo, w_post, x2_out, h2_out);
}

/// `a` (normalized attention, one kv head per SN slice) -> o_proj -> residual -> rmsnorm.
fn o_proj_epilogue(
    device: &mut Device,
    a: DmTensor<bf16, Chip, attention::CA, attention::SN, m![G, D]>,
    x: &HbmTensor<bf16, Chip, m![H]>,
    wo: &HbmTensor<bf16, Chip, WO>,
    w_post: &HbmTensor<bf16, Chip, m![H]>,
    x2_out: &mut HbmTensor<bf16, Chip, m![H]>,
    h2_out: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let a: DmTensor<bf16, Chip, m![Q / 4096], m![Q / 1024 % 4, 1 # 64], m![Q % 1024]> = unsafe { a.reshape() };
    let a: DmTensor<bf16, Chip, OC, OS, m![Q % 32]> = a.to_dm(&mut device.tdma);
    let a_trf: TrfTensor<bf16, Chip, OC, OS, m![1], m![Q % 32]> =
        device.sub.begin(a.view()).fetch::<m![1], m![Q % 32]>().collect::<m![Q / 16 % 2], m![Q % 16]>().to_trf();
    let o = proj_o(device, wo, &a_trf);
    // o row = hc * 1024 + c * 512 + e
    let o: DmTensor<bf16, Chip, epilogue::EC, OSO, epilogue::EH> = unsafe { o.reshape() };
    epilogue::residual_rmsnorm(device, &o, x, w_post, x2_out, h2_out);
}

/// Whole block for contexts <= 512 * K: front half, attention over chunks 0..K, back half.
macro_rules! attn_block_k {
    ($name:ident, [$($c:literal),*]) => {
        /// Whole attention block in one launch for contexts `<= 512 *` K tokens (K in the name):
        /// the current token is written at `kv_offset` (= pos * D * 2) and attended to in the
        /// same launch; chunks `0..K` are attended with `masks` (fully masked chunks contribute 0).
        #[device(chip = 1, pe = 8)]
        #[expect(clippy::too_many_arguments)]
        pub fn $name(
            device: &mut Device,
            x: &HbmTensor<bf16, Chip, m![H]>,
            w_in: &HbmTensor<bf16, Chip, m![H]>,
            wq: &HbmTensor<bf16, Chip, WQ>,
            wkv: &HbmTensor<bf16, Chip, WKV>,
            kv_offset: &HbmTensor<i32, Chip, m![1]>,
            cos: &HbmTensor<bf16, Chip, m![D]>,
            sin: &HbmTensor<bf16, Chip, m![D]>,
            k_cache: &mut HbmTensor<bf16, Chip, KvCache>,
            v_cache: &mut HbmTensor<bf16, Chip, KvCache>,
            masks: &HbmTensor<f32, Chip, Masks>,
            wo: &HbmTensor<bf16, Chip, WO>,
            w_post: &HbmTensor<bf16, Chip, m![H]>,
            x2_out: &mut HbmTensor<bf16, Chip, m![H]>,
            h2_out: &mut HbmTensor<bf16, Chip, m![H]>,
        ) {
            let q = qkv_rope(device, x, w_in, wq, wkv, kv_offset, cos, sin, k_cache, v_cache);
            let q = attention::q_trf_from_dm(device, &q);
            let tbl = attention::mask_table(device, masks);
            let mask = attention::mask_vrf(device, &tbl, 0);
            #[allow(unused_mut)]
            let (mut max, mut sum, mut out) =
                attention::first_dm(device, &q, kv_chunk(k_cache, 0), kv_chunk(v_cache, 0), &mask);
            $(
                let mask = attention::mask_vrf(device, &tbl, $c);
                let (m2, s2, o2) = attention::next_dm(
                    device,
                    &q,
                    kv_chunk(k_cache, $c),
                    kv_chunk(v_cache, $c),
                    &mask,
                    &max,
                    &sum,
                    &out,
                );
                max = m2;
                sum = s2;
                out = o2;
            )*
            let _ = &max;
            let a = attention::norm_dm(device, &out, &sum);
            o_proj_epilogue(device, a, x, wo, w_post, x2_out, h2_out);
        }
    };
}

attn_block_k!(attn_block_c1, []);
attn_block_k!(attn_block_c2, [1]);
attn_block_k!(attn_block_c4, [1, 2, 3]);
attn_block_k!(attn_block_c8, [1, 2, 3, 4, 5, 6, 7]);
attn_block_k!(attn_block_c16, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
