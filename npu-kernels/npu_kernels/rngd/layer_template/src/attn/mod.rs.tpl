//! Decode attention block + router on one RNGD chip (8 PEs), generated for H=${H}, ${N} kv heads x ${G} q heads,
//! head dim ${D}, q/k norm ${QK_NORM}.
//!
//! KV cache per layer: `[N, Cm * T, D]` bf16 = `Cm` chunks of T=${T} tokens per kv head, zero-initialized once by the
//! host. Masks: `[Cm, T]` f32, 0 = valid, -1e30 = invalid (additive).
//!
//! `attn_block_c{K}`: the whole block in one launch for contexts <= ${T} * K tokens:
//!   rmsnorm -> q/k/v GEMV -> [q/k head norm] -> RoPE -> cache write -> attention over chunks 0..K -> o_proj ->
//!   residual -> post rmsnorm -> router logits.
use furiosa_opt_std::prelude::*;

pub(crate) mod attention;
pub mod epilogue;
pub mod gemv;
pub mod rmsnorm;
pub(crate) mod rope;

axes![
    Q = ${Q},
    N = ${N},
    G = ${G},
    D = ${D},
    T = ${T},
    // GEMV chunking: Wq rows = Qc x RW, [Wk; Wv] rows = RW (= KV2 x N x D), Wo rows = Hc x Hr.
    RW = ${RW},
    Qc = ${Qc},
    KV2 = 2,
    Hr = ${Hr},
    Hc = ${Hc},
    // interleave axis for two-operand vector ops (also: "both clusters")
    Ip = 2,
    // KV cache table: max chunks of T tokens per layer
    Cm = ${Cm},
];

pub use crate::axes::{Chip, H, NE};

/// Single active cluster (cluster 1 is padding).
pub(crate) type C1 = m![1 # 2];
/// One kv head per ${SPN}-slice group.
pub(crate) type SliceN = m![N, 1 # ${SPN}];
/// One q head per ${SPH}-slice group.
pub(crate) type SliceQ = m![N, G, 1 # ${SPH}];

use gemv::*;

/// One layer's K or V cache: `[N, Cm * T, D]` (head-major; chunk c = tokens c*T .. c*T+T-1).
pub type KvCache = m![N, Cm, T, D];
/// Additive masks for every chunk of the cache table.
pub type Masks = m![Cm, T];

/// Writes one token row (`[N, D]`, one head per SliceN group) into the cache at byte offset `kv_offset`
/// (= pos * D * 2) of each head's `[Cm * T, D]` plane.
pub(crate) fn store_kv(
    device: &mut Device,
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    val: DmTensor<bf16, Chip, C1, SliceN, m![D]>,
    cache: &mut HbmTensor<bf16, Chip, KvCache>,
) {
    cache
        .view_mut()
        .scatter::<m![Cm, T], m![N, 1 # ${SPN}, D]>()
        .by_byte_offsets::<m![1]>(kv_offset.view())
        .from_dm(&mut device.tdma, val);
}

/// `v / rms` (the deferred rmsnorm factor), one head per slice.
fn scale_inv<S: M>(
    device: &mut Device,
    v: &DmTensor<bf16, Chip, C1, S, m![D]>,
    rms_vrf: &VrfTensor<f32, Chip, C1, S, m![1 # 8]>,
) -> DmTensor<bf16, Chip, C1, S, m![D]> {
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

/// Front half: rmsnorm, q/k/v GEMVs, [q/k head norm], RoPE, cache write. Returns q `[N, G, D]` (after RoPE) in HBM.
#[expect(clippy::too_many_arguments)]
pub(crate) fn qkv_rope(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    w_in: &HbmTensor<bf16, Chip, m![H]>,
    wq: &HbmTensor<bf16, Chip, WQ>,
    wkv: &HbmTensor<bf16, Chip, WKV>,${QK_PARAMS}
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![D]>,
    sin: &HbmTensor<bf16, Chip, m![D]>,
    k_cache: &mut HbmTensor<bf16, Chip, KvCache>,
    v_cache: &mut HbmTensor<bf16, Chip, KvCache>,
) -> HbmTensor<bf16, Chip, m![N, G, D]> {
    let wkv: DmTensor<bf16, Chip, RC, RS, RElem> = wkv.to_dm(&mut device.tdma);
    // rms computed redundantly on each kv-head slice (x broadcast), no DM->DM hop
    let xs: DmTensor<bf16, Chip, C1, SliceN, m![H]> = x.to_dm(&mut device.tdma);
    let rms: DmTensor<f32, Chip, C1, SliceN, m![1 # 8]> = rmsnorm::rms_s1(device, &xs);
    let rms_vrf: VrfTensor<f32, Chip, C1, SliceN, m![1 # 8]> =
        device.sub.begin(rms.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let u = rmsnorm::u_trf(device, x, w_in);

    let mut kv: DmTensor<bf16, Chip, RC, RSO, m![RW % ${RPS}]> = DmTensor::new();
    gemv_r(device, &wkv, &u, kv.view_mut());
    let kv_hbm: HbmTensor<bf16, Chip, m![RW]> = kv.to_hbm(&mut device.tdma);
    let kv3: HbmTensor<bf16, Chip, m![KV2, N, D]> = unsafe { kv_hbm.reshape() };
    let k: DmTensor<bf16, Chip, C1, SliceN, m![D]> =
        kv3.view().tile::<m![KV2], 1, m![1 # 2, N, D]>(0).to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, C1, SliceN, m![D]> =
        kv3.view().tile::<m![KV2], 1, m![1 # 2, N, D]>(1).to_dm(&mut device.tdma);
    let v = scale_inv(device, &v, &rms_vrf);
    store_kv(device, kv_offset, v, v_cache);

    let q = proj_q(device, wq, &u);
    // q row = qc * RW + rw == n * G * D + g * D + d: two plain DMAs through HBM
    let q_hbm: HbmTensor<bf16, Chip, m![Qc, RW]> = q.to_hbm(&mut device.tdma);
    let q3: HbmTensor<bf16, Chip, m![N, G, D]> = unsafe { q_hbm.reshape() };
${QK_FRONT}
}

pub(crate) fn kv_chunk(cache: &HbmTensor<bf16, Chip, KvCache>, c: usize) -> HbmTensorView<'_, bf16, Chip, m![N, 1 # ${Cm}, T, D]> {
    cache.view().tile::<m![Cm], 1, m![N, 1 # ${Cm}, T, D]>(c)
}

/// `a` (normalized attention, one kv head per SN slice) -> o_proj -> residual -> rmsnorm; (x2, h2) in DM.
pub(crate) fn o_proj_epilogue(
    device: &mut Device,
    a: DmTensor<bf16, Chip, attention::CA, attention::SN, m![G, D]>,
    x: &HbmTensor<bf16, Chip, m![H]>,
    wo: &HbmTensor<bf16, Chip, WO>,
    w_post: &HbmTensor<bf16, Chip, m![H]>,
) -> (
    DmTensor<bf16, Chip, epilogue::EC, epilogue::ES, epilogue::EH>,
    DmTensor<bf16, Chip, epilogue::EC, epilogue::ES, epilogue::EH>,
) {
    let a: DmTensor<bf16, Chip, m![Q / ${QH}], m![Q / ${GD} % ${NPC}, 1 # ${SNP}], m![Q % ${GD}]> = unsafe { a.reshape() };
    let a: DmTensor<bf16, Chip, OC, OS, m![Q % 32]> = a.to_dm(&mut device.tdma);
    let a_trf: TrfTensor<bf16, Chip, OC, OS, m![1], m![Q % 32]> =
        device.sub.begin(a.view()).fetch::<m![1], m![Q % 32]>().collect::<m![Q / 16 % 2], m![Q % 16]>().to_trf();
    let o = proj_o(device, wo, &a_trf);
    // o row = hc * Hr + r == h: through HBM into the epilogue layout (H halves per cluster)
    let o_hbm: HbmTensor<bf16, Chip, m![Hc, Hr]> = o.to_hbm(&mut device.tdma);
    let o_h: HbmTensor<bf16, Chip, m![H]> = unsafe { o_hbm.reshape() };
    let o: DmTensor<bf16, Chip, epilogue::EC, epilogue::ES, epilogue::EH> = o_h.to_dm(&mut device.tdma);
    epilogue::residual_rmsnorm_dm(device, &o, x, w_post)
}

/// Router logits `Wr . h2` from h2 in DM.
pub(crate) fn router_dm(
    device: &mut Device,
    h2: &DmTensor<bf16, Chip, epilogue::EC, epilogue::ES, epilogue::EH>,
    wr: &HbmTensor<bf16, Chip, WR>,
) -> DmTensor<bf16, Chip, GC, GSO, m![NE % ${RR}]> {
    let w: DmTensor<bf16, Chip, GC, GS, GElem> = wr.to_dm(&mut device.tdma);
    let hx: DmTensor<bf16, Chip, GC, GS, m![H % 32]> = h2.to_dm(&mut device.tdma);
    let x_trf: TrfTensor<bf16, Chip, GC, GS, m![1], m![H % 32]> =
        device.sub.begin(hx.view()).fetch::<m![1], m![H % 32]>().collect::<m![H / 16 % 2], m![H % 16]>().to_trf();
    gemv_router(device, &w, &x_trf)
}
${BLOCKS}
