//! Decode attention over one T=512 KV chunk with GQA (G=8 q heads per kv head).
//!
//! Layouts:
//! Cluster = `N / 4` (4 kv heads per cluster).
//! * `SA = [N % 4, 1 # 2, T / 16]`: 16 tokens per slice. Scores/softmax live as `[T % 16, G]` (G in the
//!   packet, so row reductions run over Time; no f32 transpose); `P @ V` with V transposed into TRF.
//! * `SN = [N % 4, 1 # 64]`: the head slice of each kv head: per-head reductions (inter-slice reduce
//!   over the 32 token-slices) land here, as do the running max / sum / out.
//! The row max is broadcast back from SN to SA with one small DM->DM DMA.
use furiosa_opt_std::prelude::*;

use super::{Chip, Cm, D, G, N, T};

/// kv heads 0-3 on cluster 0, 4-7 on cluster 1.
pub(crate) type CA = m![N / 4];

pub(crate) type SN = m![N % 4, 1 # 64];
type SA = m![N % 4, 1 # 2, T / 16];

/// `sqrt(D)`.
const SQRT_HEAD_DIM: f32 = 11.313_708;

fn g_to_vrf(device: &mut Device, x: &DmTensor<f32, Chip, CA, SN, m![G]>) -> VrfTensor<f32, Chip, CA, SN, m![G]> {
    device.sub.begin(x.view()).fetch::<m![1], m![G]>().collect::<m![1], m![G]>().to_vrf()
}

pub(crate) type QTrf = TrfTensor<bf16, Chip, CA, SA, m![G], m![D]>;

/// q (one kv-head group per SliceN32 slice) broadcast DM->DM to its token-slices, into TRF.
pub(crate) fn q_trf_from_dm(
    device: &mut Device,
    q: &DmTensor<bf16, Chip, super::C1, super::SliceN32, m![G, D]>,
) -> QTrf {
    let q: DmTensor<bf16, Chip, CA, SA, m![G, D]> = q.to_dm(&mut device.tdma);
    device.sub.begin(q.view()).fetch::<m![G, D / 16], m![D % 16]>().collect::<m![G, D / 16], m![D % 16]>().to_trf()
}

/// q `[N, G, D]` broadcast to the 32 token-slices of its kv head, into TRF.
pub(crate) fn q_trf_from_hbm(device: &mut Device, q: &HbmTensor<bf16, Chip, m![N, G, D]>) -> QTrf {
    let q: DmTensor<bf16, Chip, CA, SA, m![G, D]> = q.to_dm(&mut device.tdma);
    device.sub.begin(q.view()).fetch::<m![G, D / 16], m![D % 16]>().collect::<m![G, D / 16], m![D % 16]>().to_trf()
}

pub(crate) type MaskVrf = VrfTensor<f32, Chip, CA, SA, m![T % 16]>;
/// All chunk masks of the cache table, `[Cm, T % 16]` per token-slice (1 KB).
pub(crate) type MaskTbl = DmTensor<f32, Chip, CA, SA, m![Cm, T % 16]>;

pub(crate) fn mask_table(device: &mut Device, masks: &HbmTensor<f32, Chip, m![Cm, T]>) -> MaskTbl {
    masks.to_dm(&mut device.tdma)
}

pub(crate) fn mask_vrf(device: &mut Device, tbl: &MaskTbl, c: usize) -> MaskVrf {
    device
        .sub
        .begin(tbl.view().tile::<m![Cm], 1, m![1 # 16, T % 16]>(c))
        .fetch::<m![T / 8 % 2], m![T % 8]>()
        .collect::<m![T / 8 % 2], m![T % 8]>()
        .to_vrf()
}

/// `s[t, g] = (q[g] . k[t] + mask[t]) / sqrt(D)`, 16 tokens per slice.
fn scores<KV: M>(
    device: &mut Device,
    q_trf: &QTrf,
    k: HbmTensorView<'_, bf16, Chip, KV>,
    mask_vrf: &MaskVrf,
) -> DmTensor<f32, Chip, CA, SA, m![T % 16, G]> {
    let k: DmTensor<bf16, Chip, CA, SA, m![T % 16, D]> = k.to_dm(&mut device.tdma);

    device
        .main
        .begin(k.view())
        .fetch::<m![T % 16, D / 16], m![D % 16]>()
        .collect::<m![T % 16, D / 16], m![D % 16]>()
        .contract_outer::<m![T % 16, D / 32], m![D % 32], _, _, _>(q_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![T % 16]>()
        .contract_lane::<m![T % 16], m![G]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![T % 16, G / 4], m![G % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, mask_vrf)
        .vector_fp_div(SQRT_HEAD_DIM)
        .vector_widen_concat::<m![T % 16], m![G]>()
        .vector_final()
        .commit_trim::<m![G]>()
        .commit()
}

/// Reduce `[T % 16, G]` over the slice's tokens, then over the 32 token-slices of each head.
fn row_reduce(
    device: &mut Device,
    s: &DmTensor<f32, Chip, CA, SA, m![T % 16, G]>,
    intra: IntraSliceReduceOpF32,
    inter: InterSliceReduceOpF32,
) -> DmTensor<f32, Chip, CA, SN, m![G]> {
    let local: DmTensor<f32, Chip, CA, SA, m![G]> = device
        .main
        .begin(s.view())
        .fetch::<m![T % 16], m![G]>()
        .collect::<m![T % 16], m![G]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![T % 16, G / 4], m![G % 4]>()
        .vector_intra_slice_reduce::<T, m![G / 4], m![G % 4]>(intra)
        .vector_widen_concat::<m![1], m![G]>()
        .vector_final()
        .commit_trim::<m![G]>()
        .commit();
    device
        .main
        .begin(local.view())
        .fetch::<m![1], m![G]>()
        .collect::<m![1], m![G]>()
        .vector_init()
        .vector_inter_slice_reduce::<SN, m![1]>(inter)
        .vector_final()
        .commit_trim::<m![G]>()
        .commit()
}

/// `exp(s - max)` with `max` broadcast from the head slice to its 32 token-slices.
fn exp_shift(
    device: &mut Device,
    s: &DmTensor<f32, Chip, CA, SA, m![T % 16, G]>,
    max: &DmTensor<f32, Chip, CA, SN, m![G]>,
) -> DmTensor<f32, Chip, CA, SA, m![T % 16, G]> {
    let max: DmTensor<f32, Chip, CA, SA, m![G]> = max.to_dm(&mut device.tdma);
    let max_vrf: VrfTensor<f32, Chip, CA, SA, m![G]> =
        device.sub.begin(max.view()).fetch::<m![1], m![G]>().collect::<m![1], m![G]>().to_vrf();
    device
        .main
        .begin(s.view())
        .fetch::<m![T % 16], m![G]>()
        .collect::<m![T % 16], m![G]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![T % 16, G / 4], m![G % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &max_vrf)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![T % 16], m![G]>()
        .vector_final()
        .commit_trim::<m![G]>()
        .commit()
}

/// `out[g, d] = sum_t p[t, g] v[t, d]` (unnormalized), reduced into the head slice.
fn weighted_sum<KV: M>(
    device: &mut Device,
    e: &DmTensor<f32, Chip, CA, SA, m![T % 16, G]>,
    v: HbmTensorView<'_, bf16, Chip, KV>,
) -> DmTensor<bf16, Chip, CA, SN, m![G, D]> {
    let p: DmTensor<bf16, Chip, CA, SA, m![G, T % 16]> = device
        .main
        .begin(e.view())
        .fetch::<m![T % 16], m![G]>()
        .collect::<m![T % 16], m![G]>()
        .cast::<bf16, m![G # 16]>()
        .transpose::<m![T / 4 % 4, G], m![T % 4 # 16]>()
        .commit_trim::<m![T % 4]>()
        .commit();

    let v: DmTensor<bf16, Chip, CA, SA, m![T % 16, D]> = v.to_dm(&mut device.tdma);
    let v: DmTensor<bf16, Chip, CA, SA, m![D, T % 16]> = device
        .main
        .begin(v.view())
        .fetch::<m![D / 8, T % 16], m![D % 8 # 16]>()
        .collect::<m![D / 8, T % 16], m![D % 8 # 16]>()
        .transpose::<m![D / 8, T / 4 % 4, D % 8], m![T % 4 # 16]>()
        .commit_trim::<m![T % 4]>()
        .commit();
    let v_trf: TrfTensor<bf16, Chip, CA, SA, m![D % 8], m![D / 8, T % 16]> = device
        .sub
        .begin(v.view())
        .fetch::<m![D % 8, D / 8], m![T % 16]>()
        .collect::<m![D % 8, D / 8], m![T % 16]>()
        .to_trf();

    device
        .main
        .begin(p.view())
        .fetch::<m![G], m![T % 16]>()
        .collect::<m![G], m![T % 16]>()
        .contract_outer::<m![G, D / 8], m![T % 16], _, _, _>(&v_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![G, D / 8]>()
        .contract_lane::<m![G, D / 8], m![D % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<SN, m![G, D / 8]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit_trim::<m![D % 8]>()
        .commit()
}

pub(crate) type MaxSum = DmTensor<f32, Chip, CA, SN, m![G]>;
pub(crate) type Out = DmTensor<bf16, Chip, CA, SN, m![G, D]>;

/// First KV chunk: returns (max, sum, unnormalized out).
pub(crate) fn first_dm<KV: M>(
    device: &mut Device,
    q: &QTrf,
    k: HbmTensorView<'_, bf16, Chip, KV>,
    v: HbmTensorView<'_, bf16, Chip, KV>,
    mask: &MaskVrf,
) -> (MaxSum, MaxSum, Out) {
    let s = scores(device, q, k, mask);
    let max = row_reduce(device, &s, IntraSliceReduceOpF32::Max, InterSliceReduceOpF32::Max);
    let e = exp_shift(device, &s, &max);
    let sum = row_reduce(device, &e, IntraSliceReduceOpF32::Add, InterSliceReduceOpF32::Add);
    let out = weighted_sum(device, &e, v);
    (max, sum, out)
}

/// Subsequent KV chunk folded into the running (online-softmax) state.
#[expect(clippy::too_many_arguments)]
pub(crate) fn next_dm<KV: M>(
    device: &mut Device,
    q: &QTrf,
    k: HbmTensorView<'_, bf16, Chip, KV>,
    v: HbmTensorView<'_, bf16, Chip, KV>,
    mask: &MaskVrf,
    old_max: &MaxSum,
    old_sum: &MaxSum,
    old_out: &Out,
) -> (MaxSum, MaxSum, Out) {
    let s = scores(device, q, k, mask);
    let local_max = row_reduce(device, &s, IntraSliceReduceOpF32::Max, InterSliceReduceOpF32::Max);

    let old_max_vrf = g_to_vrf(device, old_max);
    let max: MaxSum = device
        .main
        .begin(local_max.view())
        .fetch::<m![1], m![G]>()
        .collect::<m![1], m![G]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Max, &old_max_vrf)
        .vector_final()
        .commit_trim::<m![G]>()
        .commit();

    // alpha = exp(old_max - max)
    let alpha: MaxSum = device
        .main
        .begin(max.view())
        .fetch::<m![1], m![G]>()
        .collect::<m![1], m![G]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![G / 4], m![G % 4]>()
        .vector_fp_binary_with_mode(FpBinaryOp::SubF, BinaryArgMode::Mode10, &old_max_vrf)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![1], m![G]>()
        .vector_final()
        .commit_trim::<m![G]>()
        .commit();
    let alpha_vrf = g_to_vrf(device, &alpha);

    let e = exp_shift(device, &s, &max);
    let expsum = row_reduce(device, &e, IntraSliceReduceOpF32::Add, InterSliceReduceOpF32::Add);
    let expsum_vrf = g_to_vrf(device, &expsum);

    // sum = old_sum * alpha + expsum
    let sum: MaxSum = device
        .main
        .begin(old_sum.view())
        .fetch::<m![1], m![G]>()
        .collect::<m![1], m![G]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![G / 4], m![G % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &alpha_vrf)
        .vector_widen_concat::<m![1], m![G]>()
        .vector_clip(ClipBinaryOpF32::Add, &expsum_vrf)
        .vector_final()
        .commit_trim::<m![G]>()
        .commit();

    let result = weighted_sum(device, &e, v);
    let result_vrf: VrfTensor<f32, Chip, CA, SN, m![G, D]> = device
        .sub
        .begin(result.view())
        .fetch::<m![G, D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![G, D / 8], m![D % 8]>()
        .to_vrf();

    // out = old_out * alpha + result
    let out: Out = device
        .main
        .begin(old_out.view())
        .fetch::<m![G, D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![G, D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![G, D / 4], m![D % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &alpha_vrf)
        .vector_widen_concat::<m![G, D / 8], m![D % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &result_vrf)
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit_trim::<m![D % 8]>()
        .commit();
    (max, sum, out)
}

pub(crate) fn load_state(
    device: &mut Device,
    max_hbm: &HbmTensor<f32, Chip, m![N, G]>,
    sum_hbm: &HbmTensor<f32, Chip, m![N, G]>,
    out_hbm: &HbmTensor<bf16, Chip, m![N, G, D]>,
) -> (MaxSum, MaxSum, Out) {
    (max_hbm.to_dm(&mut device.tdma), sum_hbm.to_dm(&mut device.tdma), out_hbm.to_dm(&mut device.tdma))
}

/// `out / sum` per head, in the `SN` layout.
pub(crate) fn norm_dm(device: &mut Device, out: &Out, sum: &MaxSum) -> Out {
    let sum_vrf = g_to_vrf(device, sum);
    device
        .main
        .begin(out.view())
        .fetch::<m![G, D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![G, D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![G, D / 4], m![D % 4]>()
        .vector_fp_div(&sum_vrf)
        .vector_widen_concat::<m![G, D / 8], m![D % 8]>()
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit_trim::<m![D % 8]>()
        .commit()
}
