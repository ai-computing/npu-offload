//! RMSNorm pieces: the input norm is split into `u = x * w` (before the projections) and a deferred `1 / rms`
//! factor; the per-head RMSNorm (q_norm / k_norm) is a whole-vector norm over D on one slice.
use furiosa_opt_std::prelude::*;

use super::gemv::{RC, RS};
use super::{C1, Chip, D, H};

pub(crate) const RMS_EPS: f32 = ${EPS};

/// `sqrt(mean(x^2) + eps)` of an `[H]` vector held on one slice.
pub(crate) fn rms_s1<S: M>(
    device: &mut Device,
    input: &DmTensor<bf16, Chip, C1, S, m![H]>,
) -> DmTensor<f32, Chip, C1, S, m![1 # 8]> {
    let ms: DmTensor<f32, Chip, C1, S, m![1 # 8]> = device
        .main
        .begin(input.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(${HF})
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, RMS_EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    sqrt_s1(device, &ms)
}

fn sqrt_s1<S: M>(device: &mut Device, ms: &DmTensor<f32, Chip, C1, S, m![1 # 8]>) -> DmTensor<f32, Chip, C1, S, m![1 # 8]> {
    device
        .main
        .begin(ms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// HF RMSNorm of one `[D]` head per slice: `bf16(bf16(x / rms(x)) * w)`.
pub(crate) fn head_norm<S: M>(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, C1, S, m![D]>,
    w: &HbmTensor<bf16, Chip, m![D]>,
) -> DmTensor<bf16, Chip, C1, S, m![D]> {
    let ms: DmTensor<f32, Chip, C1, S, m![1 # 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![D / 4], m![D % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<D, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(${DF})
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, RMS_EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms = sqrt_s1(device, &ms);
    let rms_vrf: VrfTensor<f32, Chip, C1, S, m![1 # 8]> =
        device.sub.begin(rms.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let y: DmTensor<bf16, Chip, C1, S, m![D]> = device
        .main
        .begin(x.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![D / 4], m![D % 4]>()
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![D / 8], m![D % 8]>()
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit_trim::<m![D % 8]>()
        .commit();
    let wd: DmTensor<bf16, Chip, C1, S, m![D]> = w.to_dm(&mut device.tdma);
    let w_vrf: VrfTensor<f32, Chip, C1, S, m![D]> = device
        .sub
        .begin(wd.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .to_vrf();
    device
        .main
        .begin(y.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![D / 4], m![D % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &w_vrf)
        .vector_widen_concat::<m![D / 8], m![D % 8]>()
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit_trim::<m![D % 8]>()
        .commit()
}

/// `u = x * w` (bf16) computed directly in the GEMV input layout and loaded into TRF.
/// The `1 / rms` factor of the input norm is applied after the projections (v scaling, RoPE / head norm).
pub fn u_trf(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    w: &HbmTensor<bf16, Chip, m![H]>,
) -> TrfTensor<bf16, Chip, RC, RS, m![1], m![H % 32]> {
    let xr: DmTensor<bf16, Chip, RC, RS, m![H % 32]> = x.to_dm(&mut device.tdma);
    let wr: DmTensor<bf16, Chip, RC, RS, m![H % 32]> = w.to_dm(&mut device.tdma);
    let w_vrf: VrfTensor<f32, Chip, RC, RS, m![H % 32]> = device
        .sub
        .begin(wr.view())
        .fetch::<m![H / 16 % 2], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 4], m![H % 8]>()
        .to_vrf();
    let u: DmTensor<bf16, Chip, RC, RS, m![H % 32]> = device
        .main
        .begin(xr.view())
        .fetch::<m![H / 16 % 2], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 4], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 8], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &w_vrf)
        .vector_widen_concat::<m![H / 8 % 4], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    device.sub.begin(u.view()).fetch::<m![1], m![H % 32]>().collect::<m![H / 16 % 2], m![H % 16]>().to_trf()
}
