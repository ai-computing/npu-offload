//! rotate_half RoPE of one `[D]` head per slice. `rotate_half` is a plain half swap on the device; its sign lives in
//! `sin` (host: `host::rope_sin_signed`).
use furiosa_opt_std::prelude::*;

use super::{C1, Chip, D};

fn tables<S: M>(
    device: &mut Device,
    cos: &HbmTensor<bf16, Chip, m![D]>,
    sin: &HbmTensor<bf16, Chip, m![D]>,
) -> (VrfTensor<f32, Chip, C1, S, m![D]>, VrfTensor<f32, Chip, C1, S, m![D]>) {
    let cos: DmTensor<bf16, Chip, C1, S, m![D]> = cos.to_dm(&mut device.tdma);
    let sin: DmTensor<bf16, Chip, C1, S, m![D]> = sin.to_dm(&mut device.tdma);
    let cos_vrf: VrfTensor<f32, Chip, C1, S, m![D]> = device
        .sub
        .begin(cos.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .to_vrf();
    let sin_vrf: VrfTensor<f32, Chip, C1, S, m![D]> = device
        .sub
        .begin(sin.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .to_vrf();
    (cos_vrf, sin_vrf)
}

fn rotate_half<S: M>(device: &mut Device, k: &DmTensor<bf16, Chip, C1, S, m![D]>) -> DmTensor<bf16, Chip, C1, S, m![D]> {
    let fst_half_k = k.view().tile::<m![D], ${DH}, m![D = ${DH} # ${D}]>(0);
    let snd_half_k = k.view().tile::<m![D], ${DH}, m![D = ${DH} # ${D}]>(${DH});
    let mut rot: DmTensor<bf16, Chip, C1, S, m![D]> = DmTensor::new();
    device
        .main
        .begin(fst_half_k)
        .fetch::<m![1], m![D = ${DH}]>()
        .collect::<m![D = ${DH} / 16], m![D = ${DH} % 16]>()
        .commit_trim::<m![D = ${DH} % 16]>()
        .commit_view(rot.view_mut().tile::<m![D], ${DH}, m![D = ${DH} #{!} ${D}]>(${DH}));
    device
        .main
        .begin(snd_half_k)
        .fetch::<m![1], m![D = ${DH}]>()
        .collect::<m![D = ${DH} / 16], m![D = ${DH} % 16]>()
        .commit_trim::<m![D = ${DH} % 16]>()
        .commit_view(rot.view_mut().tile::<m![D], ${DH}, m![D = ${DH} #{!} ${D}]>(0));
    rot
}

fn combine<S: M>(
    device: &mut Device,
    tc: &DmTensor<f32, Chip, C1, S, m![D]>,
    ts: &DmTensor<f32, Chip, C1, S, m![D]>,
) -> DmTensor<bf16, Chip, C1, S, m![D]> {
    let ts_vrf: VrfTensor<f32, Chip, C1, S, m![D]> =
        device.sub.begin(ts.view()).fetch::<m![D / 8], m![D % 8]>().collect::<m![D / 8], m![D % 8]>().to_vrf();
    device
        .main
        .begin(tc.view())
        .fetch::<m![D / 8], m![D % 8]>()
        .collect::<m![D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &ts_vrf)
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit_trim::<m![D % 8]>()
        .commit()
}

/// `x * cos + rotate_half(x) * sin` for one head per slice.
pub(crate) fn rope_vec<S: M>(
    device: &mut Device,
    k: &DmTensor<bf16, Chip, C1, S, m![D]>,
    cos: &HbmTensor<bf16, Chip, m![D]>,
    sin: &HbmTensor<bf16, Chip, m![D]>,
) -> DmTensor<bf16, Chip, C1, S, m![D]> {
    let (cos_vrf, sin_vrf) = tables::<S>(device, cos, sin);
    let rot = rotate_half(device, k);
    let tc: DmTensor<f32, Chip, C1, S, m![D]> = device
        .main
        .begin(k.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![D / 4], m![D % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![D / 8], m![D % 8]>()
        .vector_final()
        .commit_trim::<m![D % 8]>()
        .commit();
    let ts: DmTensor<f32, Chip, C1, S, m![D]> = device
        .main
        .begin(rot.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![D / 4], m![D % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![D / 8], m![D % 8]>()
        .vector_final()
        .commit_trim::<m![D % 8]>()
        .commit();
    combine(device, &tc, &ts)
}

/// [`rope_vec`] of `x / rms` (the deferred input-norm factor, one value per slice).
pub(crate) fn rope_vec_div<S: M>(
    device: &mut Device,
    k: &DmTensor<bf16, Chip, C1, S, m![D]>,
    cos: &HbmTensor<bf16, Chip, m![D]>,
    sin: &HbmTensor<bf16, Chip, m![D]>,
    rms_vrf: &VrfTensor<f32, Chip, C1, S, m![1 # 8]>,
) -> DmTensor<bf16, Chip, C1, S, m![D]> {
    let (cos_vrf, sin_vrf) = tables::<S>(device, cos, sin);
    let rot = rotate_half(device, k);
    let tc: DmTensor<f32, Chip, C1, S, m![D]> = device
        .main
        .begin(k.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![D / 4], m![D % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_fp_div(rms_vrf)
        .vector_widen_concat::<m![D / 8], m![D % 8]>()
        .vector_final()
        .commit_trim::<m![D % 8]>()
        .commit();
    let ts: DmTensor<f32, Chip, C1, S, m![D]> = device
        .main
        .begin(rot.view())
        .fetch::<m![D / 16], m![D % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![D / 8], m![D % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![D / 4], m![D % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_fp_div(rms_vrf)
        .vector_widen_concat::<m![D / 8], m![D % 8]>()
        .vector_final()
        .commit_trim::<m![D % 8]>()
        .commit();
    combine(device, &tc, &ts)
}
