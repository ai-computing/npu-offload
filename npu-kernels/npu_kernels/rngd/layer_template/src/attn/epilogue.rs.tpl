//! o_proj epilogue on both clusters: x2 = o + x, h2 = rmsnorm(x2, w) with H split in halves over the clusters (one
//! slice per cluster). One cross-cluster transfer: the two partial sums of squares.
use furiosa_opt_std::prelude::*;

use super::rmsnorm::RMS_EPS;
use super::{Chip, H, Ip};

/// Cluster = `H / ${HH}`, one slice, element `[H % ${HH}]`.
pub(crate) type EC = m![H / ${HH}];
pub(crate) type ES = m![1 # 256];
pub(crate) type EH = m![H % ${HH}];
/// Both clusters (each holds both partials after the gather).
type CB = m![Ip];

pub(crate) fn residual_rmsnorm_dm(
    device: &mut Device,
    o: &DmTensor<bf16, Chip, EC, ES, EH>,
    x: &HbmTensor<bf16, Chip, m![H]>,
    w: &HbmTensor<bf16, Chip, m![H]>,
) -> (DmTensor<bf16, Chip, EC, ES, EH>, DmTensor<bf16, Chip, EC, ES, EH>) {
    let xd: DmTensor<bf16, Chip, EC, ES, EH> = x.to_dm(&mut device.tdma);
    let wd: DmTensor<bf16, Chip, EC, ES, EH> = w.to_dm(&mut device.tdma);

    // x2 = o + x
    let x2: DmTensor<bf16, Chip, EC, ES, EH> = device
        .main
        .begin_interleaved::<Ip, _, _, _, _, _>(o.view(), xd.view())
        .fetch::<m![H % ${HH} / 8, Ip], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % ${HH} / 8, Ip], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Ip, m![H % ${HH} / 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // per-cluster sum of squares
    let ss: DmTensor<f32, Chip, EC, ES, m![1 # 8]> = device
        .main
        .begin(x2.view())
        .fetch::<m![H % ${HH} / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H % ${HH} / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % ${HH} / 4], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // all-gather the two partials into both clusters
    let ss: DmTensor<f32, Chip, CB, ES, m![H / ${HH}, 1 # 8]> = ss.to_dm(&mut device.tdma);
    let ms: DmTensor<f32, Chip, CB, ES, m![1 # 8]> = device
        .main
        .begin(ss.view())
        .fetch::<m![H / ${HH}], m![1 # 8]>()
        .collect::<m![H / ${HH}], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(${HF})
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, RMS_EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, CB, ES, m![1 # 8]> = device
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
        .commit();
    let rms: DmTensor<f32, Chip, EC, ES, m![1 # 8]> = unsafe { rms.reshape() };
    let rms_vrf: VrfTensor<f32, Chip, EC, ES, m![1 # 8]> =
        device.sub.begin(rms.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();

    // h2 = x2 * w / rms
    let h2: DmTensor<bf16, Chip, EC, ES, EH> = device
        .main
        .begin_interleaved::<Ip, _, _, _, _, _>(x2.view(), wd.view())
        .fetch::<m![H % ${HH} / 8, Ip], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % ${HH} / 8, Ip], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Ip, m![H % ${HH} / 8]>()
        .vector_narrow_split::<m![H % ${HH} / 4], m![H % 4]>()
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![H % ${HH} / 8], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    (x2, h2)
}
