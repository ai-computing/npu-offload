//! o_proj epilogue on both clusters: x2 = o + x, h2 = rmsnorm(x2, w) with H split 2048 per cluster
//! (slice 0 of each cluster, the o_proj output placement). Only one cross-cluster transfer
//! (the two partial sums of squares).
use furiosa_opt_std::prelude::*;

use super::gemv::OSO;
use super::{Chip, H, Ip, N};

/// Cluster = `H / 512 % 2` (the o_proj row split), element `[H / 1024, H % 512]`.
pub(crate) type EC = m![H / 512 % 2];
pub(crate) type EH = m![H / 1024, H % 512];
/// Both clusters, gathered per-cluster partials.
type CB = m![N / 4];

const RMS_EPS: f32 = 1e-5;

pub(crate) fn residual_rmsnorm(
    device: &mut Device,
    o: &DmTensor<bf16, Chip, EC, OSO, EH>,
    x: &HbmTensor<bf16, Chip, m![H]>,
    w: &HbmTensor<bf16, Chip, m![H]>,
    x2_out: &mut HbmTensor<bf16, Chip, m![H]>,
    h2_out: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let (x2, h2) = residual_rmsnorm_dm(device, o, x, w);
    x2.view().to_hbm_view(&mut device.tdma, x2_out.view_mut());
    h2.view().to_hbm_view(&mut device.tdma, h2_out.view_mut());
}

/// Same math as [`residual_rmsnorm`]; returns (x2, h2) in DM (bf16, `EC` x `OSO` x `EH`).
pub(crate) fn residual_rmsnorm_dm(
    device: &mut Device,
    o: &DmTensor<bf16, Chip, EC, OSO, EH>,
    x: &HbmTensor<bf16, Chip, m![H]>,
    w: &HbmTensor<bf16, Chip, m![H]>,
) -> (DmTensor<bf16, Chip, EC, OSO, EH>, DmTensor<bf16, Chip, EC, OSO, EH>) {
    let xd: DmTensor<bf16, Chip, EC, OSO, EH> = x.to_dm(&mut device.tdma);
    let wd: DmTensor<bf16, Chip, EC, OSO, EH> = w.to_dm(&mut device.tdma);

    // x2 = o + x
    let x2: DmTensor<bf16, Chip, EC, OSO, EH> = device
        .main
        .begin_interleaved::<Ip, _, _, _, _, _>(o.view(), xd.view())
        .fetch::<m![H / 1024, H % 512 / 8, Ip], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 1024, H % 512 / 8, Ip], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Ip, m![H / 1024, H % 512 / 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // per-cluster sum of squares
    let ss: DmTensor<f32, Chip, EC, OSO, m![1 # 8]> = device
        .main
        .begin(x2.view())
        .fetch::<m![H / 1024, H % 512 / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 1024, H % 512 / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 1024, H % 512 / 4], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // all-gather the two partials into both clusters
    let ss: DmTensor<f32, Chip, CB, OSO, m![H / 512 % 2, 1 # 8]> = ss.to_dm(&mut device.tdma);
    let ms: DmTensor<f32, Chip, CB, OSO, m![1 # 8]> = device
        .main
        .begin(ss.view())
        .fetch::<m![H / 512 % 2], m![1 # 8]>()
        .collect::<m![H / 512 % 2], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(4096.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, RMS_EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, CB, OSO, m![1 # 8]> = device
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
    let rms: DmTensor<f32, Chip, EC, OSO, m![1 # 8]> = unsafe { rms.reshape() };
    let rms_vrf: VrfTensor<f32, Chip, EC, OSO, m![1 # 8]> =
        device.sub.begin(rms.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();

    // h2 = x2 * w / rms
    let h2: DmTensor<bf16, Chip, EC, OSO, EH> = device
        .main
        .begin_interleaved::<Ip, _, _, _, _, _>(x2.view(), wd.view())
        .fetch::<m![H / 1024, H % 512 / 8, Ip], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 1024, H % 512 / 8, Ip], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Ip, m![H / 1024, H % 512 / 8]>()
        .vector_narrow_split::<m![H / 1024, H % 512 / 4], m![H % 4]>()
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![H / 1024, H % 512 / 8], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    (x2, h2)
}
