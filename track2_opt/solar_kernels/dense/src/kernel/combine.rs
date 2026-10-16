//! MoE combine: x_next = y + extra + sum_k gates[k] * expert_out[k]   (f32 math, one bf16 rounding).
//! 2 clusters x 16 slices, 128 hidden elements per slice.
use furiosa_opt_std::prelude::*;

use crate::kernel::axes::{Chip, H, K};

type CC = m![H / 2048];
type CS = m![H / 128 % 16, 1 # 16];

#[device(chip = 1, pe = 8)]
pub fn combine(
    device: &mut Device,
    y: &HbmTensor<bf16, Chip, m![H]>,
    expert_out: &HbmTensor<bf16, Chip, m![K, H]>,
    gates: &HbmTensor<f32, Chip, m![K]>,
    extra: &HbmTensor<bf16, Chip, m![H]>,
    x_next: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let g: DmTensor<f32, Chip, CC, CS, m![K]> = gates.to_dm(&mut device.tdma);
    let g_vrf: VrfTensor<f32, Chip, CC, CS, m![K]> = device
        .sub
        .begin(g.view())
        .fetch::<m![1], m![K]>()
        .collect::<m![1], m![K]>()
        .to_vrf();

    // s = y + extra (f32, sub context, overlaps the expert-row DMA)
    let ed: DmTensor<bf16, Chip, CC, CS, m![H % 128]> = extra.to_dm(&mut device.tdma);
    let ex_vrf: VrfTensor<f32, Chip, CC, CS, m![H % 128]> = device
        .sub
        .begin(ed.view())
        .fetch::<m![H / 16 % 8], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 16], m![H % 8]>()
        .to_vrf();
    let yd: DmTensor<bf16, Chip, CC, CS, m![H % 128]> = y.to_dm(&mut device.tdma);
    let s: DmTensor<f32, Chip, CC, CS, m![H % 128]> = device
        .sub
        .begin(yd.view())
        .fetch::<m![H / 16 % 8], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 16], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 32], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &ex_vrf)
        .vector_widen_concat::<m![H / 8 % 16], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();
    let s_vrf: VrfTensor<f32, Chip, CC, CS, m![H % 128]> = device
        .sub
        .begin(s.view())
        .fetch::<m![H / 8 % 16], m![H % 8]>()
        .collect::<m![H / 8 % 16], m![H % 8]>()
        .to_vrf();

    let rows: DmTensor<bf16, Chip, CC, CS, m![K, H % 128]> = expert_out.to_dm(&mut device.tdma);
    let out: DmTensor<bf16, Chip, CC, CS, m![H % 128]> = device
        .main
        .begin(rows.view())
        .fetch::<m![H / 8 % 16, K], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 16, K], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 8 % 16, K, H / 4 % 2], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &g_vrf)
        .vector_intra_slice_reduce::<K, m![H / 8 % 16, H / 4 % 2], m![H % 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_concat::<m![H / 8 % 16], m![H % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &s_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    out.view().to_hbm_view(&mut device.tdma, x_next.view_mut());
}
