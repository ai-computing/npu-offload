//! Final RMSNorm + LM head GEMV, T=1 decode, full chip.  logits[W] = rmsnorm(x, w_final) @ Wlm^T
//! W = 196608 = 24 chunks x 8192 rows. Per chunk: 2 clusters x (8 row groups x 32 H-blocks) slices,
//! each slice holds 512 rows x 128 cols (128 KB contiguous in HBM); x[H%128] in TRF; partial sums
//! reduced over the 32 H-block slices with the inter-slice reducer.
use furiosa_opt_std::prelude::*;

use crate::axes::{Chip, H, W};

type LC = m![W / 4096 % 2];
type LS = m![W / 512 % 8, H / 128];
type LSO = m![W / 512 % 8, 1 # 32];
type LElem = m![W / 8 % 64, H / 4 % 32, W % 8, H % 4];
/// HBM layout of Wlm ([W, H] row-major permuted by host `perm_wlm`).
pub type WLM = m![W / 8192, W / 4096 % 2, W / 512 % 8, H / 128, W / 8 % 64, H / 4 % 32, W % 8, H % 4];

const EPS: f32 = 1e-5;

/// rms = sqrt(mean(x^2) + eps), computed in a single slice (f32).
fn rms1(device: &mut Device, x: &HbmTensor<bf16, Chip, m![H]>) -> DmTensor<f32, Chip, m![1 # 2], m![1 # 256], m![1 # 8]> {
    let input: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![H]> = x.to_dm(&mut device.tdma);
    let ms: DmTensor<f32, Chip, m![1 # 2], m![1 # 256], m![1 # 8]> = device
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
        .vector_fp_div(4096.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
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

/// One 8192-row chunk: logits in DM (bf16), slices LSO, elements [W % 512].
fn chunk(
    device: &mut Device,
    w: HbmTensorView<'_, bf16, Chip, m![1 # 24, W / 4096 % 2, W / 512 % 8, H / 128, W / 8 % 64, H / 4 % 32, W % 8, H % 4]>,
    x_trf: &TrfTensor<bf16, Chip, LC, LS, m![1], m![H % 128]>,
    out: DmTensorViewMut<'_, bf16, Chip, LC, LSO, m![1 #{!} 24, W % 512]>,
) {
    let w: DmTensor<bf16, Chip, LC, LS, LElem> = w.to_dm(&mut device.tdma);
    device
        .main
        .begin(w.view())
        .fetch::<m![W / 8 % 64, H / 4 % 32, W / 4 % 2], m![W % 4, H % 4]>()
        .collect::<m![W / 8 % 64, H / 4 % 32, W / 4 % 2], m![W % 4, H % 4]>()
        .contract_outer::<m![W / 8 % 64, H / 4 % 32], m![W % 8, H % 4], _, _, _>(x_trf)
        .contract_packet::<m![W % 8]>()
        .contract_time::<m![W / 8 % 64]>()
        .contract_lane::<m![W / 8 % 64], m![W % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<LSO, m![W / 8 % 64]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![W % 8 # 16]>()
        .commit_trim::<m![W % 8]>()
        .commit_view(out);
}

/// rmsnorm(x) * w in the GEMV placement (each slice its H % 128 block), loaded into TRF.
fn prologue(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    w_final: &HbmTensor<bf16, Chip, m![H]>,
) -> TrfTensor<bf16, Chip, LC, LS, m![1], m![H % 128]> {
    let rms = rms1(device, x);
    let rms_hbm: HbmTensor<f32, Chip, m![1 # 8]> = rms.to_hbm(&mut device.tdma);
    let rms_d: DmTensor<f32, Chip, LC, LS, m![1 # 8]> = rms_hbm.to_dm(&mut device.tdma);
    let rms_vrf: VrfTensor<f32, Chip, LC, LS, m![1 # 8]> = device
        .sub
        .begin(rms_d.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let wd: DmTensor<bf16, Chip, LC, LS, m![H % 128]> = w_final.to_dm(&mut device.tdma);
    let w_vrf: VrfTensor<f32, Chip, LC, LS, m![H % 128]> = device
        .sub
        .begin(wd.view())
        .fetch::<m![H / 16 % 8], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 16], m![H % 8]>()
        .to_vrf();
    let xd: DmTensor<bf16, Chip, LC, LS, m![H % 128]> = x.to_dm(&mut device.tdma);
    let hn: DmTensor<bf16, Chip, LC, LS, m![H % 128]> = device
        .main
        .begin(xd.view())
        .fetch::<m![H / 16 % 8], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 16], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 32], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &w_vrf)
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![H / 8 % 16], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    device
        .sub
        .begin(hn.view())
        .fetch::<m![H / 16 % 8], m![H % 16]>()
        .collect::<m![H / 16 % 8], m![H % 16]>()
        .to_trf()
}

#[device(chip = 1, pe = 8)]
pub fn lm_head(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    w_final: &HbmTensor<bf16, Chip, m![H]>,
    wlm: &HbmTensor<bf16, Chip, WLM>,
    logits: &mut HbmTensor<bf16, Chip, m![W]>,
) {
    let x_trf = prologue(device, x, w_final);
    let mut lg: DmTensor<bf16, Chip, LC, LSO, m![W / 8192, W % 512]> = DmTensor::new();
    #[unroll]
    for i in 0..24 {
        chunk(
            device,
            wlm.view().tile::<m![W / 8192], 1, m![1 # 24, W / 4096 % 2, W / 512 % 8, H / 128, W / 8 % 64, H / 4 % 32, W % 8, H % 4]>(i),
            &x_trf,
            lg.view_mut().tile::<m![W / 8192], 1, m![1 #{!} 24, W % 512]>(i),
        );
    }
    lg.view().to_hbm_view(&mut device.tdma, logits.view_mut());
}

/// Per-segment (512 rows) argmax keys of the bf16 logits of chunk `i`:
/// key = f32 bits of the bf16 logit OR'ed with (511 - j) in the (always zero) low bits, then max.
fn chunk_keys(
    device: &mut Device,
    lg: DmTensorView<'_, bf16, Chip, LC, LSO, m![1 # 24, W % 512]>,
    idx_vrf: &VrfTensor<f32, Chip, LC, LSO, m![W % 512]>,
    keys: DmTensorViewMut<'_, f32, Chip, LC, LSO, m![1 #{!} 24, 1 # 8]>,
) {
    device
        .main
        .begin(lg)
        .fetch::<m![W / 16 % 32], m![W % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![W / 8 % 64], m![W % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitOr, idx_vrf)
        .vector_narrow_split::<m![W / 4 % 128], m![W % 4]>()
        .vector_intra_slice_reduce::<W, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(keys);
}

/// lm_head + argmax-friendly keys. `iota[j] = f32::from_bits(511 - j)` (constant, upload once).
/// `keys[s]` (s = 0..384, rows s*512..s*512+512): bf16 logit max = f32::from_bits(bits & 0xFFFF_0000),
/// row = s*512 + 511 - (bits & 0x1FF) (ties -> lowest row). Host: argmax over 384 keys (1.5 KB download).
#[device(chip = 1, pe = 8)]
pub fn lm_head_am(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    w_final: &HbmTensor<bf16, Chip, m![H]>,
    wlm: &HbmTensor<bf16, Chip, WLM>,
    iota: &HbmTensor<f32, Chip, m![W % 512]>,
    logits: &mut HbmTensor<bf16, Chip, m![W]>,
    keys: &mut HbmTensor<f32, Chip, m![W / 512]>,
) {
    let x_trf = prologue(device, x, w_final);
    let idx: DmTensor<f32, Chip, LC, LSO, m![W % 512]> = iota.to_dm(&mut device.tdma);
    let idx_vrf: VrfTensor<f32, Chip, LC, LSO, m![W % 512]> = device
        .sub
        .begin(idx.view())
        .fetch::<m![W / 8 % 64], m![W % 8]>()
        .collect::<m![W / 8 % 64], m![W % 8]>()
        .to_vrf();
    let mut lg: DmTensor<bf16, Chip, LC, LSO, m![W / 8192, W % 512]> = DmTensor::new();
    let mut kd: DmTensor<f32, Chip, LC, LSO, m![W / 8192, 1 # 8]> = DmTensor::new();
    #[unroll]
    for i in 0..24 {
        chunk(
            device,
            wlm.view().tile::<m![W / 8192], 1, m![1 # 24, W / 4096 % 2, W / 512 % 8, H / 128, W / 8 % 64, H / 4 % 32, W % 8, H % 4]>(i),
            &x_trf,
            lg.view_mut().tile::<m![W / 8192], 1, m![1 #{!} 24, W % 512]>(i),
        );
        chunk_keys(
            device,
            lg.view().tile::<m![W / 8192], 1, m![1 # 24, W % 512]>(i),
            &idx_vrf,
            kd.view_mut().tile::<m![W / 8192], 1, m![1 #{!} 24, 1 # 8]>(i),
        );
    }
    lg.view().to_hbm_view(&mut device.tdma, logits.view_mut());
    kd.view().to_hbm_view(&mut device.tdma, keys.view_mut());
}
