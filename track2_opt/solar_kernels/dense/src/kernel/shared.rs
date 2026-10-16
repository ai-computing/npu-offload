//! Solar shared expert (SwiGLU, F=1280) + residual + router logits, T=1 decode, full chip.
//!   y      = x2 + Wsd @ (silu(Wsg @ h2) * (Wsu @ h2))      (bf16 out, f32 add before rounding)
//!   logits = h2 @ Wr^T                                     (f32 accumulate, f32 out)
use furiosa_opt_std::prelude::*;

use crate::kernel::axes::{Chip, F, H, R};

axes![P = 33];

// gate / up GEMV placement (same as moe_opt ffn1): 2 clusters x (2 x 128) slices, 320 rows x 32 cols each.
type GC = m![F / 640];
type GS = m![F / 320 % 2, H / 32];
type GSO = m![F / 320 % 2, 1 # 128];
type GElem = m![F / 8 % 40, H / 4 % 8, F % 8, H % 4];
/// HBM layout of Wsg / Wsu ([F, H] row-major permuted by host `perm_wg`).
pub type WG = m![F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4];

// down GEMV placement: 2 clusters x (16 x 16) slices, 128 rows x 80 cols each.
type DC = m![H / 2048];
type DS = m![H / 128 % 16, F / 80];
type DSO = m![H / 128 % 16, 1 # 16];
type DElem = m![H / 8 % 16, F / 4 % 20, H % 8, F % 4];
/// HBM layout of Wsd ([H, F] row-major permuted by host `perm_wd`).
pub type WD = m![H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4];

// router GEMV placement: 2 clusters x (2 x 128) slices, 32 rows x 32 cols each.
type RC = m![R / 64];
type RS = m![R / 32 % 2, H / 32];
type RSO = m![R / 32 % 2, 1 # 128];
type RElem = m![R / 8 % 4, H / 4 % 8, R % 8, H % 4];
/// HBM layout of Wr ([R, H] row-major permuted by host `perm_wr`).
pub type WR = m![R / 32, H / 32, R / 8 % 4, H / 4 % 8, R % 8, H % 4];

fn gemv_g(
    device: &mut Device,
    w: &DmTensor<bf16, Chip, GC, GS, GElem>,
    x_trf: &TrfTensor<bf16, Chip, GC, GS, m![1], m![H % 32]>,
) -> DmTensor<bf16, Chip, GC, GSO, m![F % 320]> {
    device
        .main
        .begin(w.view())
        .fetch::<m![F / 8 % 40, H / 4 % 8, F / 4 % 2], m![F % 4, H % 4]>()
        .collect::<m![F / 8 % 40, H / 4 % 8, F / 4 % 2], m![F % 4, H % 4]>()
        .contract_outer::<m![F / 8 % 40, H / 4 % 8], m![F % 8, H % 4], _, _, _>(x_trf)
        .contract_packet::<m![F % 8]>()
        .contract_time::<m![F / 8 % 40]>()
        .contract_lane::<m![F / 8 % 40], m![F % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<GSO, m![F / 8 % 40]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![F % 8 # 16]>()
        .commit_trim::<m![F % 8]>()
        .commit()
}

/// Down GEMV + residual add; `$tail` finishes the vector chain (cast or not) and commits.
macro_rules! down_res {
    ($device:expr, $wd:expr, $h_trf:expr, $x2_vrf:expr) => {
        $device
            .main
            .begin($wd.view())
            .fetch::<m![H / 8 % 16, F / 4 % 20, H / 4 % 2], m![H % 4, F % 4]>()
            .collect::<m![H / 8 % 16, F / 4 % 20, H / 4 % 2], m![H % 4, F % 4]>()
            .contract_outer::<m![H / 8 % 16, F / 4 % 20], m![H % 8, F % 4], _, _, _>($h_trf)
            .contract_packet::<m![H % 8]>()
            .contract_time::<m![H / 8 % 16]>()
            .contract_lane::<m![H / 8 % 16], m![H % 8]>(LaneMode::Sequential)
            .vector_init()
            .vector_inter_slice_reduce::<DSO, m![H / 8 % 16]>(InterSliceReduceOpF32::Add)
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H / 4 % 32], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::AddF, $x2_vrf)
            .vector_widen_concat::<m![H / 8 % 16], m![H % 8]>()
            .vector_final()
    };
}

/// Everything up to the down GEMV: Wd in DM, h in TRF (down placement), x2 in VRF, router logits in DM.
fn shared_router_dm(
    device: &mut Device,
    h2: &HbmTensor<bf16, Chip, m![H]>,
    x2: &HbmTensor<bf16, Chip, m![H]>,
    wg: &HbmTensor<bf16, Chip, WG>,
    wu: &HbmTensor<bf16, Chip, WG>,
    wd: &HbmTensor<bf16, Chip, WD>,
    wr: &HbmTensor<bf16, Chip, WR>,
) -> (
    DmTensor<bf16, Chip, DC, DS, DElem>,
    TrfTensor<bf16, Chip, DC, DS, m![1], m![F % 80]>,
    VrfTensor<f32, Chip, DC, DSO, m![H % 128]>,
    DmTensor<f32, Chip, RC, RSO, m![R % 32]>,
) {
    // activations
    let hr: DmTensor<bf16, Chip, RC, RS, m![H % 32]> = h2.to_dm(&mut device.tdma);
    let xr_trf: TrfTensor<bf16, Chip, RC, RS, m![1], m![H % 32]> = device
        .sub
        .begin(hr.view())
        .fetch::<m![1], m![H % 32]>()
        .collect::<m![H / 16 % 2], m![H % 16]>()
        .to_trf();

    let hx: DmTensor<bf16, Chip, GC, GS, m![H % 32]> = h2.to_dm(&mut device.tdma);
    let x_trf: TrfTensor<bf16, Chip, GC, GS, m![1], m![H % 32]> = device
        .sub
        .begin(hx.view())
        .fetch::<m![1], m![H % 32]>()
        .collect::<m![H / 16 % 2], m![H % 16]>()
        .to_trf();
    // router first: its 1 MB weight lands quickly and the GEMV overlaps the gate/up weight DMA
    let wr: DmTensor<bf16, Chip, RC, RS, RElem> = wr.to_dm(&mut device.tdma);
    let logits: DmTensor<f32, Chip, RC, RSO, m![R % 32]> = device
        .main
        .begin(wr.view())
        .fetch::<m![R / 8 % 4, H / 4 % 8, R / 4 % 2], m![R % 4, H % 4]>()
        .collect::<m![R / 8 % 4, H / 4 % 8, R / 4 % 2], m![R % 4, H % 4]>()
        .contract_outer::<m![R / 8 % 4, H / 4 % 8], m![R % 8, H % 4], _, _, _>(&xr_trf)
        .contract_packet::<m![R % 8]>()
        .contract_time::<m![R / 8 % 4]>()
        .contract_lane::<m![R / 8 % 4], m![R % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<RSO, m![R / 8 % 4]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![R % 8]>()
        .commit();

    // residual operand
    let x2d: DmTensor<bf16, Chip, DC, DSO, m![H % 128]> = x2.to_dm(&mut device.tdma);
    let x2_vrf: VrfTensor<f32, Chip, DC, DSO, m![H % 128]> = device
        .sub
        .begin(x2d.view())
        .fetch::<m![H / 16 % 8], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 16], m![H % 8]>()
        .to_vrf();
    // expert weights (DMA issue order = consumption order)
    let wg: DmTensor<bf16, Chip, GC, GS, GElem> = wg.to_dm(&mut device.tdma);
    let wu: DmTensor<bf16, Chip, GC, GS, GElem> = wu.to_dm(&mut device.tdma);
    let wd: DmTensor<bf16, Chip, DC, DS, DElem> = wd.to_dm(&mut device.tdma);
    let gate = gemv_g(device, &wg, &x_trf);
    let up = gemv_g(device, &wu, &x_trf);

    let silu: DmTensor<f32, Chip, GC, GSO, m![F % 320]> = device
        .sub
        .begin(gate.view())
        .fetch::<m![F / 8 % 40], m![F % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![F / 8 % 40], m![F % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![F / 4 % 80], m![F % 4]>()
        .vector_stash()
        .vector_fp_unary(FpUnaryOp::Sigmoid)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_widen_concat::<m![F / 8 % 40], m![F % 8]>()
        .vector_final()
        .commit_trim::<m![F % 8]>()
        .commit();
    let silu_vrf: VrfTensor<f32, Chip, GC, GSO, m![F % 320]> = device
        .sub
        .begin(silu.view())
        .fetch::<m![F / 8 % 40], m![F % 8]>()
        .collect::<m![F / 8 % 40], m![F % 8]>()
        .to_vrf();
    let h: DmTensor<bf16, Chip, GC, GSO, m![F % 320]> = device
        .main
        .begin(up.view())
        .fetch::<m![F / 8 % 40], m![F % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![F / 8 % 40], m![F % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![F / 4 % 80], m![F % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &silu_vrf)
        .vector_widen_concat::<m![F / 8 % 40], m![F % 8]>()
        .vector_final()
        .cast::<bf16, m![F % 8 # 16]>()
        .commit_trim::<m![F % 8]>()
        .commit();

    // redistribute h[F] to the down-proj placement
    let h_hbm: HbmTensor<bf16, Chip, m![F]> = h.to_hbm(&mut device.tdma);
    let hd: DmTensor<bf16, Chip, DC, DS, m![F % 80]> = h_hbm.to_dm(&mut device.tdma);
    let h_trf: TrfTensor<bf16, Chip, DC, DS, m![1], m![F % 80]> = device
        .sub
        .begin(hd.view())
        .fetch::<m![F / 16 % 5], m![F % 16]>()
        .collect::<m![F / 16 % 5], m![F % 16]>()
        .to_trf();
    (wd, h_trf, x2_vrf, logits)
}

/// y_out = x2 + shared(h2) (bf16), logits_out = h2 @ Wr^T (f32).
#[device(chip = 1, pe = 8)]
pub fn shared_router(
    device: &mut Device,
    h2: &HbmTensor<bf16, Chip, m![H]>,
    x2: &HbmTensor<bf16, Chip, m![H]>,
    wg: &HbmTensor<bf16, Chip, WG>,
    wu: &HbmTensor<bf16, Chip, WG>,
    wd: &HbmTensor<bf16, Chip, WD>,
    wr: &HbmTensor<bf16, Chip, WR>,
    y_out: &mut HbmTensor<bf16, Chip, m![H]>,
    logits_out: &mut HbmTensor<f32, Chip, m![R]>,
) {
    let (wd, h_trf, x2_vrf, logits) = shared_router_dm(device, h2, x2, wg, wu, wd, wr);
    logits.view().to_hbm_view(&mut device.tdma, logits_out.view_mut());
    let y: DmTensor<bf16, Chip, DC, DSO, m![H % 128]> = down_res!(device, wd, &h_trf, &x2_vrf)
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    y.view().to_hbm_view(&mut device.tdma, y_out.view_mut());
}

/// Same as [`shared_router`], plus ONE combined f32 buffer for the host, `cat` = [P = 33, R = 128] f32:
/// rows 0..32 (`cat[0..4096]`) = y as f32 *before* the final bf16 rounding (host `bf16::from_f32` (RNE)
/// reproduces `y_out` bit-exactly), row 32 (`cat[4096..4224]`) = router logits.
/// `y_out` (bf16) is still written for `combine`.
#[device(chip = 1, pe = 8)]
pub fn shared_router_cat(
    device: &mut Device,
    h2: &HbmTensor<bf16, Chip, m![H]>,
    x2: &HbmTensor<bf16, Chip, m![H]>,
    wg: &HbmTensor<bf16, Chip, WG>,
    wu: &HbmTensor<bf16, Chip, WG>,
    wd: &HbmTensor<bf16, Chip, WD>,
    wr: &HbmTensor<bf16, Chip, WR>,
    y_out: &mut HbmTensor<bf16, Chip, m![H]>,
    cat: &mut HbmTensor<f32, Chip, m![P, R]>,
) {
    let (wd, h_trf, x2_vrf, logits) = shared_router_dm(device, h2, x2, wg, wu, wd, wr);
    logits.view().to_hbm_view(&mut device.tdma, cat.view_mut().tile::<m![P], 1, m![1 #{!} 33, R]>(32));
    let y: DmTensor<f32, Chip, DC, DSO, m![H % 128]> = down_res!(device, wd, &h_trf, &x2_vrf)
        .commit_trim::<m![H % 8]>()
        .commit();
    let yb: DmTensor<bf16, Chip, DC, DSO, m![H % 128]> = device
        .main
        .begin(y.view())
        .fetch::<m![H / 8 % 16], m![H % 8]>()
        .collect::<m![H / 8 % 16], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    yb.view().to_hbm_view(&mut device.tdma, y_out.view_mut());
    // SAFETY: relabel H = (H / 128) x (H % 128) as (P = 32) x R; same cells, same order.
    let yr = unsafe { y.reshape::<Chip, m![P = 32 / 16], m![P = 32 % 16, 1 # 16], m![R]>() };
    yr.view().to_hbm_view(&mut device.tdma, cat.view_mut().tile::<m![P], 32, m![P = 32 #{!} 33, R]>(0));
}
