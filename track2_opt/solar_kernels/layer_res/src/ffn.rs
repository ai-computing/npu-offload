//! Solar-Open-style expert FFN (SwiGLU) on the full chip, bf16, T=1.
//! out = Wd @ (silu(Wg @ x) * (Wu @ x)), Wg/Wu [F, H], Wd [H, F].
use furiosa_opt_std::prelude::*;

pub use crate::axes::{Chip, E, F, H, K};


// gate / up GEMV placement: 2 clusters x (2 x 128) slices, each slice 320 rows x 32 cols.
type GC = m![F / 640];
type GS = m![F / 320 % 2, H / 32];
type GSO = m![F / 320 % 2, 1 # 128];
type GElem = m![F / 8 % 40, H / 4 % 8, F % 8, H % 4];
/// HBM layout of one Wg (or Wu): contiguous 20 KB chunk per (cluster, slice).
pub type WG = m![F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4];

// down GEMV placement: 2 clusters x (16 x 16) slices, each slice 128 rows x 80 cols.
type DC = m![H / 2048];
type DS = m![H / 128 % 16, F / 80];
type DSO = m![H / 128 % 16, 1 # 16];
type DElem = m![H / 8 % 16, F / 4 % 20, H % 8, F % 4];
/// HBM layout of one Wd.
pub type WD = m![H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4];

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

/// One expert: weights are views selected from resident tables.
fn expert<WGv: M, WDv: M, Ov: M>(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, GC, GS, m![1], m![H % 32]>,
    wg: HbmTensorView<'_, bf16, Chip, WGv>,
    wu: HbmTensorView<'_, bf16, Chip, WGv>,
    wd: HbmTensorView<'_, bf16, Chip, WDv>,
    out: HbmTensorViewMut<'_, bf16, Chip, Ov>,
) {
    let wg: DmTensor<bf16, Chip, GC, GS, GElem> = wg.to_dm(&mut device.tdma);
    let wu: DmTensor<bf16, Chip, GC, GS, GElem> = wu.to_dm(&mut device.tdma);
    let wd: DmTensor<bf16, Chip, DC, DS, DElem> = wd.to_dm(&mut device.tdma);
    let gate = gemv_g(device, &wg, x_trf);
    let up = gemv_g(device, &wu, x_trf);

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

    // redistribute h[F] to the down-proj placement (each slice needs its 80-wide chunk)
    let h_hbm: HbmTensor<bf16, Chip, m![F]> = h.to_hbm(&mut device.tdma);
    let hx: DmTensor<bf16, Chip, DC, DS, m![F % 80]> = h_hbm.to_dm(&mut device.tdma);
    let h_trf: TrfTensor<bf16, Chip, DC, DS, m![1], m![F % 80]> = device
        .sub
        .begin(hx.view())
        .fetch::<m![F / 16 % 5], m![F % 16]>()
        .collect::<m![F / 16 % 5], m![F % 16]>()
        .to_trf();
    let y: DmTensor<bf16, Chip, DC, DSO, m![H % 128]> = device
        .main
        .begin(wd.view())
        .fetch::<m![H / 8 % 16, F / 4 % 20, H / 4 % 2], m![H % 4, F % 4]>()
        .collect::<m![H / 8 % 16, F / 4 % 20, H / 4 % 2], m![H % 4, F % 4]>()
        .contract_outer::<m![H / 8 % 16, F / 4 % 20], m![H % 8, F % 4], _, _, _>(&h_trf)
        .contract_packet::<m![H % 8]>()
        .contract_time::<m![H / 8 % 16]>()
        .contract_lane::<m![H / 8 % 16], m![H % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<DSO, m![H / 8 % 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    y.view().to_hbm_view(&mut device.tdma, out);
}

fn load_x(device: &mut Device, x: &HbmTensor<bf16, Chip, m![H]>) -> TrfTensor<bf16, Chip, GC, GS, m![1], m![H % 32]> {
    let x: DmTensor<bf16, Chip, GC, GS, m![H % 32]> = x.to_dm(&mut device.tdma);
    device
        .sub
        .begin(x.view())
        .fetch::<m![1], m![H % 32]>()
        .collect::<m![H / 16 % 2], m![H % 16]>()
        .to_trf()
}

/// Single expert FFN with dedicated weight tensors (no table).
#[device(chip = 1, pe = 8)]
pub fn ffn1(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    wg: &HbmTensor<bf16, Chip, WG>,
    wu: &HbmTensor<bf16, Chip, WG>,
    wd: &HbmTensor<bf16, Chip, WD>,
    out: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let x_trf = load_x(device, x);
    expert(device, &x_trf, wg.view(), wu.view(), wd.view(), out.view_mut());
}

/// Expert tables resident in HBM: E experts in device layout.
pub type WGT = m![E, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4];
pub type WDT = m![E, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4];

/// One expert chosen at runtime by an HBM scalar index (runtime HBM tile).
#[device(chip = 1, pe = 8)]
pub fn ffn_sel1(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    wg_t: &HbmTensor<bf16, Chip, WGT>,
    wu_t: &HbmTensor<bf16, Chip, WGT>,
    wd_t: &HbmTensor<bf16, Chip, WDT>,
    id: &HbmScalar<usize>,
    out: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let x_trf = load_x(device, x);
    let e = id.to_spm(&mut device.tdma);
    let wg = wg_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
    let wu = wu_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
    let wd = wd_t.view().tile::<m![E], 1, m![1 # 24, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e);
    expert(device, &x_trf, wg, wu, wd, out.view_mut());
}

/// K=8 experts selected at runtime. `ids` packs 8 expert ids base-64 (e_k * 64^k) and the
/// active count in bits 56..63. Expert k's output goes to row k of `out` (combine is done by caller).
#[device(chip = 1, pe = 8)]
pub fn ffn_k8(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    wg_t: &HbmTensor<bf16, Chip, WGT>,
    wu_t: &HbmTensor<bf16, Chip, WGT>,
    wd_t: &HbmTensor<bf16, Chip, WDT>,
    ids: &HbmScalar<u64>,
    out: &mut HbmTensor<bf16, Chip, m![K, H]>,
) {
    let x_trf = load_x(device, x);
    let packed = ids.to_spm(&mut device.tdma);
    let active = (packed / 72057594037927936u64) as usize; // 2^56
    if 0usize < active {
        let e = ((packed / 1u64) % 64) as usize;
        let wg = wg_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
        let wu = wu_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
        let wd = wd_t.view().tile::<m![E], 1, m![1 # 24, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e);
        let o = out.view_mut().tile::<m![K], 1, m![1 #{!} 8, H]>(0);
        expert(device, &x_trf, wg, wu, wd, o);
    }
    if 1usize < active {
        let e = ((packed / 64u64) % 64) as usize;
        let wg = wg_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
        let wu = wu_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
        let wd = wd_t.view().tile::<m![E], 1, m![1 # 24, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e);
        let o = out.view_mut().tile::<m![K], 1, m![1 #{!} 8, H]>(1);
        expert(device, &x_trf, wg, wu, wd, o);
    }
    if 2usize < active {
        let e = ((packed / 4096u64) % 64) as usize;
        let wg = wg_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
        let wu = wu_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
        let wd = wd_t.view().tile::<m![E], 1, m![1 # 24, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e);
        let o = out.view_mut().tile::<m![K], 1, m![1 #{!} 8, H]>(2);
        expert(device, &x_trf, wg, wu, wd, o);
    }
    if 3usize < active {
        let e = ((packed / 262144u64) % 64) as usize;
        let wg = wg_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
        let wu = wu_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
        let wd = wd_t.view().tile::<m![E], 1, m![1 # 24, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e);
        let o = out.view_mut().tile::<m![K], 1, m![1 #{!} 8, H]>(3);
        expert(device, &x_trf, wg, wu, wd, o);
    }
    if 4usize < active {
        let e = ((packed / 16777216u64) % 64) as usize;
        let wg = wg_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
        let wu = wu_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
        let wd = wd_t.view().tile::<m![E], 1, m![1 # 24, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e);
        let o = out.view_mut().tile::<m![K], 1, m![1 #{!} 8, H]>(4);
        expert(device, &x_trf, wg, wu, wd, o);
    }
    if 5usize < active {
        let e = ((packed / 1073741824u64) % 64) as usize;
        let wg = wg_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
        let wu = wu_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
        let wd = wd_t.view().tile::<m![E], 1, m![1 # 24, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e);
        let o = out.view_mut().tile::<m![K], 1, m![1 #{!} 8, H]>(5);
        expert(device, &x_trf, wg, wu, wd, o);
    }
    if 6usize < active {
        let e = ((packed / 68719476736u64) % 64) as usize;
        let wg = wg_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
        let wu = wu_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
        let wd = wd_t.view().tile::<m![E], 1, m![1 # 24, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e);
        let o = out.view_mut().tile::<m![K], 1, m![1 #{!} 8, H]>(6);
        expert(device, &x_trf, wg, wu, wd, o);
    }
    if 7usize < active {
        let e = ((packed / 4398046511104u64) % 64) as usize;
        let wg = wg_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
        let wu = wu_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
        let wd = wd_t.view().tile::<m![E], 1, m![1 # 24, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e);
        let o = out.view_mut().tile::<m![K], 1, m![1 #{!} 8, H]>(7);
        expert(device, &x_trf, wg, wu, wd, o);
    }
}

/// Gate/up GEMVs + SiLU*up for one expert; h is committed into row k of a shared DM buffer.
fn gate_up_into<WGv: M>(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, GC, GS, m![1], m![H % 32]>,
    wg: HbmTensorView<'_, bf16, Chip, WGv>,
    wu: HbmTensorView<'_, bf16, Chip, WGv>,
    hs: DmTensorViewMut<'_, bf16, Chip, GC, GSO, m![1 #{!} 8, F % 320]>,
) {
    let wg: DmTensor<bf16, Chip, GC, GS, GElem> = wg.to_dm(&mut device.tdma);
    let wu: DmTensor<bf16, Chip, GC, GS, GElem> = wu.to_dm(&mut device.tdma);
    let gate = gemv_g(device, &wg, x_trf);
    let up = gemv_g(device, &wu, x_trf);

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
    device
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
        .commit_view(hs);
}

/// Down GEMV for one expert from its h row (already in down placement) into row k of ys.
fn down_into(
    device: &mut Device,
    wd: &DmTensor<bf16, Chip, DC, DS, DElem>,
    hx: DmTensorView<'_, bf16, Chip, DC, DS, m![1 # 8, F % 80]>,
    ys: DmTensorViewMut<'_, bf16, Chip, DC, DSO, m![1 #{!} 8, H % 128]>,
) {
    let h_trf: TrfTensor<bf16, Chip, DC, DS, m![1], m![F % 80]> = device
        .sub
        .begin(hx)
        .fetch::<m![F / 16 % 5], m![F % 16]>()
        .collect::<m![F / 16 % 5], m![F % 16]>()
        .to_trf();
    device
        .main
        .begin(wd.view())
        .fetch::<m![H / 8 % 16, F / 4 % 20, H / 4 % 2], m![H % 4, F % 4]>()
        .collect::<m![H / 8 % 16, F / 4 % 20, H / 4 % 2], m![H % 4, F % 4]>()
        .contract_outer::<m![H / 8 % 16, F / 4 % 20], m![H % 8, F % 4], _, _, _>(&h_trf)
        .contract_packet::<m![H % 8]>()
        .contract_time::<m![H / 8 % 16]>()
        .contract_lane::<m![H / 8 % 16], m![H % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<DSO, m![H / 8 % 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(ys);
}

/// Unguarded variant (always 8 experts).
/// K=8 experts selected at runtime. `ids` packs 8 expert ids base-64 (e_k * 64^k) and the
/// active count in bits 56..63. Expert k's output goes to row k of `out` (combine is done by caller).
/// Phase A: gate/up for all experts (h rows batched in DM); one h redistribution for all 8;
/// phase B: down GEMVs as each Wd arrives. Outputs gathered in DM, stored once.
#[device(chip = 1, pe = 8)]
pub fn ffn_k8u(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    wg_t: &HbmTensor<bf16, Chip, WGT>,
    wu_t: &HbmTensor<bf16, Chip, WGT>,
    wd_t: &HbmTensor<bf16, Chip, WDT>,
    ids: &HbmScalar<u64>,
    out: &mut HbmTensor<bf16, Chip, m![K, H]>,
) {
    let x_trf = load_x(device, x);
    let packed = ids.to_spm(&mut device.tdma);
    let mut hs: DmTensor<bf16, Chip, GC, GSO, m![K, F % 320]> = DmTensor::new();
    let mut ys: DmTensor<bf16, Chip, DC, DSO, m![K, H % 128]> = DmTensor::new();
    let e0 = ((packed / 1u64) % 64) as usize;
    gate_up_into(
        device,
        &x_trf,
        wg_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e0),
        wu_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e0),
        hs.view_mut().tile::<m![K], 1, m![1 #{!} 8, F % 320]>(0),
    );
    let e1 = ((packed / 64u64) % 64) as usize;
    gate_up_into(
        device,
        &x_trf,
        wg_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e1),
        wu_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e1),
        hs.view_mut().tile::<m![K], 1, m![1 #{!} 8, F % 320]>(1),
    );
    let e2 = ((packed / 4096u64) % 64) as usize;
    gate_up_into(
        device,
        &x_trf,
        wg_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e2),
        wu_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e2),
        hs.view_mut().tile::<m![K], 1, m![1 #{!} 8, F % 320]>(2),
    );
    let e3 = ((packed / 262144u64) % 64) as usize;
    gate_up_into(
        device,
        &x_trf,
        wg_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e3),
        wu_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e3),
        hs.view_mut().tile::<m![K], 1, m![1 #{!} 8, F % 320]>(3),
    );
    let e4 = ((packed / 16777216u64) % 64) as usize;
    gate_up_into(
        device,
        &x_trf,
        wg_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e4),
        wu_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e4),
        hs.view_mut().tile::<m![K], 1, m![1 #{!} 8, F % 320]>(4),
    );
    let e5 = ((packed / 1073741824u64) % 64) as usize;
    gate_up_into(
        device,
        &x_trf,
        wg_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e5),
        wu_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e5),
        hs.view_mut().tile::<m![K], 1, m![1 #{!} 8, F % 320]>(5),
    );
    let e6 = ((packed / 68719476736u64) % 64) as usize;
    gate_up_into(
        device,
        &x_trf,
        wg_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e6),
        wu_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e6),
        hs.view_mut().tile::<m![K], 1, m![1 #{!} 8, F % 320]>(6),
    );
    let e7 = ((packed / 4398046511104u64) % 64) as usize;
    gate_up_into(
        device,
        &x_trf,
        wg_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e7),
        wu_t.view().tile::<m![E], 1, m![1 # 24, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e7),
        hs.view_mut().tile::<m![K], 1, m![1 #{!} 8, F % 320]>(7),
    );
    let wd0: DmTensor<bf16, Chip, DC, DS, DElem> = wd_t.view().tile::<m![E], 1, m![1 # 24, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e0).to_dm(&mut device.tdma);
    let h_hbm: HbmTensor<bf16, Chip, m![K, F]> = hs.to_hbm(&mut device.tdma);
    let hx: DmTensor<bf16, Chip, DC, DS, m![K, F % 80]> = h_hbm.to_dm(&mut device.tdma);
    let wd1: DmTensor<bf16, Chip, DC, DS, DElem> = wd_t.view().tile::<m![E], 1, m![1 # 24, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e1).to_dm(&mut device.tdma);
    let wd2: DmTensor<bf16, Chip, DC, DS, DElem> = wd_t.view().tile::<m![E], 1, m![1 # 24, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e2).to_dm(&mut device.tdma);
    let wd3: DmTensor<bf16, Chip, DC, DS, DElem> = wd_t.view().tile::<m![E], 1, m![1 # 24, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e3).to_dm(&mut device.tdma);
    let wd4: DmTensor<bf16, Chip, DC, DS, DElem> = wd_t.view().tile::<m![E], 1, m![1 # 24, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e4).to_dm(&mut device.tdma);
    let wd5: DmTensor<bf16, Chip, DC, DS, DElem> = wd_t.view().tile::<m![E], 1, m![1 # 24, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e5).to_dm(&mut device.tdma);
    let wd6: DmTensor<bf16, Chip, DC, DS, DElem> = wd_t.view().tile::<m![E], 1, m![1 # 24, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e6).to_dm(&mut device.tdma);
    let wd7: DmTensor<bf16, Chip, DC, DS, DElem> = wd_t.view().tile::<m![E], 1, m![1 # 24, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e7).to_dm(&mut device.tdma);
    down_into(
        device,
        &wd0,
        hx.view().tile::<m![K], 1, m![1 # 8, F % 80]>(0),
        ys.view_mut().tile::<m![K], 1, m![1 #{!} 8, H % 128]>(0),
    );
    down_into(
        device,
        &wd1,
        hx.view().tile::<m![K], 1, m![1 # 8, F % 80]>(1),
        ys.view_mut().tile::<m![K], 1, m![1 #{!} 8, H % 128]>(1),
    );
    down_into(
        device,
        &wd2,
        hx.view().tile::<m![K], 1, m![1 # 8, F % 80]>(2),
        ys.view_mut().tile::<m![K], 1, m![1 #{!} 8, H % 128]>(2),
    );
    down_into(
        device,
        &wd3,
        hx.view().tile::<m![K], 1, m![1 # 8, F % 80]>(3),
        ys.view_mut().tile::<m![K], 1, m![1 #{!} 8, H % 128]>(3),
    );
    down_into(
        device,
        &wd4,
        hx.view().tile::<m![K], 1, m![1 # 8, F % 80]>(4),
        ys.view_mut().tile::<m![K], 1, m![1 #{!} 8, H % 128]>(4),
    );
    down_into(
        device,
        &wd5,
        hx.view().tile::<m![K], 1, m![1 # 8, F % 80]>(5),
        ys.view_mut().tile::<m![K], 1, m![1 #{!} 8, H % 128]>(5),
    );
    down_into(
        device,
        &wd6,
        hx.view().tile::<m![K], 1, m![1 # 8, F % 80]>(6),
        ys.view_mut().tile::<m![K], 1, m![1 #{!} 8, H % 128]>(6),
    );
    down_into(
        device,
        &wd7,
        hx.view().tile::<m![K], 1, m![1 # 8, F % 80]>(7),
        ys.view_mut().tile::<m![K], 1, m![1 #{!} 8, H % 128]>(7),
    );
    ys.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
