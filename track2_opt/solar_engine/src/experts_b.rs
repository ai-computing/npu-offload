//! Chip-B fused routed-experts kernel and expert-slot swap.
//!
//! `experts_b`: x_B = sum_{k < active} gates[k] * expert_{e_k}(x)   (bf16 out)
//! Same per-expert chain (and bf16 roundings) as `ffn32::ffn_k8`, and the same f32 gated reduction
//! as `combine::combine` with y = extra = 0, but the K expert rows never leave DM and there is a
//! single launch. `ids` packs the active count in bits 56..63 and expert k's table slot in bits 6k..6k+5.
use furiosa_opt_std::prelude::*;

pub use crate::axes::{Chip, E2 as E, F, H, K};
use crate::ffn32::{DC, DElem, DS, DSO, GC, GElem, GS, GSO, WD, WDT, WG, WGT, gemv_g, load_x};

/// One expert, output row committed into `ys` (DM, combine placement).
fn expert_into<WGv: M, WDv: M>(
    device: &mut Device,
    x_trf: &TrfTensor<bf16, Chip, GC, GS, m![1], m![H % 32]>,
    wg: HbmTensorView<'_, bf16, Chip, WGv>,
    wu: HbmTensorView<'_, bf16, Chip, WGv>,
    wd: HbmTensorView<'_, bf16, Chip, WDv>,
    ys: DmTensorViewMut<'_, bf16, Chip, DC, DSO, m![1 #{!} 8, H % 128]>,
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

    let h_hbm: HbmTensor<bf16, Chip, m![F]> = h.to_hbm(&mut device.tdma);
    let hx: DmTensor<bf16, Chip, DC, DS, m![F % 80]> = h_hbm.to_dm(&mut device.tdma);
    let h_trf: TrfTensor<bf16, Chip, DC, DS, m![1], m![F % 80]> = device
        .sub
        .begin(hx.view())
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

macro_rules! slot {
    ($device:ident, $x_trf:ident, $wg_t:ident, $wu_t:ident, $wd_t:ident, $ys:ident, $packed:ident, $active:ident, $k:literal, $div:literal) => {
        if $k < $active {
            let e = (($packed / $div) % 64) as usize;
            let wg = $wg_t.view().tile::<m![E], 1, m![1 # 32, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
            let wu = $wu_t.view().tile::<m![E], 1, m![1 # 32, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e);
            let wd = $wd_t.view().tile::<m![E], 1, m![1 # 32, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e);
            expert_into($device, &$x_trf, wg, wu, wd, $ys.view_mut().tile::<m![K], 1, m![1 #{!} 8, H % 128]>($k));
        }
    };
}

/// Fused chip-B experts: guarded expert FFNs (like `ffn_k8`) + gated sum (like `combine` with
/// y = extra = 0) in one launch.
/// - `x` bf16[H] (= h2), `gates` f32[K] (0 for inactive k), `ids` as in `ffn_k8`.
/// - `zeros` bf16[K, H] all-zero resident tensor (initializes the DM row buffer so inactive rows
///   are finite zeros; never written).
/// - `out` bf16[H] = bf16( sum_k gates[k] * bf16(expert_k(x)) )  (f32 accumulation).
#[device(chip = 1, pe = 8)]
pub fn experts_b(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    gates: &HbmTensor<f32, Chip, m![K]>,
    wg_t: &HbmTensor<bf16, Chip, WGT>,
    wu_t: &HbmTensor<bf16, Chip, WGT>,
    wd_t: &HbmTensor<bf16, Chip, WDT>,
    ids: &HbmScalar<u64>,
    zeros: &HbmTensor<bf16, Chip, m![K, H]>,
    out: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let x_trf = load_x(device, x);
    let packed = ids.to_spm(&mut device.tdma);
    let active = (packed / 72057594037927936u64) as usize; // 2^56
    let mut ys: DmTensor<bf16, Chip, DC, DSO, m![K, H % 128]> = zeros.to_dm(&mut device.tdma);
    slot!(device, x_trf, wg_t, wu_t, wd_t, ys, packed, active, 0usize, 1u64);
    slot!(device, x_trf, wg_t, wu_t, wd_t, ys, packed, active, 1usize, 64u64);
    slot!(device, x_trf, wg_t, wu_t, wd_t, ys, packed, active, 2usize, 4096u64);
    slot!(device, x_trf, wg_t, wu_t, wd_t, ys, packed, active, 3usize, 262144u64);
    slot!(device, x_trf, wg_t, wu_t, wd_t, ys, packed, active, 4usize, 16777216u64);
    slot!(device, x_trf, wg_t, wu_t, wd_t, ys, packed, active, 5usize, 1073741824u64);
    slot!(device, x_trf, wg_t, wu_t, wd_t, ys, packed, active, 6usize, 68719476736u64);
    slot!(device, x_trf, wg_t, wu_t, wd_t, ys, packed, active, 7usize, 4398046511104u64);

    // gated sum, identical chain to combine::combine minus the (+0) residual add
    let g: DmTensor<f32, Chip, DC, DSO, m![K]> = gates.to_dm(&mut device.tdma);
    let g_vrf: VrfTensor<f32, Chip, DC, DSO, m![K]> = device
        .sub
        .begin(g.view())
        .fetch::<m![1], m![K]>()
        .collect::<m![1], m![K]>()
        .to_vrf();
    let o: DmTensor<bf16, Chip, DC, DSO, m![H % 128]> = device
        .main
        .begin(ys.view())
        .fetch::<m![H / 8 % 16, K], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 16, K], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 8 % 16, K, H / 4 % 2], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &g_vrf)
        .vector_intra_slice_reduce::<K, m![H / 8 % 16, H / 4 % 2], m![H % 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_concat::<m![H / 8 % 16], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    o.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

/// Expert-slot swap, device side: HBM->HBM copy of one expert (staging, already in WG/WD device
/// layout) into table slot `slot` (runtime). Use when the new expert was staged earlier.
#[device(chip = 1, pe = 8)]
pub fn swap_slot(
    device: &mut Device,
    sg: &HbmTensor<bf16, Chip, WG>,
    su: &HbmTensor<bf16, Chip, WG>,
    sd: &HbmTensor<bf16, Chip, WD>,
    slot: &HbmScalar<usize>,
    wg_t: &mut HbmTensor<bf16, Chip, WGT>,
    wu_t: &mut HbmTensor<bf16, Chip, WGT>,
    wd_t: &mut HbmTensor<bf16, Chip, WDT>,
) {
    let e = slot.to_spm(&mut device.tdma);
    sg.view().to_hbm_view(
        &mut device.tdma,
        wg_t.view_mut().tile::<m![E], 1, m![1 #{!} 32, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e),
    );
    su.view().to_hbm_view(
        &mut device.tdma,
        wu_t.view_mut().tile::<m![E], 1, m![1 #{!} 32, F / 320, H / 32, F / 8 % 40, H / 4 % 8, F % 8, H % 4]>(e),
    );
    sd.view().to_hbm_view(
        &mut device.tdma,
        wd_t.view_mut().tile::<m![E], 1, m![1 #{!} 32, H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4]>(e),
    );
}
