//! Routed-expert FFN kernel (H=${H}, F=${F}, table of E=${E} experts, up to K=${K} per launch), from the expert
//! generator (`npu_kernels.rngd.plan`): out[k] = Wd_k @ (silu(Wg_k @ x) * (Wu_k @ x)).
use furiosa_opt_std::prelude::*;
pub use crate::axes::{Chip, E, F, H, K};

// gate / up GEMV: 2 clusters (F halves) x (${EA} row groups x ${NH32} H-blocks) slices, each ${R} rows x 32 cols
pub type GC = m![F / ${FH}];
pub type GS = m![F / ${R} % ${EA}, H / 32];
pub type GSO = m![F / ${R} % ${EA}, 1 # ${NH32}];
pub type GElem = m![F / 8 % ${R8}, H / 4 % 8, F % 8, H % 4];
/// HBM layout of one Wg (or Wu)
pub type WG = m![F / ${R}, H / 32, F / 8 % ${R8}, H / 4 % 8, F % 8, H % 4];
// down GEMV: 2 clusters (H halves) x (${C} row groups x ${NFC} F-blocks) slices, each ${HR} rows x ${FC} cols
pub type DC = m![H / ${HH}];
pub type DS = m![H / ${HR} % ${C}, F / ${FC}];
pub type DSO = m![H / ${HR} % ${C}, 1 # ${NFC}];
pub type DElem = m![H / 8 % ${HR8}, F / 4 % ${FC4}, H % 8, F % 4];
/// HBM layout of one Wd
pub type WD = m![H / ${HR}, F / ${FC}, H / 8 % ${HR8}, F / 4 % ${FC4}, H % 8, F % 4];
/// resident expert tables
pub type WGT = m![E, F / ${R}, H / 32, F / 8 % ${R8}, H / 4 % 8, F % 8, H % 4];
pub type WDT = m![E, H / ${HR}, F / ${FC}, H / 8 % ${HR8}, F / 4 % ${FC4}, H % 8, F % 4];

pub fn gemv_g(
    device: &mut Device,
    w: &DmTensor<bf16, Chip, GC, GS, GElem>,
    x_trf: &TrfTensor<bf16, Chip, GC, GS, m![1], m![H % 32]>,
) -> DmTensor<bf16, Chip, GC, GSO, m![F % ${R}]> {
    device
        .main
        .begin(w.view())
        .fetch::<m![F / 8 % ${R8}, H / 4 % 8, F / 4 % 2], m![F % 4, H % 4]>()
        .collect::<m![F / 8 % ${R8}, H / 4 % 8, F / 4 % 2], m![F % 4, H % 4]>()
        .contract_outer::<m![F / 8 % ${R8}, H / 4 % 8], m![F % 8, H % 4], _, _, _>(x_trf)
        .contract_packet::<m![F % 8]>()
        .contract_time::<m![F / 8 % ${R8}]>()
        .contract_lane::<m![F / 8 % ${R8}], m![F % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<GSO, m![F / 8 % ${R8}]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![F % 8 # 16]>()
        .commit_trim::<m![F % 8]>()
        .commit()
}

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
    let silu: DmTensor<f32, Chip, GC, GSO, m![F % ${R}]> = device
        .sub
        .begin(gate.view())
        .fetch::<m![F / 8 % ${R8}], m![F % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![F / 8 % ${R8}], m![F % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![F / 4 % ${R4}], m![F % 4]>()
        .vector_stash()
        .vector_fp_unary(FpUnaryOp::Sigmoid)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_widen_concat::<m![F / 8 % ${R8}], m![F % 8]>()
        .vector_final()
        .commit_trim::<m![F % 8]>()
        .commit();
    let silu_vrf: VrfTensor<f32, Chip, GC, GSO, m![F % ${R}]> = device
        .sub
        .begin(silu.view())
        .fetch::<m![F / 8 % ${R8}], m![F % 8]>()
        .collect::<m![F / 8 % ${R8}], m![F % 8]>()
        .to_vrf();
    let h: DmTensor<bf16, Chip, GC, GSO, m![F % ${R}]> = device
        .main
        .begin(up.view())
        .fetch::<m![F / 8 % ${R8}], m![F % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![F / 8 % ${R8}], m![F % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![F / 4 % ${R4}], m![F % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &silu_vrf)
        .vector_widen_concat::<m![F / 8 % ${R8}], m![F % 8]>()
        .vector_final()
        .cast::<bf16, m![F % 8 # 16]>()
        .commit_trim::<m![F % 8]>()
        .commit();
    // redistribute h[F] to the down-proj placement (each slice needs its ${FC}-wide chunk)
    let h_hbm: HbmTensor<bf16, Chip, m![F]> = h.to_hbm(&mut device.tdma);
    let hx: DmTensor<bf16, Chip, DC, DS, m![F % ${FC}]> = h_hbm.to_dm(&mut device.tdma);
    let h_trf: TrfTensor<bf16, Chip, DC, DS, m![1], m![F % ${FC}]> = device
        .sub
        .begin(hx.view())
        .fetch::<m![F / 16 % ${FC16}], m![F % 16]>()
        .collect::<m![F / 16 % ${FC16}], m![F % 16]>()
        .to_trf();
    let y: DmTensor<bf16, Chip, DC, DSO, m![H % ${HR}]> = device
        .main
        .begin(wd.view())
        .fetch::<m![H / 8 % ${HR8}, F / 4 % ${FC4}, H / 4 % 2], m![H % 4, F % 4]>()
        .collect::<m![H / 8 % ${HR8}, F / 4 % ${FC4}, H / 4 % 2], m![H % 4, F % 4]>()
        .contract_outer::<m![H / 8 % ${HR8}, F / 4 % ${FC4}], m![H % 8, F % 4], _, _, _>(&h_trf)
        .contract_packet::<m![H % 8]>()
        .contract_time::<m![H / 8 % ${HR8}]>()
        .contract_lane::<m![H / 8 % ${HR8}], m![H % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<DSO, m![H / 8 % ${HR8}]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    y.view().to_hbm_view(&mut device.tdma, out);
}

pub fn load_x(device: &mut Device, x: &HbmTensor<bf16, Chip, m![H]>) -> TrfTensor<bf16, Chip, GC, GS, m![1], m![H % 32]> {
    let x: DmTensor<bf16, Chip, GC, GS, m![H % 32]> = x.to_dm(&mut device.tdma);
    device.sub.begin(x.view()).fetch::<m![1], m![H % 32]>().collect::<m![H / 16 % 2], m![H % 16]>().to_trf()
}

/// Up to K experts selected at runtime from the resident tables. `ids` packs the table slots in ${SB}-bit
/// fields (slot_k << ${SB}k) and the active count in bits 56..63; expert k's output goes to row k of `out`.
#[device(chip = 1, pe = 8)]
pub fn ffn_k(
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
${SLOTS}
}
