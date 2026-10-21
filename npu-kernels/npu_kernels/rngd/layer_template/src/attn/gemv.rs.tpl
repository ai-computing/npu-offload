//! Weight-streaming GEMVs of the attention block (bf16, one token, full chip), the scheme of the expert gate/up GEMV:
//! rows split over the 2 clusters x ${A} row groups, the contraction dim split 32-wide over slices, each (cluster,
//! slice) piece of the weight contiguous in HBM, x (32 elements per slice) in TRF, partial sums reduced with the
//! inter-slice reducer. Wq streams in ${Qc} chunks of RW = ${RW} rows, Wo in ${Hc} chunks of Hr = ${Hr} rows, so the
//! GEMV of chunk i overlaps the DMA of chunk i + 1. Host layouts: `host::perm_gemv(w, rows, cols, rows_per_slice)`.
use furiosa_opt_std::prelude::*;

use super::{Chip, H, Hc, Hr, Q, Qc, RW};
use crate::axes::NE;

// ---- q and [k; v]: chunks of RW = ${RW} rows of [*, H = ${H}], ${RPS} rows x 32 columns per slice ----
pub type RC = m![RW / ${RCD}];
pub type RS = m![RW / ${RPS} % ${A}, H / 32];
pub type RSO = m![RW / ${RPS} % ${A}, 1 # ${NH32}];
pub type RElem = m![RW / 8 % ${RPS8}, H / 4 % 8, RW % 8, H % 4];
/// HBM layout of Wq ([${Q}, ${H}] = ${Qc} chunks of ${RW} rows). Host: `host::perm_wq`.
pub type WQ = m![Qc, RW / ${RPS}, H / 32, RW / 8 % ${RPS8}, H / 4 % 8, RW % 8, H % 4];
/// HBM layout of [Wk; Wv] ([${RW}, ${H}], k rows first). Host: `host::perm_wkv`.
pub type WKV = m![RW / ${RPS}, H / 32, RW / 8 % ${RPS8}, H / 4 % 8, RW % 8, H % 4];

// ---- o_proj: Wo [H = ${H}, Q = ${Q}] in ${Hc} chunks of ${Hr} rows, ${RPSo} rows x 32 columns per slice ----
pub type OC = m![Hr / ${HrH}];
pub type OS = ${OS};
pub(crate) type OSO = ${OSO};
pub type OElem = m![Hr / 8 % ${RPSo8}, Q / 4 % 8, Hr % 8, Q % 4];
/// HBM layout of Wo. Host: `host::perm_wo`.
pub type WO = m![Hc, Hr / ${RPSo}, Q / 32, Hr / 8 % ${RPSo8}, Q / 4 % 8, Hr % 8, Q % 4];

// ---- router: Wr [NE = ${NE}, H] in one chunk, ${RR} rows x 32 columns per slice ----
pub type GC = m![NE / ${NEH}];
pub type GS = m![NE / ${RR} % ${A}, H / 32];
pub type GSO = m![NE / ${RR} % ${A}, 1 # ${NH32}];
pub type GElem = m![NE / 8 % ${RR8}, H / 4 % 8, NE % 8, H % 4];
/// HBM layout of the router weight. Host: `host::perm_wr`.
pub type WR = m![NE / ${RR}, H / 32, NE / 8 % ${RR8}, H / 4 % 8, NE % 8, H % 4];

/// `y[RW] = W[RW, H] . h[H]` for one chunk, committed into `out`.
pub(crate) fn gemv_r<Out: M>(
    device: &mut Device,
    w: &DmTensor<bf16, Chip, RC, RS, RElem>,
    x_trf: &TrfTensor<bf16, Chip, RC, RS, m![1], m![H % 32]>,
    out: DmTensorViewMut<'_, bf16, Chip, RC, RSO, Out>,
) {
    device
        .main
        .begin(w.view())
        .fetch::<m![RW / 8 % ${RPS8}, H / 4 % 8, RW / 4 % 2], m![RW % 4, H % 4]>()
        .collect::<m![RW / 8 % ${RPS8}, H / 4 % 8, RW / 4 % 2], m![RW % 4, H % 4]>()
        .contract_outer::<m![RW / 8 % ${RPS8}, H / 4 % 8], m![RW % 8, H % 4], _, _, _>(x_trf)
        .contract_packet::<m![RW % 8]>()
        .contract_time::<m![RW / 8 % ${RPS8}]>()
        .contract_lane::<m![RW / 8 % ${RPS8}], m![RW % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<RSO, m![RW / 8 % ${RPS8}]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![RW % 8 # 16]>()
        .commit_trim::<m![RW % 8]>()
        .commit_view(out);
}

/// `y[Hr] = Wo_chunk[Hr, Q] . a[Q]` for one chunk of Wo, committed into `out`.
pub(crate) fn gemv_o<Out: M>(
    device: &mut Device,
    w: &DmTensor<bf16, Chip, OC, OS, OElem>,
    x_trf: &TrfTensor<bf16, Chip, OC, OS, m![1], m![Q % 32]>,
    out: DmTensorViewMut<'_, bf16, Chip, OC, OSO, Out>,
) {
    device
        .main
        .begin(w.view())
        .fetch::<m![Hr / 8 % ${RPSo8}, Q / 4 % 8, Hr / 4 % 2], m![Hr % 4, Q % 4]>()
        .collect::<m![Hr / 8 % ${RPSo8}, Q / 4 % 8, Hr / 4 % 2], m![Hr % 4, Q % 4]>()
        .contract_outer::<m![Hr / 8 % ${RPSo8}, Q / 4 % 8], m![Hr % 8, Q % 4], _, _, _>(x_trf)
        .contract_packet::<m![Hr % 8]>()
        .contract_time::<m![Hr / 8 % ${RPSo8}]>()
        .contract_lane::<m![Hr / 8 % ${RPSo8}], m![Hr % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<OSO, m![Hr / 8 % ${RPSo8}]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Hr % 8 # 16]>()
        .commit_trim::<m![Hr % 8]>()
        .commit_view(out);
}

/// router logits `y[NE] = Wr . h[H]`.
pub(crate) fn gemv_router(
    device: &mut Device,
    w: &DmTensor<bf16, Chip, GC, GS, GElem>,
    x_trf: &TrfTensor<bf16, Chip, GC, GS, m![1], m![H % 32]>,
) -> DmTensor<bf16, Chip, GC, GSO, m![NE % ${RR}]> {
    device
        .main
        .begin(w.view())
        .fetch::<m![NE / 8 % ${RR8}, H / 4 % 8, NE / 4 % 2], m![NE % 4, H % 4]>()
        .collect::<m![NE / 8 % ${RR8}, H / 4 % 8, NE / 4 % 2], m![NE % 4, H % 4]>()
        .contract_outer::<m![NE / 8 % ${RR8}, H / 4 % 8], m![NE % 8, H % 4], _, _, _>(x_trf)
        .contract_packet::<m![NE % 8]>()
        .contract_time::<m![NE / 8 % ${RR8}]>()
        .contract_lane::<m![NE / 8 % ${RR8}], m![NE % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<GSO, m![NE / 8 % ${RR8}]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![NE % 8 # 16]>()
        .commit_trim::<m![NE % 8]>()
        .commit()
}

/// Streams all of Wq through `gemv_r`; result `[Qc, RW % ${RPS}]` per output slice.
pub fn proj_q(
    device: &mut Device,
    wq: &HbmTensor<bf16, Chip, WQ>,
    x_trf: &TrfTensor<bf16, Chip, RC, RS, m![1], m![H % 32]>,
) -> DmTensor<bf16, Chip, RC, RSO, m![Qc, RW % ${RPS}]> {
    let mut q: DmTensor<bf16, Chip, RC, RSO, m![Qc, RW % ${RPS}]> = DmTensor::new();
    #[unroll]
    for c in 0..${Qc} {
        let w: DmTensor<bf16, Chip, RC, RS, RElem> = wq
            .view()
            .tile::<m![Qc], 1, m![1 # ${Qc}, RW / ${RPS}, H / 32, RW / 8 % ${RPS8}, H / 4 % 8, RW % 8, H % 4]>(c)
            .to_dm(&mut device.tdma);
        gemv_r(device, &w, x_trf, q.view_mut().tile::<m![Qc], 1, m![1 #{!} ${Qc}, RW % ${RPS}]>(c));
    }
    q
}

/// Streams all of Wo; result `[Hc, Hr % ${RPSo}]` per output slice.
pub(crate) fn proj_o(
    device: &mut Device,
    wo: &HbmTensor<bf16, Chip, WO>,
    x_trf: &TrfTensor<bf16, Chip, OC, OS, m![1], m![Q % 32]>,
) -> DmTensor<bf16, Chip, OC, OSO, m![Hc, Hr % ${RPSo}]> {
    let mut o: DmTensor<bf16, Chip, OC, OSO, m![Hc, Hr % ${RPSo}]> = DmTensor::new();
    #[unroll]
    for c in 0..${Hc} {
        let w: DmTensor<bf16, Chip, OC, OS, OElem> = wo
            .view()
            .tile::<m![Hc], 1, m![1 # ${Hc}, Hr / ${RPSo}, Q / 32, Hr / 8 % ${RPSo8}, Q / 4 % 8, Hr % 8, Q % 4]>(c)
            .to_dm(&mut device.tdma);
        gemv_o(device, &w, x_trf, o.view_mut().tile::<m![Hc], 1, m![1 #{!} ${Hc}, Hr % ${RPSo}]>(c));
    }
    o
}
