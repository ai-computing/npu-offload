//! Weight-streaming GEMVs for the Solar attention block (bf16, T=1, full chip).
//!
//! Same scheme as moe_opt `gemv_g`: rows split over 2 clusters x 2 row-groups, the
//! contraction dim split 32-wide over slices, each (cluster, slice) chunk of the weight
//! contiguous in HBM, x chunk (32 elements) resident in TRF, partial sums reduced with
//! the inter-slice reducer. Weights are streamed in 2048-row (16.8 MB) chunks so the
//! GEMV of chunk i overlaps the DMA of chunk i+1.
//!
//! All weight layouts are produced by `host::perm_gemv(w, rows, cols, 512)`.
use furiosa_opt_std::prelude::*;

use super::{Chip, H, Hc, Hr, Q, Qc, RW};

// ---- Q / KV projections: 2048-row chunks of [*, H=4096] ----------------------------------
pub type RC = m![RW / 1024];
pub type RS = m![RW / 512 % 2, H / 32];
pub type RSO = m![RW / 512 % 2, 1 # 128];
pub type RElem = m![RW / 8 % 64, H / 4 % 8, RW % 8, H % 4];
/// HBM layout of Wq ([8192, 4096] = 4 chunks of 2048 rows). Host: `host::perm_wq`.
pub type WQ = m![Qc, RW / 512, H / 32, RW / 8 % 64, H / 4 % 8, RW % 8, H % 4];
/// HBM layout of [Wk; Wv] ([2048, 4096], k rows first). Host: `host::perm_wkv`.
pub type WKV = m![RW / 512, H / 32, RW / 8 % 64, H / 4 % 8, RW % 8, H % 4];

// ---- O projection: Wo [H=4096, Q=8192] in 4 chunks of 1024 rows ---------------------------
pub(crate) type OC = m![Hr / 512];
pub(crate) type OS = m![Q / 32];
pub(crate) type OSO = m![1 # 256];
pub(crate) type OElem = m![Hr / 8 % 64, Q / 4 % 8, Hr % 8, Q % 4];
/// HBM layout of Wo. Host: `host::perm_wo`.
pub type WO = m![Hc, Hr / 512, Q / 32, Hr / 8 % 64, Q / 4 % 8, Hr % 8, Q % 4];

/// `y[RW] = W[RW, H] . h[H]` for one 2048-row chunk, committed into `out`.
pub(crate) fn gemv_r<Out: M>(
    device: &mut Device,
    w: &DmTensor<bf16, Chip, RC, RS, RElem>,
    x_trf: &TrfTensor<bf16, Chip, RC, RS, m![1], m![H % 32]>,
    out: DmTensorViewMut<'_, bf16, Chip, RC, RSO, Out>,
) {
    device
        .main
        .begin(w.view())
        .fetch::<m![RW / 8 % 64, H / 4 % 8, RW / 4 % 2], m![RW % 4, H % 4]>()
        .collect::<m![RW / 8 % 64, H / 4 % 8, RW / 4 % 2], m![RW % 4, H % 4]>()
        .contract_outer::<m![RW / 8 % 64, H / 4 % 8], m![RW % 8, H % 4], _, _, _>(x_trf)
        .contract_packet::<m![RW % 8]>()
        .contract_time::<m![RW / 8 % 64]>()
        .contract_lane::<m![RW / 8 % 64], m![RW % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<RSO, m![RW / 8 % 64]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![RW % 8 # 16]>()
        .commit_trim::<m![RW % 8]>()
        .commit_view(out);
}

/// `y[Hr] = Wo_chunk[Hr, Q] . a[Q]` for one 1024-row chunk of Wo, committed into `out`.
pub(crate) fn gemv_o<Out: M>(
    device: &mut Device,
    w: &DmTensor<bf16, Chip, OC, OS, OElem>,
    x_trf: &TrfTensor<bf16, Chip, OC, OS, m![1], m![Q % 32]>,
    out: DmTensorViewMut<'_, bf16, Chip, OC, OSO, Out>,
) {
    device
        .main
        .begin(w.view())
        .fetch::<m![Hr / 8 % 64, Q / 4 % 8, Hr / 4 % 2], m![Hr % 4, Q % 4]>()
        .collect::<m![Hr / 8 % 64, Q / 4 % 8, Hr / 4 % 2], m![Hr % 4, Q % 4]>()
        .contract_outer::<m![Hr / 8 % 64, Q / 4 % 8], m![Hr % 8, Q % 4], _, _, _>(x_trf)
        .contract_packet::<m![Hr % 8]>()
        .contract_time::<m![Hr / 8 % 64]>()
        .contract_lane::<m![Hr / 8 % 64], m![Hr % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<OSO, m![Hr / 8 % 64]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Hr % 8 # 16]>()
        .commit_trim::<m![Hr % 8]>()
        .commit_view(out);
}

/// Streams all of Wq (4 chunks) through `gemv_r`; result `[Qc, RW % 512]` per output slice.
pub fn proj_q(
    device: &mut Device,
    wq: &HbmTensor<bf16, Chip, WQ>,
    x_trf: &TrfTensor<bf16, Chip, RC, RS, m![1], m![H % 32]>,
) -> DmTensor<bf16, Chip, RC, RSO, m![Qc, RW % 512]> {
    let mut q: DmTensor<bf16, Chip, RC, RSO, m![Qc, RW % 512]> = DmTensor::new();
    #[unroll]
    for c in 0..4 {
        let w: DmTensor<bf16, Chip, RC, RS, RElem> = wq
            .view()
            .tile::<m![Qc], 1, m![1 # 4, RW / 512, H / 32, RW / 8 % 64, H / 4 % 8, RW % 8, H % 4]>(c)
            .to_dm(&mut device.tdma);
        gemv_r(device, &w, x_trf, q.view_mut().tile::<m![Qc], 1, m![1 #{!} 4, RW % 512]>(c));
    }
    q
}

/// Streams all of Wo (4 chunks); result `[Hc, Hr % 512]` per output slice (slice 0 of each cluster).
pub(crate) fn proj_o(
    device: &mut Device,
    wo: &HbmTensor<bf16, Chip, WO>,
    x_trf: &TrfTensor<bf16, Chip, OC, OS, m![1], m![Q % 32]>,
) -> DmTensor<bf16, Chip, OC, OSO, m![Hc, Hr % 512]> {
    let mut o: DmTensor<bf16, Chip, OC, OSO, m![Hc, Hr % 512]> = DmTensor::new();
    #[unroll]
    for c in 0..4 {
        let w: DmTensor<bf16, Chip, OC, OS, OElem> = wo
            .view()
            .tile::<m![Hc], 1, m![1 # 4, Hr / 512, Q / 32, Hr / 8 % 64, Q / 4 % 8, Hr % 8, Q % 4]>(c)
            .to_dm(&mut device.tdma);
        gemv_o(device, &w, x_trf, o.view_mut().tile::<m![Hc], 1, m![1 #{!} 4, Hr % 512]>(c));
    }
    o
}

/// One preloaded 1024-row chunk of Wo in DM.
pub(crate) type WoDm = DmTensor<bf16, Chip, OC, OS, OElem>;

/// DMA chunk `c` of Wo into DM (issue early to overlap with latency-bound work).
pub(crate) fn load_wo(device: &mut Device, wo: &HbmTensor<bf16, Chip, WO>, c: usize) -> WoDm {
    wo.view()
        .tile::<m![Hc], 1, m![1 # 4, Hr / 512, Q / 32, Hr / 8 % 64, Q / 4 % 8, Hr % 8, Q % 4]>(c)
        .to_dm(&mut device.tdma)
}

/// [`proj_o`] on preloaded chunks.
pub(crate) fn proj_o_pre(
    device: &mut Device,
    w: (&WoDm, &WoDm, &WoDm, &WoDm),
    x_trf: &TrfTensor<bf16, Chip, OC, OS, m![1], m![Q % 32]>,
) -> DmTensor<bf16, Chip, OC, OSO, m![Hc, Hr % 512]> {
    let mut o: DmTensor<bf16, Chip, OC, OSO, m![Hc, Hr % 512]> = DmTensor::new();
    gemv_o(device, w.0, x_trf, o.view_mut().tile::<m![Hc], 1, m![1 #{!} 4, Hr % 512]>(0));
    gemv_o(device, w.1, x_trf, o.view_mut().tile::<m![Hc], 1, m![1 #{!} 4, Hr % 512]>(1));
    gemv_o(device, w.2, x_trf, o.view_mut().tile::<m![Hc], 1, m![1 #{!} 4, Hr % 512]>(2));
    gemv_o(device, w.3, x_trf, o.view_mut().tile::<m![Hc], 1, m![1 #{!} 4, Hr % 512]>(3));
    o
}
