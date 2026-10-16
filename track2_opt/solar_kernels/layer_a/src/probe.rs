//! NPU timing probes (not part of the integration surface).
use furiosa_opt_std::prelude::*;

use crate::attn::gemv::{RC, RSO, WQ, proj_q};
use crate::attn::{Qc, RW, rmsnorm};
use crate::axes::{Chip, H};

/// Wq streaming GEMV only (64 MB).
#[device(chip = 1, pe = 8)]
pub fn probe_q(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    w_in: &HbmTensor<bf16, Chip, m![H]>,
    wq: &HbmTensor<bf16, Chip, WQ>,
    q_out: &mut HbmTensor<bf16, Chip, m![Qc, RW]>,
) {
    let u = rmsnorm::u_trf(device, x, w_in);
    let q: DmTensor<bf16, Chip, RC, RSO, m![Qc, RW % 512]> = proj_q(device, wq, &u);
    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
}

/// Launch floor: one tiny DMA round trip.
#[device(chip = 1, pe = 8)]
pub fn probe_floor(device: &mut Device, x: &HbmTensor<bf16, Chip, m![H]>, y: &mut HbmTensor<bf16, Chip, m![H]>) {
    let d: DmTensor<bf16, Chip, m![H / 2048], m![H / 128 % 16, 1 # 16], m![H % 128]> = x.to_dm(&mut device.tdma);
    d.view().to_hbm_view(&mut device.tdma, y.view_mut());
}

axes![QA = 8192];

/// Wq as 2 chunks of 4096 rows (64 KB contiguous per slice). HBM layout = perm_gemv(rps = 1024).
pub type WQ2 = m![QA / 4096, QA / 1024 % 4, H / 32, QA / 8 % 128, H / 4 % 8, QA % 8, H % 4];
type C2 = m![QA / 2048 % 2];
type S2 = m![QA / 1024 % 2, H / 32];
type S2O = m![QA / 1024 % 2, 1 # 128];
type E2 = m![QA / 8 % 128, H / 4 % 8, QA % 8, H % 4];

#[device(chip = 1, pe = 8)]
pub fn probe_q2(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    wq: &HbmTensor<bf16, Chip, WQ2>,
    q_out: &mut HbmTensor<bf16, Chip, m![QA]>,
) {
    let xd: DmTensor<bf16, Chip, C2, S2, m![H % 32]> = x.to_dm(&mut device.tdma);
    let x_trf: TrfTensor<bf16, Chip, C2, S2, m![1], m![H % 32]> =
        device.sub.begin(xd.view()).fetch::<m![1], m![H % 32]>().collect::<m![H / 16 % 2], m![H % 16]>().to_trf();
    let mut q: DmTensor<bf16, Chip, C2, S2O, m![QA / 4096, QA % 1024]> = DmTensor::new();
    #[unroll]
    for c in 0..2 {
        let w: DmTensor<bf16, Chip, C2, S2, E2> = wq
            .view()
            .tile::<m![QA / 4096], 1, m![1 # 2, QA / 1024 % 4, H / 32, QA / 8 % 128, H / 4 % 8, QA % 8, H % 4]>(c)
            .to_dm(&mut device.tdma);
        device
            .main
            .begin(w.view())
            .fetch::<m![QA / 8 % 128, H / 4 % 8, QA / 4 % 2], m![QA % 4, H % 4]>()
            .collect::<m![QA / 8 % 128, H / 4 % 8, QA / 4 % 2], m![QA % 4, H % 4]>()
            .contract_outer::<m![QA / 8 % 128, H / 4 % 8], m![QA % 8, H % 4], _, _, _>(&x_trf)
            .contract_packet::<m![QA % 8]>()
            .contract_time::<m![QA / 8 % 128]>()
            .contract_lane::<m![QA / 8 % 128], m![QA % 8]>(LaneMode::Sequential)
            .vector_init()
            .vector_inter_slice_reduce::<S2O, m![QA / 8 % 128]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![QA % 8 # 16]>()
            .commit_trim::<m![QA % 8]>()
            .commit_view(q.view_mut().tile::<m![QA / 4096], 1, m![1 #{!} 2, QA % 1024]>(c));
    }
    q.view().to_hbm_view(&mut device.tdma, q_out.view_mut());
}
