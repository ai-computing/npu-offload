//! LM head: logits[vocab ${VOCAB}, padded to ${LNC} x ${LF}] = W @ h, weight-streaming GEMV (the scheme of
//! `npu_kernels.rngd.gemv`): ${LNC} chunks of LF=${LF} rows, 2 clusters x (${A} row groups x ${NH32} column blocks)
//! slices of ${LR} rows x 32 columns; chunk c+1 streams while chunk c computes.
use furiosa_opt_std::prelude::*;

pub use crate::axes::{Chip, H};

axes![LF = ${LF}, LNC = ${LNC}];

pub type LGC = m![LF / ${LFH}];
pub type LGS = m![LF / ${LR} % ${A}, H / 32];
pub type LGSO = m![LF / ${LR} % ${A}, 1 # ${NH32}];
pub type LGElem = m![LF / 8 % ${LR8}, H / 4 % 8, LF % 8, H % 4];
/// HBM layout of the LM head weight: chunk-major. Host: `host::perm_gemv(w_padded, LNC * LF, H, LR)`.
pub type WLM = m![LNC, LF / ${LR}, H / 32, LF / 8 % ${LR8}, H / 4 % 8, LF % 8, H % 4];
/// logits, chunk-major = row order
pub type LOut = m![LNC, LF];

fn gemv_lm(
    device: &mut Device,
    w: &DmTensor<bf16, Chip, LGC, LGS, LGElem>,
    x_trf: &TrfTensor<bf16, Chip, LGC, LGS, m![1], m![H % 32]>,
) -> DmTensor<bf16, Chip, LGC, LGSO, m![LF % ${LR}]> {
    device
        .main
        .begin(w.view())
        .fetch::<m![LF / 8 % ${LR8}, H / 4 % 8, LF / 4 % 2], m![LF % 4, H % 4]>()
        .collect::<m![LF / 8 % ${LR8}, H / 4 % 8, LF / 4 % 2], m![LF % 4, H % 4]>()
        .contract_outer::<m![LF / 8 % ${LR8}, H / 4 % 8], m![LF % 8, H % 4], _, _, _>(x_trf)
        .contract_packet::<m![LF % 8]>()
        .contract_time::<m![LF / 8 % ${LR8}]>()
        .contract_lane::<m![LF / 8 % ${LR8}], m![LF % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<LGSO, m![LF / 8 % ${LR8}]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![LF % 8 # 16]>()
        .commit_trim::<m![LF % 8]>()
        .commit()
}

#[device(chip = 1, pe = 8)]
pub fn lm_head(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    w: &HbmTensor<bf16, Chip, WLM>,
    out: &mut HbmTensor<bf16, Chip, LOut>,
) {
    let xd: DmTensor<bf16, Chip, LGC, LGS, m![H % 32]> = x.to_dm(&mut device.tdma);
    let x_trf: TrfTensor<bf16, Chip, LGC, LGS, m![1], m![H % 32]> =
        device.sub.begin(xd.view()).fetch::<m![1], m![H % 32]>().collect::<m![H / 16 % 2], m![H % 16]>().to_trf();
${LM_CHUNKS}
}
