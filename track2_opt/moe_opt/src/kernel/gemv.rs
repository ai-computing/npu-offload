//! Single-expert GEMV kernels, bf16, full chip (8 PEs = 2 clusters x 256 slices).
//! Weights-stationary-per-launch: W goes HBM -> DM -> TRF (8 rows per slice), x streams.
use furiosa_opt_std::prelude::*;

// A = gate|up rows (2*F), H = hidden, F = expert width
axes![A = 2560, H = 4096, F = 1280];

pub type Chip = m![1];

/// y[A] = W[A, H] . x[H]   (gate|up concat, 20 MB)
#[device(chip = 1, pe = 8)]
pub fn gemv_gu(
    device: &mut Device,
    w: &HbmTensor<bf16, Chip, m![A, H]>,
    x: &HbmTensor<bf16, Chip, m![H]>,
    out: &mut HbmTensor<bf16, Chip, m![A]>,
) {
    type Cluster = m![A # 4096 / 2048];
    type Slice = m![A # 4096 / 8 % 256];
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = x.to_dm(&mut device.tdma);
    let w: DmTensor<bf16, Chip, Cluster, Slice, m![A % 8, H]> = w.to_dm(&mut device.tdma);
    let w_trf: TrfTensor<bf16, Chip, Cluster, Slice, m![A % 8], m![H]> = device
        .sub
        .begin(w.view())
        .fetch::<m![A % 8, H / 16], m![H % 16]>()
        .collect::<m![A % 8, H / 16], m![H % 16]>()
        .to_trf();
    let y: DmTensor<bf16, Chip, Cluster, Slice, m![A % 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .contract_outer::<m![H / 32], m![H % 32], _, _, _>(&w_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![1]>()
        .contract_lane::<m![1], m![A % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![A % 8 # 16]>()
        .commit_trim::<m![A % 8]>()
        .commit();
    y.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

/// y[H] = W[H, F] . x[F]   (down proj, 10 MB)
#[device(chip = 1, pe = 8)]
pub fn gemv_down(
    device: &mut Device,
    w: &HbmTensor<bf16, Chip, m![H, F]>,
    x: &HbmTensor<bf16, Chip, m![F]>,
    out: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    type Cluster = m![H / 2048];
    type Slice = m![H / 8 % 256];
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![F]> = x.to_dm(&mut device.tdma);
    let w: DmTensor<bf16, Chip, Cluster, Slice, m![H % 8, F]> = w.to_dm(&mut device.tdma);
    let w_trf: TrfTensor<bf16, Chip, Cluster, Slice, m![H % 8], m![F]> = device
        .sub
        .begin(w.view())
        .fetch::<m![H % 8, F / 16], m![F % 16]>()
        .collect::<m![H % 8, F / 16], m![F % 16]>()
        .to_trf();
    let y: DmTensor<bf16, Chip, Cluster, Slice, m![H % 8]> = device
        .main
        .begin(x.view())
        .fetch::<m![F / 16], m![F % 16]>()
        .collect::<m![F / 16], m![F % 16]>()
        .contract_outer::<m![F / 32], m![F % 32], _, _, _>(&w_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![1]>()
        .contract_lane::<m![1], m![H % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    y.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

/// HBM layout of W for gemv_gu2: each (cluster, slice) chunk [A%640, H%32] is contiguous (40 KB).
pub type WGu2 = m![A / 640, H / 32, A % 640, H % 32];

/// Variant 2: W streams through main from DM (no TRF weight load), x chunk (32 elts) in TRF,
/// partial sums reduced across the 128 H-slices with the inter-slice reducer.
#[device(chip = 1, pe = 8)]
pub fn gemv_gu2(
    device: &mut Device,
    w: &HbmTensor<bf16, Chip, WGu2>,
    x: &HbmTensor<bf16, Chip, m![H]>,
    out: &mut HbmTensor<bf16, Chip, m![A, 1 # 4]>,
) {
    type Cluster = m![A / 1280];
    type Slice = m![A / 640 % 2, H / 32];
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H % 32]> = x.to_dm(&mut device.tdma);
    let w: DmTensor<bf16, Chip, Cluster, Slice, m![A % 640, H % 32]> = w.to_dm(&mut device.tdma);
    let x_trf: TrfTensor<bf16, Chip, Cluster, Slice, m![1], m![H % 32]> = device
        .sub
        .begin(x.view())
        .fetch::<m![1], m![H % 32]>()
        .collect::<m![H / 16 % 2], m![H % 16]>()
        .to_trf();
    let y: DmTensor<bf16, Chip, Cluster, m![A / 640 % 2, 1 # 128], m![A % 640, 1 # 4]> = device
        .main
        .begin(w.view())
        .fetch::<m![A % 640, H / 16 % 2], m![H % 16]>()
        .collect::<m![A % 640, H / 16 % 2], m![H % 16]>()
        .contract_outer::<m![A % 640], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![A % 640]>()
        .contract_lane::<m![A % 640], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![A / 640 % 2, 1 # 128], m![A % 640]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .commit_trim::<m![1 # 4]>()
        .commit();
    y.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

/// HBM layout of W for gemv_gu3: per (cluster, slice) a contiguous 40 KB chunk ordered
/// [row-group, h-group, row%8, h%4] so each 64 B packet holds 8 rows x 4 columns.
pub type WGu3 = m![A / 640, H / 32, A / 8 % 80, H / 4 % 8, A % 8, H % 4];

/// Variant 3: like gu2 but the packet carries 8 output rows, so the output is dense.
#[device(chip = 1, pe = 8)]
pub fn gemv_gu3(
    device: &mut Device,
    w: &HbmTensor<bf16, Chip, WGu3>,
    x: &HbmTensor<bf16, Chip, m![H]>,
    out: &mut HbmTensor<bf16, Chip, m![A]>,
) {
    type Cluster = m![A / 1280];
    type Slice = m![A / 640 % 2, H / 32];
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H % 32]> = x.to_dm(&mut device.tdma);
    let w: DmTensor<bf16, Chip, Cluster, Slice, m![A / 8 % 80, H / 4 % 8, A % 8, H % 4]> = w.to_dm(&mut device.tdma);
    let x_trf: TrfTensor<bf16, Chip, Cluster, Slice, m![1], m![H % 32]> = device
        .sub
        .begin(x.view())
        .fetch::<m![1], m![H % 32]>()
        .collect::<m![H / 16 % 2], m![H % 16]>()
        .to_trf();
    let y: DmTensor<bf16, Chip, Cluster, m![A / 640 % 2, 1 # 128], m![A % 640]> = device
        .main
        .begin(w.view())
        .fetch::<m![A / 8 % 80, H / 4 % 8, A / 4 % 2], m![A % 4, H % 4]>()
        .collect::<m![A / 8 % 80, H / 4 % 8, A / 4 % 2], m![A % 4, H % 4]>()
        .contract_outer::<m![A / 8 % 80, H / 4 % 8], m![A % 8, H % 4], _, _, _>(&x_trf)
        .contract_packet::<m![A % 8]>()
        .contract_time::<m![A / 8 % 80]>()
        .contract_lane::<m![A / 8 % 80], m![A % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<m![A / 640 % 2, 1 # 128], m![A / 8 % 80]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![A % 8 # 16]>()
        .commit_trim::<m![A % 8]>()
        .commit();
    y.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

/// HBM layout of Wd for gemv_dn3: per (cluster, slice) contiguous 20 KB chunk.
pub type WDn3 = m![H / 128, F / 80, H / 8 % 16, F / 4 % 20, H % 8, F % 4];

/// Down projection y[H] = Wd[H, F] . x[F], same streaming scheme as gemv_gu3.
#[device(chip = 1, pe = 8)]
pub fn gemv_dn3(
    device: &mut Device,
    w: &HbmTensor<bf16, Chip, WDn3>,
    x: &HbmTensor<bf16, Chip, m![F]>,
    out: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    type Cluster = m![H / 2048];
    type Slice = m![H / 128 % 16, F / 80];
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![F % 80]> = x.to_dm(&mut device.tdma);
    let w: DmTensor<bf16, Chip, Cluster, Slice, m![H / 8 % 16, F / 4 % 20, H % 8, F % 4]> = w.to_dm(&mut device.tdma);
    let x_trf: TrfTensor<bf16, Chip, Cluster, Slice, m![1], m![F % 80]> = device
        .sub
        .begin(x.view())
        .fetch::<m![F / 16 % 5], m![F % 16]>()
        .collect::<m![F / 16 % 5], m![F % 16]>()
        .to_trf();
    let y: DmTensor<bf16, Chip, Cluster, m![H / 128 % 16, 1 # 16], m![H % 128]> = device
        .main
        .begin(w.view())
        .fetch::<m![H / 8 % 16, F / 4 % 20, H / 4 % 2], m![H % 4, F % 4]>()
        .collect::<m![H / 8 % 16, F / 4 % 20, H / 4 % 2], m![H % 4, F % 4]>()
        .contract_outer::<m![H / 8 % 16, F / 4 % 20], m![H % 8, F % 4], _, _, _>(&x_trf)
        .contract_packet::<m![H % 8]>()
        .contract_time::<m![H / 8 % 16]>()
        .contract_lane::<m![H / 8 % 16], m![H % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce::<m![H / 128 % 16, 1 # 16], m![H / 8 % 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    y.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
