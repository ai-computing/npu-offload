use furiosa_opt_std::prelude::*;

axes![P8 = 2048];

type Chip = m![1];

/// Tiny kernel on the full 8-PE chip: out = in + 1 (i32, 2048 elements).
#[device(chip = 1, pe = 8)]
pub fn tiny_add(device: &mut Device, input: &HbmTensor<i32, Chip, m![P8]>, out: &mut HbmTensor<i32, Chip, m![P8]>) {
    let input_dm: DmTensor<i32, Chip, m![P8 / 1024], m![P8 / 4 % 256], m![P8 % 4]> = input.to_dm(&mut device.tdma);
    let result: DmTensor<i32, Chip, m![P8 / 1024], m![P8 / 4 % 256], m![P8 % 4]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![P8 % 4]>()
        .fetch_cast::<i32>()
        .collect::<m![1], m![P8 % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 1)
        .vector_final()
        .commit_trim::<m![P8 % 4]>()
        .commit();
    result.view().to_hbm_view(&mut device.tdma, out.view_mut());
}
