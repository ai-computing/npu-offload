use furiosa_opt_std::prelude::*;
use crate::axes::{Chip, H, K};

type C = m![H / 2048];
type S = m![H / 16 % 128, 1 # 2];

fn wr_helper<Ov: M>(device: &mut Device, a: &HbmTensor<bf16, Chip, m![H]>, out: HbmTensorViewMut<'_, bf16, Chip, Ov>) {
    let d: DmTensor<bf16, Chip, C, S, m![H % 16]> = a.to_dm(&mut device.tdma);
    d.view().to_hbm_view(&mut device.tdma, out);
}

/// two partial-row branches (mode 0 -> row 1, mode != 0 -> row 0)
#[device(chip = 1, pe = 8)]
pub fn p_u(device: &mut Device, mode: &HbmScalar<u64>, a: &HbmTensor<bf16, Chip, m![H]>, b: &mut HbmTensor<bf16, Chip, m![K, H]>) {
    let m = mode.to_spm(&mut device.tdma) as usize;
    if 0usize < m {
        wr_helper(device, a, b.view_mut().tile::<m![K], 1, m![1 #{!} 8, H]>(0));
    }
    if m < 1usize {
        wr_helper(device, a, b.view_mut().tile::<m![K], 1, m![1 #{!} 8, H]>(1));
    }
}
#[device(chip = 1, pe = 8)]
pub fn p_a(device: &mut Device, a: &HbmTensor<bf16, Chip, m![H]>, b: &mut HbmTensor<bf16, Chip, m![K, H]>) {
    wr_helper(device, a, b.view_mut().tile::<m![K], 1, m![1 #{!} 8, H]>(0));
}
#[device(chip = 1, pe = 8)]
pub fn p_b(device: &mut Device, a: &HbmTensor<bf16, Chip, m![H]>, b: &mut HbmTensor<bf16, Chip, m![K, H]>) {
    wr_helper(device, a, b.view_mut().tile::<m![K], 1, m![1 #{!} 8, H]>(1));
}
#[allow(dead_code)]
type KK = m![K];

use crate::attn::{D, N, T, Cm, KvCache, C1, SliceN32, store_kv};
use crate::axes::{U, Pk, R};
/// kv scatter inside a runtime if
#[device(chip = 1, pe = 8)]
pub fn p_kv(device: &mut Device, mode: &HbmScalar<u64>, off: &HbmTensor<i32, Chip, m![1]>, a: &HbmTensor<bf16, Chip, m![N, D]>, kc: &mut HbmTensor<bf16, Chip, KvCache>) {
    let m = mode.to_spm(&mut device.tdma) as usize;
    if m < 1usize {
        let v: DmTensor<bf16, Chip, C1, SliceN32, m![D]> = a.to_dm(&mut device.tdma);
        store_kv(device, off, v, kc);
    }
}
/// [U, Pk, R] row-0 write inside if
#[device(chip = 1, pe = 8)]
pub fn p_pack(device: &mut Device, mode: &HbmScalar<u64>, a: &HbmTensor<f32, Chip, m![Pk, R]>, o: &mut HbmTensor<f32, Chip, m![U, Pk, R]>) {
    let m = mode.to_spm(&mut device.tdma) as usize;
    if m < 1usize {
        let d: DmTensor<f32, Chip, m![1 # 2], m![1 # 256], m![Pk, R]> = a.to_dm(&mut device.tdma);
        d.view().to_hbm_view(&mut device.tdma, o.view_mut().tile::<m![U], 1, m![1 #{!} 2, Pk, R]>(0));
    }
}
/// [U, H] row-0 write inside if
#[device(chip = 1, pe = 8)]
pub fn p_uh(device: &mut Device, mode: &HbmScalar<u64>, a: &HbmTensor<bf16, Chip, m![H]>, o: &mut HbmTensor<bf16, Chip, m![U, H]>) {
    let m = mode.to_spm(&mut device.tdma) as usize;
    if m < 1usize {
        wr_helper(device, a, o.view_mut().tile::<m![U], 1, m![1 #{!} 2, H]>(0));
    }
}

/// kv scatter split into two D halves (partial writes) inside a runtime if
#[device(chip = 1, pe = 8)]
pub fn p_kv2(device: &mut Device, mode: &HbmScalar<u64>, off: &HbmTensor<i32, Chip, m![1]>, a: &HbmTensor<bf16, Chip, m![N, D]>, kc: &mut HbmTensor<bf16, Chip, KvCache>) {
    let m = mode.to_spm(&mut device.tdma) as usize;
    if m < 1usize {
        let v: DmTensor<bf16, Chip, C1, SliceN32, m![D]> = a.to_dm(&mut device.tdma);
        kc.view_mut()
            .tile::<m![D], 64, m![N, Cm, T, D = 64 #{!} 128]>(0)
            .scatter::<m![Cm, T], m![N, 1 # 32, D = 64]>()
            .by_byte_offsets::<m![1]>(off.view())
            .from_dm_view(&mut device.tdma, v.view().tile::<m![D], 64, m![D = 64 # 128]>(0));
    }
}
/// kv scatter split into two D halves (partial writes) inside a runtime if
#[device(chip = 1, pe = 8)]
pub fn p_kv3(device: &mut Device, mode: &HbmScalar<u64>, off: &HbmTensor<i32, Chip, m![1]>, a: &HbmTensor<bf16, Chip, m![N, D]>, kc: &mut HbmTensor<bf16, Chip, KvCache>) {
    let m = mode.to_spm(&mut device.tdma) as usize;
    if m < 1usize {
        let v: DmTensor<bf16, Chip, C1, SliceN32, m![D]> = a.to_dm(&mut device.tdma);
        kc.view_mut()
            .tile::<m![D], 64, m![N, Cm, T, D = 64 #{!} 128]>(0)
            .scatter::<m![Cm, T], m![N, 1 # 32, D = 64 #{!} 128]>()
            .by_byte_offsets::<m![1]>(off.view())
            .from_dm_view(&mut device.tdma, v.view().tile::<m![D], 64, m![D = 64 # 128]>(0));
    }
}
/// kv scatter split into two D halves (partial writes) inside a runtime if
#[device(chip = 1, pe = 8)]
pub fn p_kv4(device: &mut Device, mode: &HbmScalar<u64>, off: &HbmTensor<i32, Chip, m![1]>, a: &HbmTensor<bf16, Chip, m![N, D]>, kc: &mut HbmTensor<bf16, Chip, KvCache>) {
    let m = mode.to_spm(&mut device.tdma) as usize;
    if m < 1usize {
        let v: DmTensor<bf16, Chip, C1, SliceN32, m![D]> = a.to_dm(&mut device.tdma);
        kc.view_mut()
            .tile::<m![D], 64, m![N, Cm, T, D = 64 # 128]>(0)
            .scatter::<m![Cm, T], m![N, 1 # 32, D = 64]>()
            .by_byte_offsets::<m![1]>(off.view())
            .from_dm_view(&mut device.tdma, v.view().tile::<m![D], 64, m![D = 64 # 128]>(0));
    }
}

/// kv row write by a runtime (chunk, t) tile, inside a runtime if
#[device(chip = 1, pe = 8)]
pub fn p_kv5(device: &mut Device, mode: &HbmScalar<u64>, a: &HbmTensor<bf16, Chip, m![N, D]>, kc: &mut HbmTensor<bf16, Chip, KvCache>) {
    let m = mode.to_spm(&mut device.tdma) as usize;
    if 0usize < m {
        let pos = m - 1usize;
        let c = pos / 512usize;
        let t = pos % 512usize;
        let v: DmTensor<bf16, Chip, C1, SliceN32, m![D]> = a.to_dm(&mut device.tdma);
        v.view().to_hbm_view(
            &mut device.tdma,
            kc.view_mut()
                .tile::<m![Cm], 1, m![N, 1 #{!} 16, T, D]>(c)
                .tile::<m![T], 1, m![N, 1 #{!} 16, 1 #{!} 512, D]>(t),
        );
    }
}
