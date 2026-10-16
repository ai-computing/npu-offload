//! Unified chip-A program on RESIDENT layer weights: one `#[device]` fn per context-chunk count K,
//! `unified_res_cK`, whose runtime `mode` scalar selects one of three bodies:
//!
//! * `mode >= 2` — **layer** step for the token at context position `pos = mode - 2`: the
//!   `layer_a_res_cK` body (attention + shared/router weights read from the resident DM slot, no
//!   weight DMA) with the `unified_a` output plumbing (packed in/out, `[U, H]` y ping-pong rows,
//!   h2 row for the ffn mode, runtime-tile k/v write at `pos`).
//! * `mode == 0` — **ffn**: the `ffn_k8` body (E = 24 hot-expert tables in HBM, `ids` scalar),
//!   input = row 0 of `h2s`, output = rows of `expert_out`.
//! * `mode == 1` — **prefetch**: HBM -> resident DM copy of the layer weights given in `pw`
//!   (the next layer's Wq/[Wk;Wv]/Wo/Wsg/Wsu/Wsd/Wr), = `prefetch_layer`.
//!
//! furiosa-opt 0.9 restriction (see `solar_unified_a::unified`): a write to a kernel parameter
//! inside a runtime `if` must be a partial (`#{!}` tile) write. This holds for runtime-owned DM
//! parameters too: a whole-tensor `to_dm_view(r.view_mut())` into a resident `DmTensor` inside the
//! `if` fails with "visa: writable parameter output is not registered". Workaround (`copy2`): every
//! resident tensor is written as two `#{!}` half tiles of its Element axis (`DmTensorViewMut::tile`);
//! `slice_tile` halves are not lowered yet ("tiling a non-in-slice axis"). Cost: ~4 us over
//! `prefetch_layer` (26 instead of 13 DMAs) + ~4 us branch/program overhead (NPU: 164.9 vs 156.6 us).
//!
//! Host recipe: allocate ONE `AttnSlot` + `SharedSlot` (`DmTensor::alloc`, 350 KiB/slice) once and
//! pass `slot.w()` to every launch. Per decode layer l: mode_layer(pos) [critical path, needs slot
//! = layer l] -> MODE_FFN (tables of l, ids) -> MODE_PREFETCH (pw = layer l+1). Prime the slot with
//! one MODE_PREFETCH for layer 0. In ffn/prefetch modes the layer-mode tensors are unused but must
//! still be valid tensors of the right type (pass the current layer's).
use furiosa_opt_std::prelude::*;

#[allow(unused_imports)]
use crate::attn::gemv::{RC, RElem, RS};
use crate::attn::gemv::{WKV, WO, WQ};
use crate::attn::{D, G, N, KvCache, Masks, attention, kv_chunk, o_proj_epilogue_pre, qkv_rope_res_at};
use crate::attn::{Hc, Hr, Q, Qc, RW};
use crate::axes::{Chip, E, F, H, K, Pk, R, U};
use crate::dense::shared::{SharedRef, WD, WG, WR, shared_router_pre};
use crate::ffn::{WDT, WGT, expert, ffn_k8_body, load_x};
use crate::layer::{AttnResMut, PackDm, SharedResMut, combine_packed, mask_get, mask_tbl, store_h2_f32};

/// `[U, H]` bf16 (row 0 live).
pub type UH = m![U, H];
/// `[U, Pk, R]` f32 (row 0 = the `[Pk, R]` pack).
pub type UPack = m![U, Pk, R];

/// Prefetch source: one layer's HBM weights (same layouts as `layer_a_cK` / `prefetch_layer`).
/// Only read in prefetch mode; any valid layer's tensors may be passed in the other modes.
#[derive(DeviceSend)]
pub struct PfW<'a> {
    pub wq: &'a HbmTensor<bf16, Chip, WQ>,
    pub wkv: &'a HbmTensor<bf16, Chip, WKV>,
    pub wo: &'a HbmTensor<bf16, Chip, WO>,
    pub wg: &'a HbmTensor<bf16, Chip, WG>,
    pub wu: &'a HbmTensor<bf16, Chip, WG>,
    pub wd: &'a HbmTensor<bf16, Chip, WD>,
    pub wr: &'a HbmTensor<bf16, Chip, WR>,
}
/// Per-layer routed-expert tables (E = 24 hot experts, `ffn_k8` layouts).
#[derive(DeviceSend)]
pub struct ExpertT<'a> {
    pub wg_t: &'a HbmTensor<bf16, Chip, WGT>,
    pub wu_t: &'a HbmTensor<bf16, Chip, WGT>,
    pub wd_t: &'a HbmTensor<bf16, Chip, WDT>,
}

/// Mode values.
pub const MODE_FFN: u64 = 0;
pub const MODE_PREFETCH: u64 = 1;
/// Layer mode for context position `pos`.
pub const fn mode_layer(pos: usize) -> u64 {
    pos as u64 + 2
}

/// Copies `$src` (HBM view) into `$dst` (resident DM) as two `#{!}` half tiles along `$ix`.
/// A whole-tensor write to a DM parameter inside a runtime `if` is rejected by furiosa-opt 0.9
/// ("visa: writable parameter output is not registered"); partial-tile writes are accepted.
macro_rules! copy2 {
    ($device:ident, $src:expr, $dst:ident, $ix:ty, [$($se:tt)*], [$($de:tt)*]) => {
        $src.tile::<$ix, 1, m![$($se)*]>(0)
            .to_dm_view(&mut $device.tdma, $dst.view_mut().tile::<$ix, 1, m![$($de)*]>(0));
        $src.tile::<$ix, 1, m![$($se)*]>(1)
            .to_dm_view(&mut $device.tdma, $dst.view_mut().tile::<$ix, 1, m![$($de)*]>(1));
    };
}

/// `prefetch_layer` body with every resident write split into two partial tiles (see `copy2`).
macro_rules! prefetch_partial {
    ($device:ident, $wq:ident, $wkv:ident, $wo:ident, $wg:ident, $wu:ident, $wd:ident, $wr:ident,
     $q0:ident, $q1:ident, $q2:ident, $q3:ident, $kv:ident, $o0:ident, $o1:ident, $o2:ident, $o3:ident,
     $g:ident, $u:ident, $d:ident, $r:ident) => {
        copy2!($device, $wkv.view(), $kv, m![RW / 256 % 2],
            [RW / 512, H / 32, 1 # 2, RW / 8 % 32, H / 4 % 8, RW % 8, H % 4],
            [1 #{!} 2, RW / 8 % 32, H / 4 % 8, RW % 8, H % 4]);
        copy2!($device, $wq.view().tile::<m![Qc], 1, m![1 # 4, RW / 512, H / 32, RW / 8 % 64, H / 4 % 8, RW % 8, H % 4]>(0), $q0, m![RW / 256 % 2],
            [1 # 4, RW / 512, H / 32, 1 # 2, RW / 8 % 32, H / 4 % 8, RW % 8, H % 4],
            [1 #{!} 2, RW / 8 % 32, H / 4 % 8, RW % 8, H % 4]);
        copy2!($device, $wq.view().tile::<m![Qc], 1, m![1 # 4, RW / 512, H / 32, RW / 8 % 64, H / 4 % 8, RW % 8, H % 4]>(1), $q1, m![RW / 256 % 2],
            [1 # 4, RW / 512, H / 32, 1 # 2, RW / 8 % 32, H / 4 % 8, RW % 8, H % 4],
            [1 #{!} 2, RW / 8 % 32, H / 4 % 8, RW % 8, H % 4]);
        copy2!($device, $wq.view().tile::<m![Qc], 1, m![1 # 4, RW / 512, H / 32, RW / 8 % 64, H / 4 % 8, RW % 8, H % 4]>(2), $q2, m![RW / 256 % 2],
            [1 # 4, RW / 512, H / 32, 1 # 2, RW / 8 % 32, H / 4 % 8, RW % 8, H % 4],
            [1 #{!} 2, RW / 8 % 32, H / 4 % 8, RW % 8, H % 4]);
        copy2!($device, $wq.view().tile::<m![Qc], 1, m![1 # 4, RW / 512, H / 32, RW / 8 % 64, H / 4 % 8, RW % 8, H % 4]>(3), $q3, m![RW / 256 % 2],
            [1 # 4, RW / 512, H / 32, 1 # 2, RW / 8 % 32, H / 4 % 8, RW % 8, H % 4],
            [1 #{!} 2, RW / 8 % 32, H / 4 % 8, RW % 8, H % 4]);
        copy2!($device, $wo.view().tile::<m![Hc], 1, m![1 # 4, Hr / 512, Q / 32, Hr / 8 % 64, Q / 4 % 8, Hr % 8, Q % 4]>(0), $o0, m![Hr / 256 % 2],
            [1 # 4, Hr / 512, Q / 32, 1 # 2, Hr / 8 % 32, Q / 4 % 8, Hr % 8, Q % 4],
            [1 #{!} 2, Hr / 8 % 32, Q / 4 % 8, Hr % 8, Q % 4]);
        copy2!($device, $wo.view().tile::<m![Hc], 1, m![1 # 4, Hr / 512, Q / 32, Hr / 8 % 64, Q / 4 % 8, Hr % 8, Q % 4]>(1), $o1, m![Hr / 256 % 2],
            [1 # 4, Hr / 512, Q / 32, 1 # 2, Hr / 8 % 32, Q / 4 % 8, Hr % 8, Q % 4],
            [1 #{!} 2, Hr / 8 % 32, Q / 4 % 8, Hr % 8, Q % 4]);
        copy2!($device, $wo.view().tile::<m![Hc], 1, m![1 # 4, Hr / 512, Q / 32, Hr / 8 % 64, Q / 4 % 8, Hr % 8, Q % 4]>(2), $o2, m![Hr / 256 % 2],
            [1 # 4, Hr / 512, Q / 32, 1 # 2, Hr / 8 % 32, Q / 4 % 8, Hr % 8, Q % 4],
            [1 #{!} 2, Hr / 8 % 32, Q / 4 % 8, Hr % 8, Q % 4]);
        copy2!($device, $wo.view().tile::<m![Hc], 1, m![1 # 4, Hr / 512, Q / 32, Hr / 8 % 64, Q / 4 % 8, Hr % 8, Q % 4]>(3), $o3, m![Hr / 256 % 2],
            [1 # 4, Hr / 512, Q / 32, 1 # 2, Hr / 8 % 32, Q / 4 % 8, Hr % 8, Q % 4],
            [1 #{!} 2, Hr / 8 % 32, Q / 4 % 8, Hr % 8, Q % 4]);
        copy2!($device, $wr.view(), $r, m![R / 16 % 2],
            [R / 32, H / 32, 1 # 2, R / 8 % 2, H / 4 % 8, R % 8, H % 4],
            [1 #{!} 2, R / 8 % 2, H / 4 % 8, R % 8, H % 4]);
        copy2!($device, $wg.view(), $g, m![F / 160 % 2],
            [F / 320, H / 32, 1 # 2, F / 8 % 20, H / 4 % 8, F % 8, H % 4],
            [1 #{!} 2, F / 8 % 20, H / 4 % 8, F % 8, H % 4]);
        copy2!($device, $wu.view(), $u, m![F / 160 % 2],
            [F / 320, H / 32, 1 # 2, F / 8 % 20, H / 4 % 8, F % 8, H % 4],
            [1 #{!} 2, F / 8 % 20, H / 4 % 8, F % 8, H % 4]);
        copy2!($device, $wd.view(), $d, m![H / 64 % 2],
            [H / 128, F / 80, 1 # 2, H / 8 % 8, F / 4 % 20, H % 8, F % 4],
            [1 #{!} 2, H / 8 % 8, F / 4 % 20, H % 8, F % 4]);
    };
}

macro_rules! unified_res_k {
    ($name:ident, $mm:ident, [$($c:literal),*]) => {
        /// mode >= 2: layer step at pos = mode - 2 on the resident weights; mode 0: ffn_k8;
        /// mode 1: prefetch `pw` into the resident slot (see module docs).
        #[device(chip = 1, pe = 8)]
        #[expect(clippy::too_many_arguments)]
        pub fn $name(
            device: &mut Device,
            mode: &HbmScalar<u64>,
            y_in: &HbmTensor<bf16, Chip, UH>,
            expert_out: &mut HbmTensor<bf16, Chip, m![K, H]>,
            inp: &HbmTensor<f32, Chip, m![Pk, R]>,
            w_in: &HbmTensor<bf16, Chip, m![H]>,
            w_post: &HbmTensor<bf16, Chip, m![H]>,
            cos: &HbmTensor<bf16, Chip, m![D]>,
            sin: &HbmTensor<bf16, Chip, m![D]>,
            k_cache: &mut HbmTensor<bf16, Chip, KvCache>,
            v_cache: &mut HbmTensor<bf16, Chip, KvCache>,
            masks: &HbmTensor<f32, Chip, Masks>,
            pw: PfW<'_>,
            a: AttnResMut<'_>,
            s: SharedResMut<'_>,
            et: ExpertT<'_>,
            ids: &HbmScalar<u64>,
            outs: &mut HbmTensor<f32, Chip, UPack>,
            ys: &mut HbmTensor<bf16, Chip, UH>,
            h2s: &mut HbmTensor<bf16, Chip, UH>,
        ) {
            let m = mode.to_spm(&mut device.tdma) as usize;
            let AttnResMut { q0: rq0, q1: rq1, q2: rq2, q3: rq3, kv: rkv, o0: ro0, o1: ro1, o2: ro2, o3: ro3 } = a;
            let SharedResMut { g: rg, u: ru, d: rd, r: rr } = s;
            if 1usize < m {
                let pos = m - 2usize;
                let kv_pos = (pos / 512usize, pos % 512usize);
                let eo: &HbmTensor<bf16, Chip, m![K, H]> = expert_out;
                // ---- layer_a_res_cK body (resident weights), unified_a output plumbing ----
                let xd = combine_packed(device, y_in.view().tile::<m![U], 1, m![1 # 2, H]>(0), eo, inp);
                let x_t: HbmTensor<bf16, Chip, m![H]> = xd.to_hbm(&mut device.tdma);
                let x: &HbmTensor<bf16, Chip, m![H]> = &x_t;
                let q = qkv_rope_res_at(device, x, w_in, (&*rq0, &*rq1, &*rq2, &*rq3), &*rkv, kv_pos, cos, sin, k_cache, v_cache);
                let q_hbm: HbmTensor<bf16, Chip, m![N, G, D]> = q.to_hbm(&mut device.tdma);
                let q = attention::q_trf_from_hbm(device, &q_hbm);
                #[allow(clippy::let_unit_value)]
                let tbl = mask_tbl!($mm, device, masks);
                let mask = mask_get!($mm, device, masks, tbl, 0);
                #[allow(unused_mut)]
                let (mut max, mut sum, mut o) =
                    attention::first_dm(device, &q, kv_chunk(k_cache, 0), kv_chunk(v_cache, 0), &mask);
                $(
                    let mask = mask_get!($mm, device, masks, tbl, $c);
                    let (m2, s2, o2) = attention::next_dm(
                        device, &q, kv_chunk(k_cache, $c), kv_chunk(v_cache, $c), &mask, &max, &sum, &o,
                    );
                    max = m2;
                    sum = s2;
                    o = o2;
                )*
                let _ = &max;
                let a = attention::norm_dm(device, &o, &sum);
                let (x2d, h2d) = o_proj_epilogue_pre(device, a, x, (&*ro0, &*ro1, &*ro2, &*ro3), w_post);
                let x2: HbmTensor<bf16, Chip, m![H]> = x2d.to_hbm(&mut device.tdma);
                let h2t: HbmTensor<bf16, Chip, m![H]> = h2d.to_hbm(&mut device.tdma);
                h2d.view().to_hbm_view(&mut device.tdma, h2s.view_mut().tile::<m![U], 1, m![1 #{!} 2, H]>(0));
                let mut pk: PackDm = DmTensor::new();
                store_h2_f32(device, &h2d, &mut pk);
                let sw = SharedRef { wg: &*rg, wu: &*ru, wd: &*rd, wr: &*rr };
                shared_router_pre(device, &h2t, &x2, sw, ys.view_mut().tile::<m![U], 1, m![1 #{!} 2, H]>(0), &mut pk);
                pk.view().to_hbm_view(&mut device.tdma, outs.view_mut().tile::<m![U], 1, m![1 #{!} 2, Pk, R]>(0));
            }
            if m < 1usize {
                let ExpertT { wg_t, wu_t, wd_t } = et;
                let x = h2s.view().tile::<m![U], 1, m![1 # 2, H]>(0);
                ffn_k8_body!(device, x, wg_t, wu_t, wd_t, ids, expert_out);
            }
            if 0usize < m {
                if m < 2usize {
                    let PfW { wq, wkv, wo, wg, wu, wd, wr } = pw;
                    prefetch_partial!(device, wq, wkv, wo, wg, wu, wd, wr, rq0, rq1, rq2, rq3, rkv, ro0, ro1, ro2, ro3, rg, ru, rd, rr);
                }
            }
        }
    };
}

unified_res_k!(unified_res_c1, direct, []);
unified_res_k!(unified_res_c2, direct, [1]);
unified_res_k!(unified_res_c4, table, [1, 2, 3]);
unified_res_k!(unified_res_c8, table, [1, 2, 3, 4, 5, 6, 7]);
unified_res_k!(unified_res_c16, table, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);


/// Probe (measurement only): `prefetch_layer` with the same two-tile split as the prefetch mode,
/// unconditional, to separate the split cost from the branch cost.
#[device(chip = 1, pe = 8)]
pub fn prefetch_split(device: &mut Device, pw: PfW<'_>, a: AttnResMut<'_>, s: SharedResMut<'_>) {
    let PfW { wq, wkv, wo, wg, wu, wd, wr } = pw;
    let AttnResMut { q0, q1, q2, q3, kv, o0, o1, o2, o3 } = a;
    let SharedResMut { g, u, d, r } = s;
    prefetch_partial!(device, wq, wkv, wo, wg, wu, wd, wr, q0, q1, q2, q3, kv, o0, o1, o2, o3, g, u, d, r);
}
