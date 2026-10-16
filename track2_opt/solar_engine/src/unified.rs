//! Unified chip-A program: one `#[device]` fn per context-chunk count K that runs EITHER the
//! `layer_a_cK` body (mode = pos + 1) OR the `ffn_k8` body (mode = 0), selected at runtime by an
//! `HbmScalar<u64>`. The host launches the same program twice per decode layer, so the runtime
//! never switches the loaded program between the two per-layer launches.
//!
//! furiosa-opt 0.9 restriction: a write to a kernel parameter inside a runtime `if` must be a
//! *partial* write (a `#{!}` tile); a whole-tensor write is rejected ("visa: writable parameter
//! output is not registered"); a `scatter` into a parameter is rejected the same way. So the
//! k/v row is written with a runtime (chunk, t) tile DMA derived from `pos` (= mode - 1; this
//! replaces `kv_offset`), and every whole-tensor output of `layer_a_cK` gets a slack row:
//!   * `ys`  [U = 2, H] bf16: row 0 = this layer's y (layer_a `y_out`); row 1 unused.
//!   * `h2s` [U, H] bf16:     row 0 = h2 (written by the layer mode, read by the ffn mode as `x`).
//!   * `outs` [U, Pk, R] f32: row 0 = the packed h2|logits output (`out`); row 1 unused.
//!   * `x_out` is dropped: x lives in a kernel-internal HBM temp (the engine never reads it).
//! `y_in` is a `[U, H]` tensor too (row 0 read), so y ping-pongs between two `[U, H]` buffers.
//! `expert_out` [K, H]: the layer mode reads it (combine), the ffn mode writes its rows (ffn_k8 `out`).
use furiosa_opt_std::prelude::*;

#[allow(unused_imports)]
use crate::attn::gemv::{RC, RElem, RS};
use crate::attn::gemv::{WKV, WO, WQ, load_wo};
use crate::attn::{D, G, N, KvCache, Masks, attention, kv_chunk, o_proj_epilogue_pre, qkv_rope_pre_at};
use crate::axes::{Chip, E, F, H, K, Pk, R, U};
use crate::dense::shared::{WD, WG, WR, load_shared, shared_router_pre};
use crate::ffn::{WDT, WGT, expert, ffn_k8_body, load_x};
use crate::layer::{PackDm, combine_packed, mask_get, mask_tbl, store_h2_f32};

/// `[U, H]` bf16 (row 0 live).
pub type UH = m![U, H];
/// `[U, Pk, R]` f32 (row 0 = the `[Pk, R]` pack).
pub type UPack = m![U, Pk, R];

/// Per-layer attention weights (HBM, same layouts as `layer_a_cK`).
#[derive(DeviceSend)]
pub struct AttnW<'a> {
    pub w_in: &'a HbmTensor<bf16, Chip, m![H]>,
    pub wq: &'a HbmTensor<bf16, Chip, WQ>,
    pub wkv: &'a HbmTensor<bf16, Chip, WKV>,
    pub wo: &'a HbmTensor<bf16, Chip, WO>,
    pub w_post: &'a HbmTensor<bf16, Chip, m![H]>,
}
/// Per-layer shared-expert + router weights.
#[derive(DeviceSend)]
pub struct SharedW<'a> {
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

macro_rules! unified_k {
    ($name:ident, $mm:ident, [$($c:literal),*]) => {
        /// mode = pos + 1 (>= 1): `layer_a_cK` for the token at context position `pos`;
        /// mode = 0: `ffn_k8` (see module docs).
        #[device(chip = 1, pe = 8)]
        #[expect(clippy::too_many_arguments)]
        pub fn $name(
            device: &mut Device,
            mode: &HbmScalar<u64>,
            y_in: &HbmTensor<bf16, Chip, UH>,
            expert_out: &mut HbmTensor<bf16, Chip, m![K, H]>,
            inp: &HbmTensor<f32, Chip, m![Pk, R]>,
            cos: &HbmTensor<bf16, Chip, m![D]>,
            sin: &HbmTensor<bf16, Chip, m![D]>,
            k_cache: &mut HbmTensor<bf16, Chip, KvCache>,
            v_cache: &mut HbmTensor<bf16, Chip, KvCache>,
            masks: &HbmTensor<f32, Chip, Masks>,
            aw: AttnW<'_>,
            sw: SharedW<'_>,
            et: ExpertT<'_>,
            ids: &HbmScalar<u64>,
            outs: &mut HbmTensor<f32, Chip, UPack>,
            ys: &mut HbmTensor<bf16, Chip, UH>,
            h2s: &mut HbmTensor<bf16, Chip, UH>,
        ) {
            let m = mode.to_spm(&mut device.tdma) as usize;
            if 0usize < m {
                let pos = m - 1usize;
                let kv_pos = (pos / 512usize, pos % 512usize);
                let AttnW { w_in, wq, wkv, wo, w_post } = aw;
                let SharedW { wg, wu, wd, wr } = sw;
                let eo: &HbmTensor<bf16, Chip, m![K, H]> = expert_out;
                // ---- layer_a_cK body (same op sequence; only the output plumbing differs) ----
                let wkv_dm: DmTensor<bf16, Chip, RC, RS, RElem> = wkv.to_dm(&mut device.tdma);
                let xd = combine_packed(device, y_in.view().tile::<m![U], 1, m![1 # 2, H]>(0), eo, inp);
                let x_t: HbmTensor<bf16, Chip, m![H]> = xd.to_hbm(&mut device.tdma);
                let x: &HbmTensor<bf16, Chip, m![H]> = &x_t;
                let q = qkv_rope_pre_at(device, x, w_in, wq, &wkv_dm, kv_pos, cos, sin, k_cache, v_cache);
                let wo0 = load_wo(device, wo, 0);
                let wo1 = load_wo(device, wo, 1);
                let wo2 = load_wo(device, wo, 2);
                let wo3 = load_wo(device, wo, 3);
                let shw = load_shared(device, wg, wu, wd, wr);
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
                let (x2d, h2d) = o_proj_epilogue_pre(device, a, x, (&wo0, &wo1, &wo2, &wo3), w_post);
                let x2: HbmTensor<bf16, Chip, m![H]> = x2d.to_hbm(&mut device.tdma);
                let h2t: HbmTensor<bf16, Chip, m![H]> = h2d.to_hbm(&mut device.tdma);
                h2d.view().to_hbm_view(&mut device.tdma, h2s.view_mut().tile::<m![U], 1, m![1 #{!} 2, H]>(0));
                let mut pk: PackDm = DmTensor::new();
                store_h2_f32(device, &h2d, &mut pk);
                shared_router_pre(device, &h2t, &x2, shw.as_ref(), ys.view_mut().tile::<m![U], 1, m![1 #{!} 2, H]>(0), &mut pk);
                pk.view().to_hbm_view(&mut device.tdma, outs.view_mut().tile::<m![U], 1, m![1 #{!} 2, Pk, R]>(0));
            }
            if m < 1usize {
                let ExpertT { wg_t, wu_t, wd_t } = et;
                let x = h2s.view().tile::<m![U], 1, m![1 # 2, H]>(0);
                ffn_k8_body!(device, x, wg_t, wu_t, wd_t, ids, expert_out);
            }
        }
    };
}

unified_k!(unified_a_c1, direct, []);
unified_k!(unified_a_c2, direct, [1]);
unified_k!(unified_a_c4, table, [1, 2, 3]);
unified_k!(unified_a_c8, table, [1, 2, 3, 4, 5, 6, 7]);
unified_k!(unified_a_c16, table, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
