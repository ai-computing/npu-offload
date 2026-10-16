//! Fused per-layer chip-A kernel: combine (previous layer's MoE output) -> attention block ->
//! shared expert + router, one launch per layer.
//!
//!   x      = bf16( (y_in + extra) + sum_k gates[k] * expert_out[k] )        (= dense::combine)
//!   x2, h2 = attention block on x                                           (= attn::attn_block_cK)
//!   y_out  = bf16( x2 + Wsd @ (silu(Wsg h2) * (Wsu h2)) ), logits = h2 @ Wr^T (= dense::shared_router)
//!
//! Packed host buffers (both `[Pk = 33, R = 128]` f32, see `host::pack`):
//!   * `inp` (host -> device): `inp[0..4096]` = extra (f32), `inp[4096..4104]` = gates, rest ignored.
//!   * `out` (device -> host): `out[0..4096]` = h2 (bf16 values widened to f32, exact),
//!     `out[4096..4224]` = router logits (f32).
use furiosa_opt_std::prelude::*;

use crate::attn::gemv::{RC, RElem, RS, WKV, WO, WQ, WoDm, WrDm, load_wo};
use crate::attn::{D, G, N, KvCache, Masks, attention, kv_chunk, o_proj_epilogue_pre, qkv_rope_pre, qkv_rope_res};
use crate::attn::{Hc, Hr, Q, Qc, RW};
use crate::axes::{Chip, H, K, Pk, R};
use crate::dense::shared::{SdDm, SgDm, SharedRef, SrDm, WD, WG, WR, load_shared, shared_router_pre};

// combine placement: 2 clusters x 16 slices, 128 hidden elements per slice
type CC = m![H / 2048];
type CS = m![H / 128 % 16, 1 # 16];

/// x = bf16((y + extra) + sum_k g_k e_k), from the packed input; same op sequence as `dense::combine`.
pub(crate) fn combine_packed<Yv: M>(
    device: &mut Device,
    y: HbmTensorView<'_, bf16, Chip, Yv>,
    expert_out: &HbmTensor<bf16, Chip, m![K, H]>,
    inp: &HbmTensor<f32, Chip, m![Pk, R]>,
) -> DmTensor<bf16, Chip, CC, CS, m![H % 128]> {
    // gates: row 32 of the pack, broadcast to every slice, then relabel R = (R / 8, K) and keep R / 8 = 0.
    let gr: DmTensor<f32, Chip, CC, CS, m![R]> =
        inp.view().tile::<m![Pk], 1, m![1 # 33, R]>(32).to_dm(&mut device.tdma);
    // SAFETY: relabel R as (R / 8) x K; same cells, same order.
    let gr = unsafe { gr.reshape::<Chip, CC, CS, m![R / 8, K]>() };
    let g_vrf: VrfTensor<f32, Chip, CC, CS, m![K]> = device
        .sub
        .begin(gr.view().tile::<m![R / 8], 1, m![1 # 16, K]>(0))
        .fetch::<m![1], m![K]>()
        .collect::<m![1], m![K]>()
        .to_vrf();

    // extra: rows 0..32 of the pack (= H), loaded in the combine placement.
    let ed: DmTensor<f32, Chip, m![Pk = 32 / 16], m![Pk = 32 % 16, 1 # 16], m![R]> =
        inp.view().tile::<m![Pk], 32, m![Pk = 32 # 33, R]>(0).to_dm(&mut device.tdma);
    // SAFETY: relabel (Pk = 32) x R as H = (H / 128) x (H % 128); same cells, same order.
    let ed = unsafe { ed.reshape::<Chip, CC, CS, m![H % 128]>() };
    let ex_vrf: VrfTensor<f32, Chip, CC, CS, m![H % 128]> = device
        .sub
        .begin(ed.view())
        .fetch::<m![H / 8 % 16], m![H % 8]>()
        .collect::<m![H / 8 % 16], m![H % 8]>()
        .to_vrf();
    let yd: DmTensor<bf16, Chip, CC, CS, m![H % 128]> = y.to_dm(&mut device.tdma);
    let s: DmTensor<f32, Chip, CC, CS, m![H % 128]> = device
        .sub
        .begin(yd.view())
        .fetch::<m![H / 16 % 8], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 16], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 32], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &ex_vrf)
        .vector_widen_concat::<m![H / 8 % 16], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();
    let s_vrf: VrfTensor<f32, Chip, CC, CS, m![H % 128]> = device
        .sub
        .begin(s.view())
        .fetch::<m![H / 8 % 16], m![H % 8]>()
        .collect::<m![H / 8 % 16], m![H % 8]>()
        .to_vrf();

    let rows: DmTensor<bf16, Chip, CC, CS, m![K, H % 128]> = expert_out.to_dm(&mut device.tdma);
    device
        .main
        .begin(rows.view())
        .fetch::<m![H / 8 % 16, K], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 16, K], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 8 % 16, K, H / 4 % 2], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &g_vrf)
        .vector_intra_slice_reduce::<K, m![H / 8 % 16, H / 4 % 2], m![H % 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_concat::<m![H / 8 % 16], m![H % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &s_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// DM staging of the packed output (one slice), written to HBM with one full-tensor DMA.
/// (Partial tile writes into the packed HBM output are dropped by the compiler in the full kernel.)
pub(crate) type PackDm = DmTensor<f32, Chip, m![1 # 2], m![1 # 256], m![Pk, R]>;

/// h2 (bf16, epilogue placement) widened to f32 into rows 0..32 of the DM pack (one DM->DM DMA).
pub(crate) fn store_h2_f32(
    device: &mut Device,
    h2: &DmTensor<bf16, Chip, crate::attn::epilogue::EC, crate::attn::gemv::OSO, crate::attn::epilogue::EH>,
    pk: &mut PackDm,
) {
    let hf: DmTensor<f32, Chip, crate::attn::epilogue::EC, crate::attn::gemv::OSO, crate::attn::epilogue::EH> = device
        .main
        .begin(h2.view())
        .fetch::<m![H / 1024, H % 512 / 8], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 1024, H % 512 / 8], m![H % 8]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    // SAFETY: relabel H = (Pk < 32) x R: H/512%2 = Pk/4%2, H/1024 = Pk/8, H%512 = (Pk%4, R); same cells.
    let hp = unsafe {
        hf.reshape::<Chip, m![Pk = 32 / 4 % 2], crate::attn::gemv::OSO, m![Pk = 32 / 8, Pk = 32 % 4, R]>()
    };
    hp.view().to_dm_view(&mut device.tdma, pk.view_mut().tile::<m![Pk], 32, m![Pk = 32 #{!} 33, R]>(0));
}

/// Mask loading: `direct` = one HBM row per chunk (K <= 2), `table` = whole table once (K >= 4).
macro_rules! mask_tbl {
    (direct, $dev:expr, $masks:expr) => {
        ()
    };
    (table, $dev:expr, $masks:expr) => {
        attention::mask_table($dev, $masks)
    };
}
macro_rules! mask_get {
    (direct, $dev:expr, $masks:expr, $tbl:expr, $c:expr) => {
        attention::mask_vrf_hbm($dev, $masks, $c)
    };
    (table, $dev:expr, $masks:expr, $tbl:expr, $c:expr) => {
        attention::mask_vrf($dev, &$tbl, $c)
    };
}

/// Body of `layer_a_cK` (shared with the unified chip-A kernel).
pub(crate) use mask_tbl;
pub(crate) use mask_get;

macro_rules! layer_a_body {
    ($mm:ident, [$($c:literal),*], $device:ident, $y_in:expr, $expert_out:expr, $inp:expr, $w_in:expr, $wq:expr,
     $wkv:expr, $kv_offset:expr, $cos:expr, $sin:expr, $k_cache:ident, $v_cache:ident, $masks:expr, $wo:expr,
     $w_post:expr, $wg:expr, $wu:expr, $wd:expr, $wr:expr, $out:ident, $x_out:ident, $y_out:ident, $h2_out:ident) => {{
            // the first big weight stream is independent of the combine: issue it first
            let wkv_dm: DmTensor<bf16, Chip, RC, RS, RElem> = $wkv.to_dm(&mut $device.tdma);
            let xd = combine_packed($device, $y_in.view(), $expert_out, $inp);
            xd.view().to_hbm_view(&mut $device.tdma, $x_out.view_mut());
            let x: &HbmTensor<bf16, Chip, m![H]> = $x_out;

            let q = qkv_rope_pre($device, x, $w_in, $wq, &wkv_dm, $kv_offset, $cos, $sin, $k_cache, $v_cache);
            // prefetch Wo and the shared-expert weights: they stream while attention (latency bound) runs
            let wo0 = load_wo($device, $wo, 0);
            let wo1 = load_wo($device, $wo, 1);
            let wo2 = load_wo($device, $wo, 2);
            let wo3 = load_wo($device, $wo, 3);
            let sw = load_shared($device, $wg, $wu, $wd, $wr);
            // q broadcast to the token-slices via HBM (HBM->DM broadcast is cheaper than DM->DM)
            let q_hbm: HbmTensor<bf16, Chip, m![N, G, D]> = q.to_hbm(&mut $device.tdma);
            let q = attention::q_trf_from_hbm($device, &q_hbm);
            #[allow(clippy::let_unit_value)]
            let tbl = mask_tbl!($mm, $device, $masks);
            let mask = mask_get!($mm, $device, $masks, tbl, 0);
            #[allow(unused_mut)]
            let (mut max, mut sum, mut o) =
                attention::first_dm($device, &q, kv_chunk($k_cache, 0), kv_chunk($v_cache, 0), &mask);
            $(
                let mask = mask_get!($mm, $device, $masks, tbl, $c);
                let (m2, s2, o2) = attention::next_dm(
                    $device, &q, kv_chunk($k_cache, $c), kv_chunk($v_cache, $c), &mask, &max, &sum, &o,
                );
                max = m2;
                sum = s2;
                o = o2;
            )*
            let _ = &max;
            let a = attention::norm_dm($device, &o, &sum);
            let (x2d, h2d) = o_proj_epilogue_pre($device, a, x, (&wo0, &wo1, &wo2, &wo3), $w_post);
            let x2: HbmTensor<bf16, Chip, m![H]> = x2d.to_hbm(&mut $device.tdma);
            h2d.view().to_hbm_view(&mut $device.tdma, $h2_out.view_mut());
            let h2: &HbmTensor<bf16, Chip, m![H]> = $h2_out;
            let mut pk: PackDm = DmTensor::new();
            store_h2_f32($device, &h2d, &mut pk);
            shared_router_pre($device, h2, &x2, sw.as_ref(), $y_out.view_mut(), &mut pk);
            pk.view().to_hbm_view(&mut $device.tdma, $out.view_mut());
    }};
}
pub(crate) use layer_a_body;

macro_rules! layer_a_k {
    ($name:ident, $mm:ident, [$($c:literal),*]) => {
        /// Fused chip-A layer for contexts `<= 512 *` K tokens (K in the name). See module docs.
        ///
        /// * `y_in`: previous layer's `y_out` (bf16 [H]); pass a zero buffer for layer 0 (and put the
        ///   embedding row in `inp` as `extra`, gates = 0).
        /// * `expert_out`: chip-A routed expert outputs of the previous layer ([K, H] bf16, `ffn_k8`).
        /// * `inp`: packed extra + gates. Attention / shared args exactly as in `attn_block_cK` and
        ///   `shared_router`.
        /// * `x_out`: x (combine result, = xcur), `y_out`: next layer's `y_in` (must not alias `y_in`),
        ///   `out`: packed h2 + logits.
        #[device(chip = 1, pe = 8)]
        #[expect(clippy::too_many_arguments)]
        pub fn $name(
            device: &mut Device,
            y_in: &HbmTensor<bf16, Chip, m![H]>,
            expert_out: &HbmTensor<bf16, Chip, m![K, H]>,
            inp: &HbmTensor<f32, Chip, m![Pk, R]>,
            w_in: &HbmTensor<bf16, Chip, m![H]>,
            wq: &HbmTensor<bf16, Chip, WQ>,
            wkv: &HbmTensor<bf16, Chip, WKV>,
            kv_offset: &HbmTensor<i32, Chip, m![1]>,
            cos: &HbmTensor<bf16, Chip, m![D]>,
            sin: &HbmTensor<bf16, Chip, m![D]>,
            k_cache: &mut HbmTensor<bf16, Chip, KvCache>,
            v_cache: &mut HbmTensor<bf16, Chip, KvCache>,
            masks: &HbmTensor<f32, Chip, Masks>,
            wo: &HbmTensor<bf16, Chip, WO>,
            w_post: &HbmTensor<bf16, Chip, m![H]>,
            wg: &HbmTensor<bf16, Chip, WG>,
            wu: &HbmTensor<bf16, Chip, WG>,
            wd: &HbmTensor<bf16, Chip, WD>,
            wr: &HbmTensor<bf16, Chip, WR>,
            out: &mut HbmTensor<f32, Chip, m![Pk, R]>,
            x_out: &mut HbmTensor<bf16, Chip, m![H]>,
            y_out: &mut HbmTensor<bf16, Chip, m![H]>,
            h2_out: &mut HbmTensor<bf16, Chip, m![H]>,
        ) {
            layer_a_body!($mm, [$($c),*], device, y_in, expert_out, inp, w_in, wq, wkv, kv_offset, cos, sin, k_cache,
                v_cache, masks, wo, w_post, wg, wu, wd, wr, out, x_out, y_out, h2_out);
        }
    };
}

layer_a_k!(layer_a_c1, direct, []);
layer_a_k!(layer_a_c2, direct, [1]);
layer_a_k!(layer_a_c4, table, [1, 2, 3]);
layer_a_k!(layer_a_c8, table, [1, 2, 3, 4, 5, 6, 7]);
layer_a_k!(layer_a_c16, table, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);


