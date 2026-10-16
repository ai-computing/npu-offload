//! Packed per-layer host buffers for `layer::layer_a_c*` (both `axes::Pack` = `[33, 128]` f32 = 4224 floats).
use furiosa_opt_std::prelude::bf16;

pub const H: usize = 4096;
pub const K: usize = 8;
pub const R: usize = 128;
/// Floats in a packed buffer.
pub const PACK: usize = 33 * 128;

/// Input pack: `[0..4096]` = extra (f32; bf16 values are exact), `[4096..4104]` = gates, rest 0.
pub fn pack_in(extra: &[f32], gates: &[f32]) -> Vec<f32> {
    assert_eq!(extra.len(), H);
    assert!(gates.len() <= K);
    let mut v = vec![0f32; PACK];
    v[..H].copy_from_slice(extra);
    v[H..H + gates.len()].copy_from_slice(gates);
    v
}

/// Same, from a bf16 `extra`.
pub fn pack_in_bf16(extra: &[bf16], gates: &[f32]) -> Vec<f32> {
    let e: Vec<f32> = extra.iter().map(|&x| f32::from(x)).collect();
    pack_in(&e, gates)
}

/// Output pack -> (h2 as bf16 (exact), router logits f32[128]).
pub fn unpack_out(v: &[f32]) -> (Vec<bf16>, Vec<f32>) {
    assert_eq!(v.len(), PACK);
    (v[..H].iter().map(|&x| bf16::from_f32(x)).collect(), v[H..H + R].to_vec())
}
