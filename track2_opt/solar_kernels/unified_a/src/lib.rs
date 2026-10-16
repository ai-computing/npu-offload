#![expect(clippy::type_complexity)] // Necessary for mapping expressions.
#![feature(proc_macro_hygiene, stmt_expr_attributes)]
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

/// Shared axes (copied from solar_engine::axes).
pub mod axes;
/// Attention block (from solar_engine/src/attn, + resident-weight `qkv_rope_res` / `proj_q_pre`).
pub mod attn;
/// Shared expert + router (from solar_engine/src/dense/shared.rs, weights passed by reference).
pub mod dense;
/// Expert FFN kernels (`ffn_k8`), unchanged from solar_engine/src/ffn.rs.
pub mod ffn;
/// `layer_a_cK` (reference, streaming).
pub mod layer;
/// Unified chip-A program: `unified_a_cK` = layer_a_cK (mode 0) | ffn_k8 (mode 1).
pub mod unified;
pub mod probe;
/// Host helpers (weight permutations, oracles, packed-buffer pack/unpack).
pub mod host;
