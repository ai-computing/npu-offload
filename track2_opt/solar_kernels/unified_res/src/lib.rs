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
/// `layer_a_cK` (streaming reference) + `prefetch_layer` + `layer_a_res_cK` (separate-program references).
pub mod layer;
/// Unified resident chip-A program: `unified_res_cK` = layer step on resident weights | ffn_k8 | prefetch.
pub mod unified;
/// Host helpers (weight permutations, oracles, packed-buffer pack/unpack).
pub mod host;
