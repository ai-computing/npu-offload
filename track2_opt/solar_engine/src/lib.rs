#![expect(clippy::type_complexity)] // Necessary for mapping expressions.
#![feature(proc_macro_hygiene, stmt_expr_attributes)]
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

pub mod axes;
/// Attention block kernels from solar_kernels/attn (own axes except H/Chip).
pub mod attn;
/// Dense kernels: shared expert + router, combine, lm_head.
pub mod dense;
/// Solar expert FFN kernels (moe_opt ffn.rs, E = 24 experts per resident layer table).
pub mod ffn;
/// Fused per-layer chip-A kernel layer_a_cK (combine + attention block + shared/router).
pub mod layer;
/// Unified chip-A program (layer_a | ffn_k8 by runtime mode) — no per-layer program switch.
pub mod unified;
/// Unified chip-A program with resident DM weights: layer (resident) | ffn_k8 | prefetch modes.
pub mod unified_res;
/// Same expert kernels with a 32-expert resident table (chip B).
pub mod ffn32;
/// Fused chip-B experts + gated sum (solar_kernels/experts_b).
pub mod experts_b;
