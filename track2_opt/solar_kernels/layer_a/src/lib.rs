#![expect(clippy::type_complexity)] // Necessary for mapping expressions.
#![feature(proc_macro_hygiene, stmt_expr_attributes)]
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

/// Shared axes (same as solar_engine::axes, plus the packed-buffer axes).
pub mod axes;
/// Attention block (copied from solar_engine/src/attn; epilogue also exposes DM-returning variants).
pub mod attn;
/// Shared expert + router, combine (copied from solar_engine/src/dense).
pub mod dense;
/// Fused per-layer chip-A kernel `layer_a_cK`.
pub mod layer;
/// Host helpers (weight permutations, oracles, packed-buffer pack/unpack).
pub mod host;
/// NPU timing probes.
pub mod probe;
