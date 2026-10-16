#![expect(clippy::type_complexity)] // Necessary for mapping expressions.
#![feature(proc_macro_hygiene, stmt_expr_attributes)]
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

pub mod axes;
/// Copy of solar_engine ffn32 (32-expert resident tables; baseline ffn_k8).
pub mod ffn32;
/// Copy of solar_engine dense::combine (baseline).
pub mod combine;
/// Fused chip-B experts kernel + expert slot swap.
pub mod experts_b;
