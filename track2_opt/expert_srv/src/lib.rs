#![expect(clippy::type_complexity)] // Necessary for mapping expressions.
#![feature(proc_macro_hygiene, stmt_expr_attributes)]
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

/// Copy of moe_opt::kernel::ffn with E = 32 (per-layer hot-expert table size).
pub mod ffn;
