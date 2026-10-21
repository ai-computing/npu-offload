//! RNGD decoder-layer engine generated for ${CRATE} (npu_kernels.rngd.layer) - edit the template, not this file.
//! Geometry and tiling: plan.json next to Cargo.toml.
#![expect(clippy::type_complexity)]
#![feature(proc_macro_hygiene, stmt_expr_attributes)]
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

pub mod axes {
    use furiosa_opt_std::prelude::*;
    // H hidden, NE router rows (experts), expert kernel: F intermediate, E table slots, K experts per launch
    axes![H = ${H}, NE = ${NE}, F = ${F}, E = ${E}, K = ${K}];
    pub type Chip = m![1];
}

/// attention block + router (one launch per layer)
pub mod attn;
/// routed experts from resident tables
pub mod ffn;
/// LM head GEMV
pub mod lm;
/// host side: geometry constants, weight layouts, oracles
pub mod host;
/// device buffers and launches shared by the binaries
pub mod run;
