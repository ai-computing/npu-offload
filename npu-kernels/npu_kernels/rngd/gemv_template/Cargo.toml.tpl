[package]
name = "${CRATE}"
version = "0.1.0"
edition = "2024"
publish = false

[package.metadata.furiosa-opt]

[lints.rust]
unexpected_cfgs = { level = "warn", check-cfg = ['cfg(backend, values("emulation", "npu", "typecheck"))', 'cfg(furiosa_opt)'] }

[lib]
path = "src/lib.rs"

[[bin]]
name = "eval"
path = "src/bin/eval.rs"

[dependencies]
eyre = "0.6"
furiosa-opt-std = "=0.9.0"
furiosa-opt-rt = "=0.9.0"
rand = "0.9"
tokio = { version = "1.29", features = ["rt", "rt-multi-thread", "sync", "macros", "time"] }

[profile.release]
debug = false
