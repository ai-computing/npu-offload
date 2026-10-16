//! Shared axes for every kernel (identical to solar_engine::axes, plus `Pk` for the packed buffers).
use furiosa_opt_std::prelude::*;

axes![H = 4096, F = 1280, R = 128, K = 8, W = 196608, E = 24, E2 = 32, Pk = 33];

pub type Chip = m![1];
/// Packed per-layer host<->device buffer: `[Pk = 33, R = 128]` f32 (4224 floats, 16.5 KiB).
pub type Pack = m![Pk, R];
