//! Shared axes for every kernel in the engine (one set of axis types so tensors flow between kernels).
use furiosa_opt_std::prelude::*;

axes![H = 4096, F = 1280, R = 128, K = 8, W = 196608, E = 24, E2 = 32, PK = 4112, Pk = 33];

pub type Chip = m![1];

/// Packed per-layer host<->device buffer for layer_a: `[Pk = 33, R = 128]` f32.
pub type Pack = m![Pk, R];
