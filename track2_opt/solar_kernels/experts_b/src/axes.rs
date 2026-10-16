//! Shared axes (copy of solar_engine/src/axes.rs + PK for the packed experts_b upload).
use furiosa_opt_std::prelude::*;

axes![H = 4096, F = 1280, R = 128, K = 8, W = 196608, E = 24, E2 = 32, PK = 4112];

pub type Chip = m![1];
