use furiosa_opt_std::prelude::*;

// H hidden, F shared-expert width, R router experts, K top-k, W vocab.
axes![H = 4096, F = 1280, R = 128, K = 8, W = 196608];

pub type Chip = m![1];
