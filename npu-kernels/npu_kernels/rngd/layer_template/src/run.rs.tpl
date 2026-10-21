//! Host-side launch helpers shared by `bin/eval` and `bin/engine`: per-layer device buffers, per-step scratch, and
//! the attention-block launch for a context length.
use furiosa_opt_std::prelude::*;

use crate::attn::gemv::{WKV, WO, WQ, WR};
use crate::attn::{D, KvCache, Masks};
${ATTN_IMPORTS}
use crate::axes::{Chip, H, NE};
use crate::host;

pub type VH = m![H];
pub type VD = m![D];
pub type VE = m![NE];

pub fn tb(v: &[u16]) -> Vec<bf16> {
    v.iter().map(|&b| bf16::from_bits(b)).collect()
}
pub fn ub(v: &[bf16]) -> Vec<u16> {
    v.iter().map(|b| b.to_bits()).collect()
}

/// Attention-block weights and KV cache of one layer, resident in HBM.
pub struct LayerDev {
    pub w_in: HbmTensor<bf16, Chip, VH>,
    pub wq: HbmTensor<bf16, Chip, WQ>,
    pub wkv: HbmTensor<bf16, Chip, WKV>,
    pub qn: HbmTensor<bf16, Chip, VD>,
    pub kn: HbmTensor<bf16, Chip, VD>,
    pub wo: HbmTensor<bf16, Chip, WO>,
    pub w_post: HbmTensor<bf16, Chip, VH>,
    pub wr: HbmTensor<bf16, Chip, WR>,
    pub kc: HbmTensor<bf16, Chip, KvCache>,
    pub vc: HbmTensor<bf16, Chip, KvCache>,
}

impl LayerDev {
    /// upload from HF-layout weights (bf16 bits); `kc` / `vc` start as zeros
    pub async fn upload(dev: &mut Device, w: &host::BlockW<'_>) -> eyre::Result<Self> {
        let d = |v: &[u16]| HostTensor::<bf16, VD>::from_vec(tb(v));
        let qn = if host::QK_NORM { w.qn.to_vec() } else { vec![0u16; D_] };
        let kn = if host::QK_NORM { w.kn.to_vec() } else { vec![0u16; D_] };
        Ok(Self {
            w_in: HostTensor::<bf16, VH>::from_vec(tb(w.w_in)).to_hbm(&mut dev.pdma).await?,
            wq: HostTensor::<bf16, WQ>::from_vec(tb(&host::perm_wq(w.wq))).to_hbm(&mut dev.pdma).await?,
            wkv: HostTensor::<bf16, WKV>::from_vec(tb(&host::perm_wkv(w.wk, w.wv))).to_hbm(&mut dev.pdma).await?,
            qn: d(&qn).to_hbm(&mut dev.pdma).await?,
            kn: d(&kn).to_hbm(&mut dev.pdma).await?,
            wo: HostTensor::<bf16, WO>::from_vec(tb(&host::perm_wo(w.wo))).to_hbm(&mut dev.pdma).await?,
            w_post: HostTensor::<bf16, VH>::from_vec(tb(w.w_post)).to_hbm(&mut dev.pdma).await?,
            wr: HostTensor::<bf16, WR>::from_vec(tb(&host::perm_wr(w.wr))).to_hbm(&mut dev.pdma).await?,
            kc: HostTensor::<bf16, KvCache>::zero().to_hbm(&mut dev.pdma).await?,
            vc: HostTensor::<bf16, KvCache>::zero().to_hbm(&mut dev.pdma).await?,
        })
    }
}

const D_: usize = host::D;

/// Per-step inputs / outputs of the attention block (shared by all layers).
pub struct Scratch {
    pub x: HbmTensor<bf16, Chip, VH>,
    pub off: HbmTensor<i32, Chip, m![1]>,
    pub cos: HbmTensor<bf16, Chip, VD>,
    pub sin: HbmTensor<bf16, Chip, VD>,
    pub masks: HbmTensor<f32, Chip, Masks>,
    pub x2: HbmTensor<bf16, Chip, VH>,
    pub h2: HbmTensor<bf16, Chip, VH>,
    pub lg: HbmTensor<bf16, Chip, VE>,
}

impl Scratch {
    pub async fn new(dev: &mut Device) -> eyre::Result<Self> {
        Ok(Self {
            x: HostTensor::<bf16, VH>::zero().to_hbm(&mut dev.pdma).await?,
            off: HostTensor::<i32, m![1]>::from_vec(vec![0]).to_hbm(&mut dev.pdma).await?,
            cos: HostTensor::<bf16, VD>::zero().to_hbm(&mut dev.pdma).await?,
            sin: HostTensor::<bf16, VD>::zero().to_hbm(&mut dev.pdma).await?,
            masks: HostTensor::<f32, Masks>::from_vec(host::masks(0)).to_hbm(&mut dev.pdma).await?,
            x2: HostTensor::<bf16, VH>::zero().to_hbm(&mut dev.pdma).await?,
            h2: HostTensor::<bf16, VH>::zero().to_hbm(&mut dev.pdma).await?,
            lg: HostTensor::<bf16, VE>::zero().to_hbm(&mut dev.pdma).await?,
        })
    }

    /// position-dependent inputs for token `pos` (context pos + 1): kv offset, RoPE row, masks
    pub async fn set_pos(&mut self, dev: &mut Device, pos: usize) -> eyre::Result<()> {
        let (c, s) = host::cos_sin(pos);
        HostTensor::<i32, m![1]>::from_vec(vec![host::kv_offset(pos)]).to_hbm(&mut dev.pdma).output(&mut self.off).await?;
        HostTensor::<bf16, VD>::from_vec(tb(&c)).to_hbm(&mut dev.pdma).output(&mut self.cos).await?;
        HostTensor::<bf16, VD>::from_vec(tb(&host::rope_sin_signed(&s))).to_hbm(&mut dev.pdma).output(&mut self.sin).await?;
        HostTensor::<f32, Masks>::from_vec(host::masks(pos + 1)).to_hbm(&mut dev.pdma).output(&mut self.masks).await?;
        Ok(())
    }
}

/// one attention-block launch (x -> x2, h2, logits) for a context of `ctx` tokens
pub async fn attn_launch(dev: &mut Device, ctx: usize, l: &mut LayerDev, s: &mut Scratch) -> eyre::Result<()> {
    let c = host::chunks_for(ctx).ok_or_else(|| eyre::eyre!("context {ctx} exceeds the compiled attention launches"))?;
    match c {
${BLOCK_ARMS}
        _ => unreachable!(),
    }
    Ok(())
}
