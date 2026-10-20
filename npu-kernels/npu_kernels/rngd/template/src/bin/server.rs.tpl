//! RNGD expert server for H=${H}, F=${F}, E=${E}, K=${K} (generated). Keeps per-layer expert tables resident in HBM
//! and serves requests over stdin/stdout (little-endian, all u32 unless noted):
//!   op 1 LOAD   layer, path_len, path[u8]       tables from <path>.g / .u / .d (device layouts, bf16)
//!   op 2 RUN    layer, n (<= K), slots[n], x[H] bf16   -> reply rows[n][H] bf16
//!   op 3 UPDATE layer, slot, path_len, path     replace one table slot from <path>.g / .u / .d (one expert each)
//!   op 4 RUN_WSUM layer, n (<= K), slots[n], weights[n] f32, x[H] bf16   -> reply sum_k w_k * row_k, [H] f32
//!   op 0 EXIT
//! Every op except RUN replies one u32 status (0 = ok).
use furiosa_opt_std::prelude::*;
use std::io::{Read, Write};
use ${CRATE}::{Chip, WDT, WGT, ffn_k};
use ${CRATE}::axes::{H as AH, K as AK};

type HA = m![AH];
type KH = m![AK, AH];
const H: usize = ${H};
const SLOT_G: usize = ${F} * ${H};

fn rd_u32(r: &mut impl Read) -> std::io::Result<u32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
fn rd_path(r: &mut impl Read) -> std::io::Result<String> {
    let n = rd_u32(r)? as usize; let mut b = vec![0u8; n]; r.read_exact(&mut b)?; Ok(String::from_utf8_lossy(&b).into_owned())
}
fn bf16s(path: &str) -> std::io::Result<Vec<bf16>> {
    let b = std::fs::read(path)?;
    Ok(b.chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect())
}
fn pack(slots: &[u32]) -> u64 {
    let mut p = (slots.len() as u64) << 56;
    for (k, &s) in slots.iter().enumerate() { p += (s as u64) << (6 * k); }
    p
}

struct Layer { g: HbmTensor<bf16, Chip, WGT>, u: HbmTensor<bf16, Chip, WGT>, d: HbmTensor<bf16, Chip, WDT> }

#[tokio::main(flavor = "current_thread")]
async fn main() -> eyre::Result<()> {
    let mut dev = Device::new(ffn_k.topology())?;
    let mut layers: Vec<Option<Layer>> = Vec::new();
    let mut ids = HbmScalar::<u64>::from_host(0, &mut dev.pdma).await?;
    let mut rows = HostTensor::<bf16, KH>::zero().to_hbm(&mut dev.pdma).await?;
    // persistent buffers: x in HBM (written in place), rows read back into page-locked host memory
    let mut x_hbm = HostTensor::<bf16, HA>::zero().to_hbm(&mut dev.pdma).await?;
    let mut rows_h = HostTensor::<bf16, KH>::zero().pinned()?;
    let prof = std::env::var("RNGD_OPT_PROF").is_ok();
    let mut ph = [0f64; 4];          // upload x, write slots, launch, read back
    let mut nrun = 0u64;
    let stdin = std::io::stdin(); let mut inp = stdin.lock();
    let stdout = std::io::stdout(); let mut out = stdout.lock();
    out.write_all(b"RDY1")?; out.flush()?;
    loop {
        let op = match rd_u32(&mut inp) { Ok(v) => v, Err(_) => break };
        match op {
            0 => break,
            1 => {
                let l = rd_u32(&mut inp)? as usize; let p = rd_path(&mut inp)?;
                let g = HostTensor::<bf16, WGT>::from_vec(bf16s(&format!("{p}.g"))?).to_hbm(&mut dev.pdma).await?;
                let u = HostTensor::<bf16, WGT>::from_vec(bf16s(&format!("{p}.u"))?).to_hbm(&mut dev.pdma).await?;
                let d = HostTensor::<bf16, WDT>::from_vec(bf16s(&format!("{p}.d"))?).to_hbm(&mut dev.pdma).await?;
                if layers.len() <= l { layers.resize_with(l + 1, || None); }
                layers[l] = Some(Layer { g, u, d });
                out.write_all(&0u32.to_le_bytes())?; out.flush()?;
            }
            2 | 4 => {
                let l = rd_u32(&mut inp)? as usize; let n = rd_u32(&mut inp)? as usize;
                let mut slots = vec![0u32; n];
                for s in slots.iter_mut() { *s = rd_u32(&mut inp)?; }
                let mut wts = vec![0f32; if op == 4 { n } else { 0 }];
                for w in wts.iter_mut() { *w = f32::from_bits(rd_u32(&mut inp)?); }
                let mut xb = vec![0u8; 2 * H]; inp.read_exact(&mut xb)?;
                let t0 = std::time::Instant::now();
                let x: Vec<bf16> = xb.chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect();
                HostTensor::<bf16, HA>::from_vec(x).to_hbm(&mut dev.pdma).output(&mut x_hbm).await?;
                let t1 = std::time::Instant::now();
                ids.write(pack(&slots), &mut dev.pdma).await?;
                let t2 = std::time::Instant::now();
                let lay = layers[l].as_ref().expect("layer not loaded");
                launch(ffn_k, (&mut dev, &x_hbm, &lay.g, &lay.u, &lay.d, &ids, &mut rows)).await?;
                let t3 = std::time::Instant::now();
                rows.to_host::<KH>(&mut dev.pdma).output(&mut rows_h).await?;
                let got: Vec<bf16> = rows_h.clone().into_vec();
                if prof {
                    let t4 = std::time::Instant::now();
                    ph[0] += (t1 - t0).as_secs_f64(); ph[1] += (t2 - t1).as_secs_f64();
                    ph[2] += (t3 - t2).as_secs_f64(); ph[3] += (t4 - t3).as_secs_f64(); nrun += 1;
                }
                let mut b = Vec::with_capacity(4 * n * H);
                if op == 4 {
                    let mut acc = vec![0f32; H];
                    for (k, &w) in wts.iter().enumerate() {
                        for (a, v) in acc.iter_mut().zip(&got[k * H..(k + 1) * H]) { *a += w * f32::from(*v); }
                    }
                    for a in &acc { b.extend_from_slice(&a.to_le_bytes()); }
                } else {
                    for v in &got[..n * H] { b.extend_from_slice(&v.to_bits().to_le_bytes()); }
                }
                out.write_all(&b)?; out.flush()?;
            }
            3 => {
                let l = rd_u32(&mut inp)? as usize; let slot = rd_u32(&mut inp)? as usize; let p = rd_path(&mut inp)?;
                // read back, patch one slot, write back (simple and safe; one expert is a few MB)
                let lay = layers[l].as_mut().expect("layer not loaded");
                let mut g: Vec<bf16> = lay.g.to_host::<WGT>(&mut dev.pdma).await?.into_vec();
                let mut u: Vec<bf16> = lay.u.to_host::<WGT>(&mut dev.pdma).await?.into_vec();
                let mut d: Vec<bf16> = lay.d.to_host::<WDT>(&mut dev.pdma).await?.into_vec();
                g[slot * SLOT_G..(slot + 1) * SLOT_G].copy_from_slice(&bf16s(&format!("{p}.g"))?);
                u[slot * SLOT_G..(slot + 1) * SLOT_G].copy_from_slice(&bf16s(&format!("{p}.u"))?);
                d[slot * SLOT_G..(slot + 1) * SLOT_G].copy_from_slice(&bf16s(&format!("{p}.d"))?);
                lay.g = HostTensor::<bf16, WGT>::from_vec(g).to_hbm(&mut dev.pdma).await?;
                lay.u = HostTensor::<bf16, WGT>::from_vec(u).to_hbm(&mut dev.pdma).await?;
                lay.d = HostTensor::<bf16, WDT>::from_vec(d).to_hbm(&mut dev.pdma).await?;
                out.write_all(&0u32.to_le_bytes())?; out.flush()?;
            }
            _ => { out.write_all(&1u32.to_le_bytes())?; out.flush()?; }
        }
    }
    if prof && nrun > 0 {
        let us = |v: f64| v / nrun as f64 * 1e6;
        eprintln!("[rngd-opt server] {nrun} runs, per run (us): upload x {:.0}, write slots {:.0}, launch {:.0}, read back {:.0}",
                  us(ph[0]), us(ph[1]), us(ph[2]), us(ph[3]));
    }
    Ok(())
}
