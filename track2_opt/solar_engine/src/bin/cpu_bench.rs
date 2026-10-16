//! Standalone benchmark of the CPU cold-expert kernel (ck_moe1) on interleaved anonymous memory.
//! numactl --interleave=all ./cpu_bench <threads> <first_core> <experts_per_call>
#[path = "../host.rs"]
#[allow(dead_code)]
mod host;
use host::*;
use std::time::Instant;
extern crate solar_engine;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let nt: usize = a[1].parse().unwrap(); let c0: usize = a[2].parse().unwrap(); let k: usize = a[3].parse().unwrap();
    pin_current(c0, 23);
    set_threads(nt);
    println!("raw local read BW ({nt} threads): {:.1} GB/s", read_bw(256 << 20, 5));
    let ne = 48;
    let src: Vec<u16> = (0..3 * F * H).map(|i| f2bf((i % 97) as f32 * 1e-4)).collect();
    let n = F * H;
    let pool: Vec<Box<[u16]>> = (0..ne).map(|_| place_expert(&src[..n], &src[n..2 * n], &src[2 * n..])).collect();
    let x: Vec<u16> = (0..H).map(|i| f2bf((i % 13) as f32 * 0.01)).collect();
    let w = vec![0.125f32; k];
    let mut ts = vec![];
    for it in 0..60 {
        let bases: Vec<*const u16> = (0..k).map(|j| pool[(it * k + j) % ne].as_ptr()).collect();
        let mut out = vec![0f32; H];
        let t = Instant::now();
        moe_cpu_placed(&bases, &w, &x, &mut out);
        if it >= 10 { ts.push(t.elapsed().as_secs_f64()); }
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ts[ts.len() / 2];
    println!("bf16 threads={nt} cores={c0}-23 experts={k}: {:.3} ms  {:.1} GB/s", med * 1e3, (k * 3 * F * H * 2) as f64 / med / 1e9);
    // int8 path: speed + accuracy vs bf16 on a realistic-ish random expert set
    let mut rng = 12345u64;
    let mut rnd = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; ((rng >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5 };
    let reals: Vec<Vec<u16>> = (0..k).map(|_| (0..3 * F * H).map(|_| f2bf(rnd() * 0.04)).collect()).collect();
    let xr: Vec<u16> = (0..H).map(|_| f2bf(rnd() * 2.0)).collect();
    let pl: Vec<Box<[u16]>> = reals.iter().map(|r| place_expert(&r[..n], &r[n..2 * n], &r[2 * n..])).collect();
    let q8: Vec<Q8> = reals.iter().map(|r| place_expert_q8(&r[..n], &r[n..2 * n], &r[2 * n..])).collect();
    let mut o1 = vec![0f32; H]; let mut o2 = vec![0f32; H];
    moe_cpu_placed(&pl.iter().map(|p| p.as_ptr()).collect::<Vec<_>>(), &w, &xr, &mut o1);
    moe_cpu_q8(&q8.iter().collect::<Vec<_>>(), &w, &xr, &mut o2);
    let num: f32 = o1.iter().zip(&o2).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
    let den: f32 = o1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let qpool: Vec<Q8> = (0..ne).map(|_| place_expert_q8(&src[..n], &src[n..2 * n], &src[2 * n..])).collect();
    let mut ts = vec![];
    for it in 0..60 {
        let ex: Vec<&Q8> = (0..k).map(|j| &qpool[(it * k + j) % ne]).collect();
        let mut out = vec![0f32; H];
        let t = Instant::now();
        moe_cpu_q8(&ex, &w, &x, &mut out);
        if it >= 10 { ts.push(t.elapsed().as_secs_f64()); }
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ts[ts.len() / 2];
    println!("int8 threads={nt} experts={k}: {:.3} ms  (rel err vs bf16 = {:.4})", med * 1e3, num / den);
}
