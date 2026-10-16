use expert_srv::ffn::ffn_k8u;
use furiosa_opt_std::prelude::*;
use std::time::Instant;
fn p50(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
macro_rules! bench_sz {
    ($d:expr, $ty:ty, $n:expr) => {{
        let h = HostTensor::<bf16, $ty>::from_vec(vec![bf16::from_f32(1.0); $n]);
        let mut t: HbmTensor<bf16, m![1], $ty> = h.to_hbm(&mut $d.pdma).await?;
        let (mut up, mut dn) = (vec![], vec![]);
        for _ in 0..200 {
            let s = Instant::now(); HostTensor::<bf16, $ty>::from_vec(vec![bf16::from_f32(1.0); $n]).to_hbm(&mut $d.pdma).output(&mut t).await?; up.push(s.elapsed().as_secs_f64() * 1e6);
            let s = Instant::now(); let _v: Vec<bf16> = t.to_host::<$ty>(&mut $d.pdma).await?.into_vec(); dn.push(s.elapsed().as_secs_f64() * 1e6);
        }
        println!("{:>7} bytes: upload p50 {:6.1} us   download p50 {:6.1} us", $n * 2, p50(up), p50(dn));
    }};
}
#[tokio::main(flavor = "current_thread")]
async fn main() -> eyre::Result<()> {
    let mut d = Device::new(ffn_k8u.topology())?;
    bench_sz!(d, m![256], 256);
    bench_sz!(d, m![4096], 4096);
    bench_sz!(d, m![8192], 8192);
    bench_sz!(d, m![32768], 32768);
    Ok(())
}
