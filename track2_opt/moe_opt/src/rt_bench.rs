//! Raw furiosa-opt-rt launch benchmark: loads a compiled image (.bin) and measures sync latency
//! and pipelined (many in flight) throughput. Always opens npu1 only.
use furiosa_opt_rt::image::Memory;
use furiosa_opt_rt::{Device, Function, Image};
use std::sync::Arc;
use std::time::Instant;

fn stats(name: &str, mut v: Vec<f64>) {
    let n = v.len();
    let mean = v.iter().sum::<f64>() / n as f64;
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("{name}: n={n} mean={mean:.1}us p10={:.1} p50={:.1} p90={:.1} p99={:.1}",
        v[n / 10], v[n / 2], v[n * 9 / 10], v[n * 99 / 100]);
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let iters: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let bytes = std::fs::read(path)?;
    let image = Image::parse(&bytes).map_err(furiosa_opt_rt::FunctionError::Image)?;
    println!("image {path}: chips={} pes={} inputs={:?} outputs={:?} slots={:?}",
        image.chips(), image.pes(), image.inputs(), image.outputs(),
        image.slots().iter().map(|s| s.size).collect::<Vec<_>>());
    let device = Arc::new(Device::builder((image.chips(), image.pes())).among([1u8]).open()?);
    let function = Function::load(&device, &image).await?;
    let slot = |i: u32| image.slots()[i as usize].size as usize;
    let inputs: Vec<_> = image.inputs().iter().map(|&i| device.alloc(Memory::Dram, slot(i))).collect::<Result<_, _>>()?;
    let outputs: Vec<_> = image.outputs().iter().map(|&i| device.alloc(Memory::Dram, slot(i))).collect::<Result<_, _>>()?;
    // zero-fill inputs
    for b in &inputs {
        let z = vec![0u8; b.size()];
        device.write([(z.as_slice(), b.on_all())]).await?;
    }
    for _ in 0..20 { function.launch(&inputs, &outputs)?.wait().await?; }
    if std::env::var("PROFILE").is_ok() {
        for lvl in [log::Level::Info, log::Level::Debug] {
            match function.profiled(lvl) {
                Ok(p) => {
                    let spans = p.launch(&inputs, &outputs)?.wait().await?;
                    println!("profile {lvl}: {} spans", spans.len());
                    for s in spans.iter().take(200) {
                        println!("  [{}] {:<40} begin={:>9} end={:>9} dur={:>8} cyc", s.cluster, s.name, s.begin, s.end, s.end - s.begin);
                    }
                }
                Err(e) => println!("profile {lvl} refused: {e}"),
            }
        }
    }

    // 1. sync: submit + wait
    let mut lat = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        function.launch(&inputs, &outputs)?.wait().await?;
        lat.push(t.elapsed().as_secs_f64() * 1e6);
    }
    stats("sync launch+wait", lat);

    // 2. pipelined: K in flight
    for depth in [2usize, 4, 8, 16, 32] {
        let t = Instant::now();
        let mut submit = Vec::new();
        let mut q = std::collections::VecDeque::new();
        for _ in 0..iters {
            if q.len() == depth {
                let l: furiosa_opt_rt::Launch = q.pop_front().unwrap();
                l.wait().await?;
            }
            let t0 = Instant::now();
            q.push_back(function.launch(&inputs, &outputs)?);
            submit.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        while let Some(l) = q.pop_front() { l.wait().await?; }
        let total = t.elapsed().as_secs_f64() * 1e6;
        println!("pipelined depth={depth}: {:.2}us per launch (throughput)", total / iters as f64);
        stats(&format!("  submit-only cost depth={depth}"), submit);
    }
    Ok(())
}
