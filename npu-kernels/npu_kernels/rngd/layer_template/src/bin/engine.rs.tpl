//! Decode engine for ${CRATE}: one RNGD chip runs the attention block + router (`attn_block_c*`), the routed experts
//! of a resident hot table (`ffn_k`) and the LM head; the host does the embedding, top-k routing, the experts outside
//! the table (CPU, bf16 weights from the safetensors mmap, overlapped with the chip), the expert combine and argmax.
//! Single request, greedy, bf16; the prompt is fed token by token through the same decode path.
//!
//! usage: engine <model_dir> <n_new> <case_dir>...
//!   case_dir/prompt.json: {"ids": [...]}; case_dir/ref.json (optional, HF reference {"tokens": [...]})
//!   per case: a teacher-forced run on ref.json tokens (when present) -> <OUT_DIR>/<case>.forced.json, then REPEAT
//!   free (greedy) runs -> <OUT_DIR>/<case>.free.json (last repeat; each repeat's rate is printed)
//! env:
//!   PLACEMENT=expert_counts.json   per-layer expert counts; the E_T=${E} most used experts of each layer go on chip
//!                                  (default: experts 0..E_T)
//!   OUT_DIR=.                      results: tokens, per-step top-5 logits, timings, expert hit rate
//!   CPU_THREADS=16  REPEAT=1  LAYERS=<n> (debug: first n layers only)
use furiosa_opt_std::prelude::*;
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;
use ${CRATE}::axes::{H as AH, K as AK};
use ${CRATE}::ffn::{WDT, WGT, ffn_k};
use ${CRATE}::host::{self, *};
use ${CRATE}::lm::{LOut, WLM, lm_head};
use ${CRATE}::run::{self, LayerDev, Scratch, VE, VH, tb, ub};

type KH = m![AK, AH];

struct Weights {
    tensors: HashMap<String, &'static [u8]>,
}

impl Weights {
    fn open(dir: &str) -> eyre::Result<Self> {
        let mut tensors = HashMap::new();
        let mut files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
            .collect();
        files.sort();
        for p in files {
            let f = std::fs::File::open(&p)?;
            let mm: &'static Mmap = Box::leak(Box::new(unsafe { Mmap::map(&f)? }));
            let st = safetensors::SafeTensors::deserialize(&mm[..])?;
            for (name, view) in st.tensors() {
                let data: &'static [u8] = unsafe { std::slice::from_raw_parts(view.data().as_ptr(), view.data().len()) };
                tensors.insert(name, data);
            }
        }
        Ok(Self { tensors })
    }
    fn bf(&self, name: &str) -> &'static [u16] {
        let r = self.tensors.get(name).unwrap_or_else(|| panic!("missing tensor {name}"));
        unsafe { std::slice::from_raw_parts(r.as_ptr() as *const u16, r.len() / 2) }
    }
    fn layer(&self, l: usize, n: &str) -> &'static [u16] {
        self.bf(&format!("model.layers.{l}.{n}"))
    }
    fn expert(&self, l: usize, e: usize) -> [&'static [u16]; 3] {
        let p = format!("model.layers.{l}.mlp.experts.{e}.");
        [self.bf(&(p.clone() + "gate_proj.weight")), self.bf(&(p.clone() + "up_proj.weight")), self.bf(&(p + "down_proj.weight"))]
    }
    fn block(&self, l: usize) -> BlockW<'static> {
        let none: &'static [u16] = &[];
        BlockW {
            w_in: self.layer(l, "input_layernorm.weight"),
            wq: self.layer(l, "self_attn.q_proj.weight"),
            wk: self.layer(l, "self_attn.k_proj.weight"),
            wv: self.layer(l, "self_attn.v_proj.weight"),
            qn: if QK_NORM { self.layer(l, "self_attn.q_norm.weight") } else { none },
            kn: if QK_NORM { self.layer(l, "self_attn.k_norm.weight") } else { none },
            wo: self.layer(l, "self_attn.o_proj.weight"),
            w_post: self.layer(l, "post_attention_layernorm.weight"),
            wr: self.layer(l, "mlp.gate.weight"),
        }
    }
}

// ---------------- CPU experts (HF rounding: bf16 after gate/up, act * up, down) ----------------

#[inline(always)]
fn dot_plain(w: &[u16], x: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    for (wc, xc) in w.chunks_exact(16).zip(x.chunks_exact(16)) {
        for i in 0..16 {
            acc[i] += bf2f(wc[i]) * xc[i];
        }
    }
    acc.iter().sum()
}
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(w: &[u16], x: &[f32]) -> f32 {
    dot_plain(w, x)
}
fn dot(w: &[u16], x: &[f32]) -> f32 {
    static AVX2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *AVX2.get_or_init(|| std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
        unsafe { dot_avx2(w, x) }
    } else {
        dot_plain(w, x)
    }
}

/// one expert on the CPU: rows in parallel
fn expert_cpu(w: [&'static [u16]; 3], x: &[f32]) -> Vec<u16> {
    let [g, u, d] = w;
    let a: Vec<f32> = (0..F)
        .into_par_iter()
        .with_min_len(16)
        .map(|f| {
            let gt = r(dot(&g[f * H..(f + 1) * H], x));
            let up = r(dot(&u[f * H..(f + 1) * H], x));
            r(r(gt / (1.0 + (-gt).exp())) * up)
        })
        .collect();
    (0..H).into_par_iter().with_min_len(32).map(|h| f2bf(dot(&d[h * F..(h + 1) * F], &a))).collect()
}

/// expert table of one layer in device layouts: slot s holds expert hot[s]
fn table(w: &Weights, l: usize, hot: &[usize]) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let parts: Vec<(Vec<u16>, Vec<u16>, Vec<u16>)> = hot
        .par_iter()
        .map(|&e| {
            let [g, u, d] = w.expert(l, e);
            (perm_wg(g), perm_wg(u), perm_wd(d))
        })
        .collect();
    let mut tg = Vec::with_capacity(E_T * F * H);
    let mut tu = Vec::with_capacity(E_T * F * H);
    let mut td = Vec::with_capacity(E_T * F * H);
    for (g, u, d) in parts {
        tg.extend(g);
        tu.extend(u);
        td.extend(d);
    }
    (tg, tu, td)
}

struct ExpDev {
    g: HbmTensor<bf16, ${CRATE}::axes::Chip, WGT>,
    u: HbmTensor<bf16, ${CRATE}::axes::Chip, WGT>,
    d: HbmTensor<bf16, ${CRATE}::axes::Chip, WDT>,
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

#[derive(Default)]
struct Prof(BTreeMap<&'static str, f64>);
impl Prof {
    fn tick(&mut self, k: &'static str, t: &mut Instant) {
        *self.0.entry(k).or_insert(0.0) += t.elapsed().as_secs_f64();
        *t = Instant::now();
    }
}

struct Engine {
    w: Weights,
    dev: Device,
    layers: Vec<LayerDev>,
    exp: Vec<ExpDev>,
    slot_of: Vec<Vec<Option<usize>>>,
    s: Scratch,
    ids: HbmScalar<u64>,
    eout: HbmTensor<bf16, ${CRATE}::axes::Chip, KH>,
    lm_w: HbmTensor<bf16, ${CRATE}::axes::Chip, WLM>,
    lm_x: HbmTensor<bf16, ${CRATE}::axes::Chip, VH>,
    lm_out: HbmTensor<bf16, ${CRATE}::axes::Chip, LOut>,
    n_layers: usize,
    hits: usize,
    misses: usize,
    prof: Prof,
    /// CHECK=1: per layer, compare the chip results with the host oracle (same input x); HOSTPATH=1: continue with the
    /// host results (all experts on the CPU), i.e. a host reference run
    check: bool,
    hostpath: bool,
    hk: Vec<Vec<u16>>,
    hv: Vec<Vec<u16>>,
    errs: BTreeMap<&'static str, (f32, usize)>,
}

fn gemv_par(w: &[u16], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows).into_par_iter().with_min_len(8).map(|i| dot(&w[i * cols..(i + 1) * cols], x)).collect()
}

/// max |a - b| / max |b|
fn rel_err(a: &[u16], b: &[u16]) -> f32 {
    let m = b.iter().map(|&v| bf2f(v).abs()).fold(0f32, f32::max).max(1e-20);
    a.iter().zip(b).map(|(&x, &y)| (bf2f(x) - bf2f(y)).abs()).fold(0f32, f32::max) / m
}

impl Engine {
    /// host oracle of the attention block + router of layer `l` for input `x` at `pos` (host KV cache)
    fn host_block(&mut self, l: usize, x: &[u16], pos: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let bw = self.w.block(l);
        let xf: Vec<f32> = x.iter().map(|&v| bf2f(v)).collect();
        let (cos, sin) = cos_sin(pos);
        let h = rmsnorm(&xf, bw.w_in);
        let q: Vec<f32> = gemv_par(bw.wq, &h, Q, H).into_iter().map(r).collect();
        let k: Vec<f32> = gemv_par(bw.wk, &h, P, H).into_iter().map(r).collect();
        let v: Vec<f32> = gemv_par(bw.wv, &h, P, H).into_iter().map(r).collect();
        let (q, k) = if QK_NORM { (head_norm(&q, bw.qn), head_norm(&k, bw.kn)) } else { (q, k) };
        let q = rope(&q, &cos, &sin);
        let k = rope(&k, &cos, &sin);
        for i in 0..P {
            self.hk[l][pos * P + i] = f2bf(k[i]);
            self.hv[l][pos * P + i] = f2bf(v[i]);
        }
        let a = attention_ref(&q, &self.hk[l], &self.hv[l], pos + 1);
        let o: Vec<f32> = gemv_par(bw.wo, &a, H, Q).into_iter().map(r).collect();
        let x2: Vec<f32> = o.iter().zip(&xf).map(|(&o, &x)| r(o + x)).collect();
        let h2 = rmsnorm(&x2, bw.w_post);
        let lg: Vec<u16> = gemv_par(bw.wr, &h2, NE, H).into_iter().map(f2bf).collect();
        (x2.iter().map(|&v| f2bf(v)).collect(), h2.iter().map(|&v| f2bf(v)).collect(), lg)
    }

    fn err(&mut self, k: &'static str, l: usize, e: f32) {
        let ent = self.errs.entry(k).or_insert((0.0, 0));
        if e > ent.0 {
            *ent = (e, l);
        }
    }

    /// one token through all layers; returns the final-norm hidden state (bf16 bits)
    async fn forward(&mut self, tok: usize, pos: usize) -> eyre::Result<Vec<u16>> {
        let mut t = Instant::now();
        let emb = self.w.bf("model.embed_tokens.weight");
        let mut x: Vec<u16> = emb[tok * H..(tok + 1) * H].to_vec();
        self.s.set_pos(&mut self.dev, pos).await?;
        self.prof.tick("set_pos", &mut t);
        for l in 0..self.n_layers {
            HostTensor::<bf16, VH>::from_vec(tb(&x)).to_hbm(&mut self.dev.pdma).output(&mut self.s.x).await?;
            self.prof.tick("upload_x", &mut t);
            run::attn_launch(&mut self.dev, pos + 1, &mut self.layers[l], &mut self.s).await?;
            self.prof.tick("attn", &mut t);
            let lg = ub(&self.s.lg.to_host::<VE>(&mut self.dev.pdma).await?.into_vec());
            let h2 = ub(&self.s.h2.to_host::<VH>(&mut self.dev.pdma).await?.into_vec());
            let x2 = ub(&self.s.x2.to_host::<VH>(&mut self.dev.pdma).await?.into_vec());
            self.prof.tick("download", &mut t);
            let (lg, h2, x2) = if self.check {
                let (hx2, hh2, hlg) = self.host_block(l, &x, pos);
                let (e1, e2, e3) = (rel_err(&x2, &hx2), rel_err(&h2, &hh2), rel_err(&lg, &hlg));
                self.err("x2", l, e1);
                self.err("h2", l, e2);
                self.err("logits", l, e3);
                if std::env::var("CHECK_VERBOSE").is_ok() {
                    eprintln!("[check] pos {pos} layer {l}: rel err x2 {e1:.4} h2 {e2:.4} router {e3:.4}");
                }
                if self.hostpath { (hlg, hh2, hx2) } else { (lg, h2, x2) }
            } else {
                (lg, h2, x2)
            };
            t = Instant::now();
            let lgf: Vec<f32> = lg.iter().map(|&v| bf2f(v)).collect();
            let (sel, wts) = route(&lgf, None);
            let mut chip: Vec<(usize, usize)> = Vec::new(); // (k, slot)
            let mut cold: Vec<usize> = Vec::new(); // k
            for (k, &e) in sel.iter().enumerate() {
                match if self.hostpath { None } else { self.slot_of[l][e] } {
                    Some(sl) => chip.push((k, sl)),
                    None => cold.push(k),
                }
            }
            self.hits += chip.len();
            self.misses += cold.len();
            // cold experts on the CPU, overlapped with the chip
            let (tx, rx) = std::sync::mpsc::channel();
            if !cold.is_empty() {
                let ws: Vec<(usize, [&'static [u16]; 3])> = cold.iter().map(|&k| (k, self.w.expert(l, sel[k]))).collect();
                let hf: Vec<f32> = h2.iter().map(|&v| bf2f(v)).collect();
                rayon::spawn(move || {
                    let outs: Vec<(usize, Vec<u16>)> = ws.into_iter().map(|(k, w3)| (k, expert_cpu(w3, &hf))).collect();
                    let _ = tx.send(outs);
                });
            } else {
                let _ = tx.send(Vec::new());
            }
            self.prof.tick("route", &mut t);
            let mut outs: Vec<Option<Vec<u16>>> = vec![None; sel.len()];
            if !chip.is_empty() {
                let slots: Vec<usize> = chip.iter().map(|c| c.1).collect();
                self.ids.write(pack_slots(&slots), &mut self.dev.pdma).await?;
                let e = &self.exp[l];
                launch(ffn_k, (&mut self.dev, &self.s.h2, &e.g, &e.u, &e.d, &self.ids, &mut self.eout)).await?;
                let rows = ub(&self.eout.to_host::<KH>(&mut self.dev.pdma).await?.into_vec());
                for (j, &(k, _)) in chip.iter().enumerate() {
                    outs[k] = Some(rows[j * H..(j + 1) * H].to_vec());
                }
            }
            self.prof.tick("experts_chip", &mut t);
            for (k, o) in rx.recv()? {
                outs[k] = Some(o);
            }
            self.prof.tick("experts_cpu_wait", &mut t);
            // HF combine: ascending expert id, bf16(out * w) accumulated in bf16
            let mut order: Vec<usize> = (0..sel.len()).collect();
            order.sort_by_key(|&k| sel[k]);
            let mut acc = vec![0f32; H];
            for k in order {
                let o = outs[k].as_ref().unwrap();
                for i in 0..H {
                    acc[i] = r(acc[i] + r(bf2f(o[i]) * wts[k]));
                }
            }
            if self.check && !self.hostpath {
                // all selected experts on the CPU vs the chip / CPU split
                let hf: Vec<f32> = h2.iter().map(|&v| bf2f(v)).collect();
                let mut acc2 = vec![0f32; H];
                let mut order: Vec<usize> = (0..sel.len()).collect();
                order.sort_by_key(|&k| sel[k]);
                for k in order {
                    let o = expert_cpu(self.w.expert(l, sel[k]), &hf);
                    for i in 0..H {
                        acc2[i] = r(acc2[i] + r(bf2f(o[i]) * wts[k]));
                    }
                }
                let a: Vec<u16> = acc.iter().map(|&v| f2bf(v)).collect();
                let b: Vec<u16> = acc2.iter().map(|&v| f2bf(v)).collect();
                self.err("moe", l, rel_err(&a, &b));
            }
            x = (0..H).map(|i| f2bf(bf2f(x2[i]) + acc[i])).collect();
            self.prof.tick("combine", &mut t);
        }
        let xf: Vec<f32> = x.iter().map(|&v| bf2f(v)).collect();
        let hn = rmsnorm(&xf, self.w.bf("model.norm.weight"));
        Ok(hn.iter().map(|&v| f2bf(v)).collect())
    }

    async fn logits(&mut self, hn: &[u16]) -> eyre::Result<Vec<u16>> {
        let mut t = Instant::now();
        HostTensor::<bf16, VH>::from_vec(tb(hn)).to_hbm(&mut self.dev.pdma).output(&mut self.lm_x).await?;
        launch(lm_head, (&mut self.dev, &self.lm_x, &self.lm_w, &mut self.lm_out)).await?;
        let lg = ub(&self.lm_out.to_host::<LOut>(&mut self.dev.pdma).await?.into_vec());
        self.prof.tick("lm_head", &mut t);
        Ok(lg)
    }

    async fn reset(&mut self) -> eyre::Result<()> {
        for c in self.hk.iter_mut().chain(self.hv.iter_mut()) {
            c.iter_mut().for_each(|v| *v = 0);
        }
        self.errs.clear();
        for l in self.layers.iter_mut() {
            l.kc = HostTensor::<bf16, ${CRATE}::attn::KvCache>::zero().to_hbm(&mut self.dev.pdma).await?;
            l.vc = HostTensor::<bf16, ${CRATE}::attn::KvCache>::zero().to_hbm(&mut self.dev.pdma).await?;
        }
        Ok(())
    }
}

fn top5(lg: &[u16]) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = lg[..VOCAB].iter().enumerate().map(|(i, &b)| (i, bf2f(b))).collect();
    // argmax = first index of the maximum (torch.argmax)
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    v.truncate(5);
    v
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> eyre::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    eyre::ensure!(args.len() >= 4, "usage: engine <model_dir> <n_new> <case_dir>...");
    let (model_dir, n_new, cases) = (&args[1], args[2].parse::<usize>()?, &args[3..]);
    rayon::ThreadPoolBuilder::new().num_threads(env("CPU_THREADS", 16)).build_global()?;
    let t0 = Instant::now();
    let w = Weights::open(model_dir)?;
    let n_layers = env("LAYERS", LAYERS).min(LAYERS);
    // hot experts per layer
    let hot: Vec<Vec<usize>> = match std::env::var("PLACEMENT") {
        Ok(p) => {
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p)?)?;
            (0..LAYERS)
                .map(|l| {
                    let c: Vec<u64> = v["counts"][l].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
                    let mut e: Vec<usize> = (0..NE).collect();
                    e.sort_by(|&a, &b| c[b].cmp(&c[a]).then(a.cmp(&b)));
                    e.truncate(E_T);
                    e
                })
                .collect()
        }
        Err(_) => (0..LAYERS).map(|_| (0..E_T).collect()).collect(),
    };
    let slot_of: Vec<Vec<Option<usize>>> = hot
        .iter()
        .map(|h| {
            let mut s = vec![None; NE];
            for (i, &e) in h.iter().enumerate() {
                s[e] = Some(i);
            }
            s
        })
        .collect();

    let mut dev = Device::new(ffn_k.topology())?;
    let mut layers = Vec::new();
    let mut exp = Vec::new();
    for l in 0..n_layers {
        layers.push(LayerDev::upload(&mut dev, &w.block(l)).await?);
        let (tg, tu, td) = table(&w, l, &hot[l]);
        exp.push(ExpDev {
            g: HostTensor::<bf16, WGT>::from_vec(tb(&tg)).to_hbm(&mut dev.pdma).await?,
            u: HostTensor::<bf16, WGT>::from_vec(tb(&tu)).to_hbm(&mut dev.pdma).await?,
            d: HostTensor::<bf16, WDT>::from_vec(tb(&td)).to_hbm(&mut dev.pdma).await?,
        });
    }
    let lm_w = HostTensor::<bf16, WLM>::from_vec(tb(&perm_lm(w.bf("lm_head.weight")))).to_hbm(&mut dev.pdma).await?;
    let lm_x = HostTensor::<bf16, VH>::zero().to_hbm(&mut dev.pdma).await?;
    let lm_out = HostTensor::<bf16, LOut>::zero().to_hbm(&mut dev.pdma).await?;
    let s = Scratch::new(&mut dev).await?;
    let idsd = HbmScalar::<u64>::from_host(0, &mut dev.pdma).await?;
    let eout = HostTensor::<bf16, KH>::zero().to_hbm(&mut dev.pdma).await?;
    // warm the page cache for the CPU experts (outside the table)
    let warm: f64 = (0..n_layers)
        .into_par_iter()
        .map(|l| {
            let mut a = 0f64;
            for e in 0..NE {
                if slot_of[l][e].is_none() {
                    for m in w.expert(l, e) {
                        a += m.iter().step_by(2048).map(|&v| v as f64).sum::<f64>();
                    }
                }
            }
            a
        })
        .sum();
    eprintln!("[engine] loaded {n_layers} layers, E_T={E_T} hot experts/layer, in {:.1} s ({warm:.0})", t0.elapsed().as_secs_f64());
    let mut eng = Engine {
        w,
        dev,
        layers,
        exp,
        slot_of,
        s,
        ids: idsd,
        eout,
        lm_w,
        lm_x,
        lm_out,
        n_layers,
        hits: 0,
        misses: 0,
        prof: Prof::default(),
        check: std::env::var("CHECK").is_ok() || std::env::var("HOSTPATH").is_ok(),
        hostpath: std::env::var("HOSTPATH").is_ok(),
        hk: if std::env::var("CHECK").is_ok() || std::env::var("HOSTPATH").is_ok() { vec![vec![0u16; T * CHUNKS[CHUNKS.len() - 1] * P]; n_layers] } else { vec![] },
        hv: if std::env::var("CHECK").is_ok() || std::env::var("HOSTPATH").is_ok() { vec![vec![0u16; T * CHUNKS[CHUNKS.len() - 1] * P]; n_layers] } else { vec![] },
        errs: BTreeMap::new(),
    };

    let repeat = env("REPEAT", 1);
    let out_dir = std::env::var("OUT_DIR").unwrap_or_else(|_| ".".into());
    let tokens_of = |p: &str, key: &str| -> eyre::Result<Vec<usize>> {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p)?)?;
        Ok(v[key].as_array().ok_or_else(|| eyre::eyre!("{p}: no {key}"))?.iter().map(|t| t.as_u64().unwrap() as usize).collect())
    };
    for case in cases {
        let name = std::path::Path::new(case).file_name().unwrap().to_string_lossy().to_string();
        let ids = tokens_of(&format!("{case}/prompt.json"), "ids")?;
        let refp = format!("{case}/ref.json");
        if std::path::Path::new(&refp).exists() {
            let force = tokens_of(&refp, "tokens")?;
            let r = run_case(&mut eng, &name, &ids, Some(&force), n_new, 0).await?;
            std::fs::write(format!("{out_dir}/{name}.forced.json"), serde_json::to_string(&r)?)?;
        }
        let mut r = serde_json::json!({});
        for rep in 0..repeat {
            r = run_case(&mut eng, &name, &ids, None, n_new, rep).await?;
        }
        std::fs::write(format!("{out_dir}/{name}.free.json"), serde_json::to_string(&r)?)?;
    }
    Ok(())
}

async fn run_case(
    eng: &mut Engine,
    name: &str,
    ids: &[usize],
    force: Option<&[usize]>,
    n_new: usize,
    rep: usize,
) -> eyre::Result<serde_json::Value> {
    eng.reset().await?;
    eng.prof = Prof::default();
    eng.hits = 0;
    eng.misses = 0;
    let tp = Instant::now();
    for (p, &tok) in ids[..ids.len() - 1].iter().enumerate() {
        eng.forward(tok, p).await?;
    }
    let prefill_s = tp.elapsed().as_secs_f64();
    let mut cur = *ids.last().unwrap();
    let mut toks = Vec::new();
    let mut tops = Vec::new();
    let mut step_ms = Vec::new();
    for i in 0..n_new {
        let t = Instant::now();
        let hn = eng.forward(cur, ids.len() - 1 + i).await?;
        let lg = eng.logits(&hn).await?;
        let tp5 = top5(&lg);
        step_ms.push(t.elapsed().as_secs_f64() * 1e3);
        toks.push(tp5[0].0);
        tops.push(tp5.iter().map(|&(i, v)| serde_json::json!([i, v])).collect::<Vec<_>>());
        cur = match force {
            Some(f) if i < f.len() => f[i],
            _ => tp5[0].0,
        };
    }
    // decode rate: steps after the first generated token
    let dec: Vec<f64> = step_ms[1..].to_vec();
    let mut srt = dec.clone();
    srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = dec.iter().sum::<f64>() / dec.len().max(1) as f64;
    let med = if srt.is_empty() { 0.0 } else { srt[srt.len() / 2] };
    let steps = (ids.len() - 1 + n_new) as f64;
    let prof: BTreeMap<&str, f64> = eng.prof.0.iter().map(|(k, v)| (*k, v / steps * 1e3)).collect();
    let hit = eng.hits as f64 / (eng.hits + eng.misses).max(1) as f64;
    let mode = if force.is_some() { "forced" } else { "free" };
    eprintln!(
        "[engine] {name} {mode} rep {rep}: prefill {} tok {:.2} s; decode {} tok mean {:.2} ms/tok ({:.1} tok/s) median {:.2} ms; hit {hit:.3}",
        ids.len() - 1,
        prefill_s,
        n_new,
        mean,
        1e3 / mean,
        med,
    );
    eprintln!("[engine] per step (ms): {prof:?}");
    if eng.check {
        eprintln!("[check] max rel err (value, layer) over the run: {:?}", eng.errs);
    }
    println!("TOKENS {name} {mode} {toks:?}");
    Ok(serde_json::json!({
        "case": name, "tokens": toks, "top": tops, "step_ms": step_ms, "decode_ms_mean": mean, "decode_ms_median": med,
        "tok_s": 1e3 / mean, "prefill_tokens": ids.len() - 1, "prefill_s": prefill_s, "hit_rate": hit,
        "forced": force.is_some(), "e_t": E_T, "layers": eng.n_layers, "prof_ms_per_step": prof, "rep": rep,
    }))
}
