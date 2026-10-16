//! solar_engine: Solar-Open-100B decode engine (Rust). Stage 1 = all-host reference path.
//!
//! usage: solar_engine <data_dir> <n_new> [n_layers]
//!   data_dir: prompt.json (ids, model_dir), rope_cos.bin / rope_sin.bin (bf16 [8192,128])
mod chip;
mod host;
use host::*;
use std::time::Instant;

struct Model {
    w: Weights,
    cos: Vec<u16>,
    sin: Vec<u16>,
    n_layers: usize,
    kc: Vec<Vec<u16>>, // per layer [S, KVD] bf16
    vc: Vec<Vec<u16>>,
    pos: usize,
    prof: std::collections::BTreeMap<&'static str, f64>,
    chips: Vec<chip::Chip>,
    slot_of: Vec<Vec<Option<(usize, usize)>>>, // [layer][expert] -> (chip, local slot)
    hits: usize,
    misses: usize,
    bias: Vec<Vec<f32>>,
    tail_skip: usize,
    spos: [usize; 2],
    no_ea: bool,
    lm_on_b: bool,
    lm_split: bool,
    balance: bool,
    bal_cpu: f64,
    bal_bfix: f64,
    bal_bper: f64,
    moved: usize,
    b_expert: Vec<Vec<usize>>, // [layer][chip-B slot] -> expert id
    prefetch_n: usize,
    pf_hit: usize,
    pf_issued: usize,
    res: bool,
    skipped: usize,
    cold_store: Vec<Vec<Option<Box<[u16]>>>>, // [layer][expert] -> g|u|d copy in process memory
    cold_q8: Vec<Vec<Option<Q8>>>,            // COLD_INT8=1: int8 copies of cold experts (approximate)
}

const MAXS: usize = 4096;

impl Model {
    fn tick(&mut self, k: &'static str, t: Instant) -> Instant {
        *self.prof.entry(k).or_insert(0.0) += t.elapsed().as_secs_f64();
        Instant::now()
    }

    /// cold expert weights: from the interleaved process-memory copy when available, else the mmap.
    fn cold(&self, l: usize, e: usize) -> (&'static [u16], &'static [u16], &'static [u16]) {
        match self.cold_store.get(l).and_then(|v| v[e].as_ref()) {
            Some(b) => {
                // cold_store is immutable after init and lives as long as the Model
                let b: &'static [u16] = unsafe { std::slice::from_raw_parts(b.as_ptr(), b.len()) };
                let n = F * H;
                (&b[..n], &b[n..2 * n], &b[2 * n..])
            }
            None => self.w.expert(l, e),
        }
    }

    fn rope(&self, v: &mut [f32], heads: usize) {
        let c = &self.cos[self.pos * D..(self.pos + 1) * D];
        let s = &self.sin[self.pos * D..(self.pos + 1) * D];
        for h in 0..heads {
            let x = &mut v[h * D..(h + 1) * D];
            let old: Vec<f32> = x.to_vec();
            for i in 0..D {
                let rot = if i < D / 2 { -old[i + D / 2] } else { old[i - D / 2] };
                // HF: q*cos + rotate_half(q)*sin in bf16
                x[i] = bf2f(f2bf(bf2f(f2bf(old[i] * bf2f(c[i]))) + bf2f(f2bf(rot * bf2f(s[i])))));
            }
        }
    }

    fn attention(&self, l: usize, q: &[f32]) -> Vec<u16> {
        let s = self.pos + 1;
        let kc = &self.kc[l];
        let vc = &self.vc[l];
        let scale = 1.0 / (D as f32).sqrt();
        let mut out = vec![0u16; QD];
        for h in 0..NQ {
            let kvh = h / (NQ / NKV);
            let qh = &q[h * D..(h + 1) * D];
            let mut sc: Vec<f32> = (0..s).map(|t| {
                let k = &kc[t * KVD + kvh * D..t * KVD + kvh * D + D];
                let dot: f32 = qh.iter().zip(k).map(|(a, &b)| a * bf2f(b)).sum();
                bf2f(f2bf(bf2f(f2bf(dot)) * scale))
            }).collect();
            let m = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut tot = 0.0;
            for v in sc.iter_mut() { *v = (*v - m).exp(); tot += *v; }
            for v in sc.iter_mut() { *v = bf2f(f2bf(*v / tot)); }
            for i in 0..D {
                let mut acc = 0.0f32;
                for t in 0..s { acc += sc[t] * bf2f(vc[t * KVD + kvh * D + i]); }
                out[h * D + i] = f2bf(acc);
            }
        }
        out
    }

    /// one decoder layer, all host. x: residual stream (bf16 values in f32).
    fn layer(&mut self, l: usize, x: &mut Vec<f32>) -> eyre::Result<()> {
        let mut t = Instant::now();
        let h = rmsnorm(x, self.w.layer(l, "input_layernorm.weight"));
        let mut q = gemv(self.w.layer(l, "self_attn.q_proj.weight"), &h, QD, H); round_bf(&mut q);
        let mut k = gemv(self.w.layer(l, "self_attn.k_proj.weight"), &h, KVD, H); round_bf(&mut k);
        let mut v = gemv(self.w.layer(l, "self_attn.v_proj.weight"), &h, KVD, H); round_bf(&mut v);
        t = self.tick("qkv", t);
        self.rope(&mut q, NQ);
        self.rope(&mut k, NKV);
        let p = self.pos;
        self.kc[l][p * KVD..(p + 1) * KVD].copy_from_slice(&to_bf(&k));
        self.vc[l][p * KVD..(p + 1) * KVD].copy_from_slice(&to_bf(&v));
        let a = self.attention(l, &q);
        t = self.tick("attn", t);
        let mut o = gemv(self.w.layer(l, "self_attn.o_proj.weight"), &a, H, QD); round_bf(&mut o);
        for i in 0..H { x[i] = bf2f(f2bf(x[i] + o[i])); }
        let h2 = rmsnorm(x, self.w.layer(l, "post_attention_layernorm.weight"));
        t = self.tick("o+norm", t);
        self.moe_host(l, x, &h2, t)
    }

    fn moe_host(&mut self, l: usize, x: &mut Vec<f32>, h2: &[u16], mut t: Instant) -> eyre::Result<()> {
        let logits = gemv(self.w.layer(l, "mlp.gate.weight"), h2, E, H);
        let (ids, gw) = route(&logits, &self.bias[l]);
        let sh = (self.w.layer(l, "mlp.shared_experts.gate_proj.weight"), self.w.layer(l, "mlp.shared_experts.up_proj.weight"),
                  self.w.layer(l, "mlp.shared_experts.down_proj.weight"));
        let mut shared = vec![0f32; H];
        moe_cpu(&[sh], &[1.0], h2, &mut shared);
        round_bf(&mut shared);
        let ex: Vec<_> = ids.iter().map(|&e| self.w.expert(l, e)).collect();
        let mut moe = vec![0f32; H];
        moe_cpu(&ex, &gw, h2, &mut moe);
        round_bf(&mut moe);
        t = self.tick("moe_host", t);
        for i in 0..H { x[i] = bf2f(f2bf(x[i] + bf2f(f2bf(moe[i] + shared[i])))); }
        self.tick("combine", t);
        Ok(())
    }

    /// fused path: one layer_a launch per layer on chip A (combine prev + attention + shared/router)
    /// MoE part of one layer for stream `s`: route, dispatch hot experts to chips A/B, cold experts on the CPU,
    /// gather chip B; returns the next layer's packed chip-A input (extra + chip-A gates).
    fn moe_phase(&mut self, l: usize, s: usize, logits: &[f32], h2: Vec<u16>, predicted: &[usize], t: &mut Instant) -> eyre::Result<Vec<f32>> {
        let (ids, gw) = route(&logits, &self.bias[l]);
        let mut sl: Vec<(Vec<usize>, [f32; 8])> = vec![(vec![], [0.0; 8]); 2];
        let mut cold = vec![];
        let mut cold_w = vec![];
        let mut cold_ids = vec![];
        for (k, &e) in ids.iter().enumerate() {
            let slot = match self.slot_of[l][e] { Some((0, _)) if self.no_ea => None, x => x };
            match slot {
                Some((c, s)) => { let n = sl[c].0.len(); sl[c].0.push(s); sl[c].1[n] = gw[k]; }
                // TAIL_SKIP=N (approximate, PolyStrata-style): drop cold experts ranked >= N, no renormalization
                None if k >= self.tail_skip => { self.skipped += 1; }
                None => { cold.push(self.cold(l, e)); cold_w.push(gw[k]); cold_ids.push(e); }
            }
        }
        // BALANCE=1: when the CPU path is shorter than chip B's, move chip-B experts to the CPU (exact: bf16 math)
        if self.balance && self.chips.len() > 1 {
            let cpu_ms = |n: usize| if n == 0 { 0.0 } else { 0.02 + self.bal_cpu * n as f64 };
            let b_ms = |n: usize| if n == 0 { 0.0 } else { self.bal_bfix + self.bal_bper * n as f64 };
            while !sl[1].0.is_empty() && cpu_ms(cold.len() + 1) < b_ms(sl[1].0.len()) {
                // move the last-queued chip-B expert (its slot k) to the CPU
                let k = sl[1].0.len() - 1;
                let slot = sl[1].0.pop().unwrap();
                let g = sl[1].1[k]; sl[1].1[k] = 0.0;
                let e = self.b_expert[l][slot];
                cold.push(self.cold(l, e)); cold_w.push(g); cold_ids.push(e);
                self.moved += 1;
            }
        }
        self.hits += ids.len() - cold.len(); self.misses += cold.len();
        self.pf_hit += cold_ids.iter().filter(|e| predicted.contains(e)).count();
        self.pf_issued += predicted.len();
        let pb = if self.chips.len() > 1 && !sl[1].0.is_empty() {
            let (bt, br) = chip::rx();
            self.chips[1].send(chip::Cmd::ExpertsB { layer: l, h2: h2.clone(), slots: sl[1].0.clone(), gates: sl[1].1, reply: bt, t0: Instant::now() });
            Some(br)
        } else { None };
        if !sl[0].0.is_empty() { self.chips[0].send(chip::Cmd::ExpertsA2 { layer: l, h2: h2.clone(), slots: sl[0].0.clone(), s }); }
        // next layer's weights -> resident DM while the host computes the cold experts
        if self.res && l + 1 < self.n_layers { self.chips[0].send(chip::Cmd::Prefetch { layer: l + 1 }); }
        *t = self.tick("route+dispatch", *t);
        let mut extra = vec![0f32; H];
        if !self.cold_q8.is_empty() {
            let qs: Vec<&Q8> = cold_ids.iter().map(|&e| self.cold_q8[l][e].as_ref().unwrap()).collect();
            moe_cpu_q8(&qs, &cold_w, &h2, &mut extra);
        } else {
            let bases: Vec<*const u16> = cold.iter().map(|c| c.0.as_ptr()).collect();
            moe_cpu_placed(&bases, &cold_w, &h2, &mut extra);
        }
        *t = self.tick("cold_cpu", *t);
        if let Some(br) = pb {
            let xb = chip::spin_recv(&br).ok_or_else(|| eyre::eyre!("chip gone"))?.map_err(|e| e)?;
            for i in 0..H { extra[i] += bf2f(xb[i]); }
        }
        *t = self.tick("chipB_wait", *t);
        // next layer's packed input: extra (rounded to bf16 like the unfused path) + chip-A gates
        let mut inp = vec![0f32; 33 * 128];
        for i in 0..H { inp[i] = bf2f(f2bf(extra[i])); }
        inp[H..H + 8].copy_from_slice(&sl[0].1);
        Ok(inp)
    }

    /// Two decode streams interleaved layer by layer: while the CPU computes stream s's cold experts, chip A runs
    /// the other stream's attention. Each stream's math is identical to step_fused (exact).
    fn step_pair(&mut self, toks: [usize; 2]) -> eyre::Result<[usize; 2]> {
        let mut t = Instant::now();
        let mut inps: [Vec<f32>; 2] = [vec![], vec![]];
        for s in 0..2 {
            let p = self.spos[s];
            self.chips[0].send(chip::Cmd::Token { emb: vec![0u16; H], cos: self.cos[p * D..(p + 1) * D].to_vec(),
                                                  sin: self.sin[p * D..(p + 1) * D].to_vec(), pos: p, s });
            let emb = &self.w.bf("model.embed_tokens.weight")[toks[s] * H..(toks[s] + 1) * H];
            let mut inp = vec![0f32; 33 * 128];
            for i in 0..H { inp[i] = bf2f(emb[i]); }
            inps[s] = inp;
        }
        let mut rxs: Vec<Option<std::sync::mpsc::Receiver<eyre::Result<(Vec<f32>, Vec<u16>)>>>> = vec![None, None];
        for s in 0..2 {
            let (lt, lr) = chip::rx();
            self.chips[0].send(chip::Cmd::LayerA2 { layer: 0, inp: std::mem::take(&mut inps[s]), reply: lt, s });
            rxs[s] = Some(lr);
        }
        for l in 0..self.n_layers {
            for s in 0..2 {
                let lr = rxs[s].take().unwrap();
                let (logits, h2) = chip::spin_recv(&lr).ok_or_else(|| eyre::eyre!("chip gone"))?.map_err(|e| e)?;
                t = self.tick("layerA", t);
                let inp = self.moe_phase(l, s, &logits, h2, &[], &mut t)?;
                if l + 1 < self.n_layers {
                    let (lt, lr) = chip::rx();
                    self.chips[0].send(chip::Cmd::LayerA2 { layer: l + 1, inp, reply: lt, s });
                    rxs[s] = Some(lr);
                } else {
                    inps[s] = inp;
                }
            }
        }
        let mut out = [0usize; 2];
        for s in 0..2 {
            let (lt, lr) = chip::rx();
            self.chips[0].send(chip::Cmd::FinalLm { inp: std::mem::take(&mut inps[s]), reply: lt, s });
            out[s] = chip::spin_recv(&lr).ok_or_else(|| eyre::eyre!("chip gone"))?.map_err(|e| e)?;
            self.spos[s] += 1;
        }
        self.tick("lm_head", t);
        Ok(out)
    }

    fn step_fused(&mut self, tok: usize) -> eyre::Result<usize> {
        let mut t = Instant::now();
        let p = self.pos;
        let emb: Vec<f32> = self.w.bf("model.embed_tokens.weight")[tok * H..(tok + 1) * H].iter().map(|&b| bf2f(b)).collect();
        self.chips[0].send(chip::Cmd::Token { emb: vec![0u16; H], cos: self.cos[p * D..(p + 1) * D].to_vec(),
                                              sin: self.sin[p * D..(p + 1) * D].to_vec(), pos: p, s: 0 });
        // layer 0 input: extra = embedding, gates = 0 (y_in = 0 on device)
        let mut inp = vec![0f32; 33 * 128];
        inp[..H].copy_from_slice(&emb);
        let mut prev_h2: Option<Vec<u16>> = None;
        for l in 0..self.n_layers {
            let (lt, lr) = chip::rx();
            self.chips[0].send(chip::Cmd::LayerA2 { layer: l, inp: std::mem::take(&mut inp), reply: lt, s: 0 });
            // while chip A runs this layer's attention (CPU idle): pre-gate layer l with the previous layer's h2
            // and pull the predicted cold experts into the CPU caches (exact: data movement only)
            let mut predicted: Vec<usize> = vec![];
            if self.prefetch_n > 0 {
                if let Some(ph) = prev_h2.as_ref() {
                    let pl = gemv(self.w.layer(l, "mlp.gate.weight"), ph, E, H);
                    let (pids, _) = route(&pl, &self.bias[l]);
                    predicted = pids.into_iter().enumerate()
                        .filter(|&(k, e)| self.slot_of[l][e].is_none() && k < self.tail_skip)
                        .map(|(_, e)| e).take(self.prefetch_n).collect();
                    let bases: Vec<*const u16> = predicted.iter().map(|&e| self.cold(l, e).0.as_ptr()).collect();
                    touch_experts(&bases);
                }
            }
            t = self.tick("prefetch_cpu", t);
            let (logits, h2) = chip::spin_recv(&lr).ok_or_else(|| eyre::eyre!("chip gone"))?.map_err(|e| e)?;
            t = self.tick("layerA", t);
            inp = self.moe_phase(l, 0, &logits, h2.clone(), &predicted, &mut t)?;
            prev_h2 = Some(h2);
        }
        let best = if self.lm_split {
            let (xt, xr) = chip::rx();
            let (at, ar) = chip::rx();
            self.chips[0].send(chip::Cmd::FinalSplit { inp, xreply: xt, reply: at, s: 0 });
            let x = chip::spin_recv(&xr).ok_or_else(|| eyre::eyre!("chip gone"))?.map_err(|e| e)?;
            let (bt, br) = chip::rx();
            self.chips[1].send(chip::Cmd::LmHalfX { x, reply: bt });
            let (va, ra) = chip::spin_recv(&ar).ok_or_else(|| eyre::eyre!("chip gone"))?.map_err(|e| e)?;
            let (vb, rb) = chip::spin_recv(&br).ok_or_else(|| eyre::eyre!("chip gone"))?.map_err(|e| e)?;
            if vb > va { rb } else { ra }
        } else if self.lm_on_b {
            let (xt, xr) = chip::rx();
            self.chips[0].send(chip::Cmd::FinalX { inp, reply: xt, s: 0 });
            let x = chip::spin_recv(&xr).ok_or_else(|| eyre::eyre!("chip gone"))?.map_err(|e| e)?;
            let (lt, lr) = chip::rx();
            self.chips[1].send(chip::Cmd::LmHeadX { x, reply: lt });
            chip::spin_recv(&lr).ok_or_else(|| eyre::eyre!("chip gone"))?.map_err(|e| e)?
        } else {
            let (lt, lr) = chip::rx();
            self.chips[0].send(chip::Cmd::FinalLm { inp, reply: lt, s: 0 });
            chip::spin_recv(&lr).ok_or_else(|| eyre::eyre!("chip gone"))?.map_err(|e| e)?
        };
        self.tick("lm_head", t);
        self.pos += 1;
        Ok(best)
    }

    /// full-NPU path: attention+shared/router+combine+lm_head on chip A, experts on A/B, cold experts on CPU
    fn step_chips(&mut self, tok: usize) -> eyre::Result<usize> {
        let mut t = Instant::now();
        let emb = self.w.bf("model.embed_tokens.weight")[tok * H..(tok + 1) * H].to_vec();
        let p = self.pos;
        self.chips[0].send(chip::Cmd::Token { emb, cos: self.cos[p * D..(p + 1) * D].to_vec(),
                                              sin: self.sin[p * D..(p + 1) * D].to_vec(), pos: p, s: 0 });
        for l in 0..self.n_layers {
            let (lt, lr) = chip::rx();
            self.chips[0].send(chip::Cmd::LayerA { layer: l, reply: lt });
            let (logits, h2) = chip::spin_recv(&lr).ok_or_else(|| eyre::eyre!("chip gone"))?.map_err(|e| e)?;
            t = self.tick("layerA", t);
            let (ids, gw) = route(&logits, &self.bias[l]);
            let mut sl: Vec<(Vec<usize>, [f32; 8])> = vec![(vec![], [0.0; 8]); 2];
            let mut cold = vec![];
            let mut cold_w = vec![];
            for (k, &e) in ids.iter().enumerate() {
                match self.slot_of[l][e] {
                    Some((c, s)) => { let n = sl[c].0.len(); sl[c].0.push(s); sl[c].1[n] = gw[k]; }
                    None => { cold.push(self.cold(l, e)); cold_w.push(gw[k]); }
                }
            }
            self.hits += ids.len() - cold.len(); self.misses += cold.len();
            let pb = if self.chips.len() > 1 && !sl[1].0.is_empty() {
                let (bt, br) = chip::rx();
                self.chips[1].send(chip::Cmd::ExpertsB { layer: l, h2: h2.clone(), slots: sl[1].0.clone(), gates: sl[1].1, reply: bt, t0: Instant::now() });
                Some(br)
            } else { None };
            if !sl[0].0.is_empty() { self.chips[0].send(chip::Cmd::ExpertsA { layer: l, slots: sl[0].0.clone() }); }
            t = self.tick("route+dispatch", t);
            let mut extra = vec![0f32; H];
            if !self.cold_q8.is_empty() {
                let qs: Vec<&Q8> = ids.iter().filter(|&&e| self.slot_of[l][e].is_none())
                    .map(|&e| self.cold_q8[l][e].as_ref().unwrap()).collect();
                moe_cpu_q8(&qs, &cold_w, &h2, &mut extra);
            } else {
                let bases: Vec<*const u16> = cold.iter().map(|c| c.0.as_ptr()).collect();
                moe_cpu_placed(&bases, &cold_w, &h2, &mut extra);
            }
            t = self.tick("cold_cpu", t);
            if let Some(br) = pb {
                let xb = chip::spin_recv(&br).ok_or_else(|| eyre::eyre!("chip gone"))?.map_err(|e| e)?;
                for i in 0..H { extra[i] += bf2f(xb[i]); }
            }
            t = self.tick("chipB_wait", t);
            self.chips[0].send(chip::Cmd::CombineA { gates: sl[0].1, extra: to_bf(&extra) });
        }
        let (lt, lr) = chip::rx();
        self.chips[0].send(chip::Cmd::LmHead { reply: lt });
        let best = chip::spin_recv(&lr).ok_or_else(|| eyre::eyre!("chip gone"))?.map_err(|e| e)?;
        self.tick("lm_head", t);
        self.pos += 1;
        Ok(best)
    }

    fn step(&mut self, tok: usize) -> eyre::Result<usize> {
        if !self.chips.is_empty() {
            return if std::env::var("UNFUSED").is_ok() { self.step_chips(tok) } else { self.step_fused(tok) };
        }
        let emb = self.w.bf("model.embed_tokens.weight");
        let mut x: Vec<f32> = emb[tok * H..(tok + 1) * H].iter().map(|&b| bf2f(b)).collect();
        for l in 0..self.n_layers { self.layer(l, &mut x)?; }
        let t = Instant::now();
        self.pos += 1;
        let hn = rmsnorm(&x, self.w.bf("model.norm.weight"));
        let lg = gemv(self.w.bf("lm_head.weight"), &hn, VOCAB, H);
        self.tick("lm_head", t);
        let mut best = 0;
        for i in 1..VOCAB { if lg[i] > lg[best] { best = i; } }
        Ok(best)
    }
}

/// Serving loop. stdin: one request per line `{"ids":[...], "max_new":N, "stop":[...], "stop_count":k}`; stdout: `READY` once, then per
/// request `{"tok":t}` lines as tokens are produced and `{"done":true,...}` (or `{"error":...}`). Greedy, exact bf16.
fn serve(m: &mut Model) -> eyre::Result<()> {
    use std::io::{BufRead, Write};
    const CTX: usize = 8192;
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    writeln!(out, "READY")?; out.flush()?;
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let req: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v, Err(e) => { writeln!(out, "{}", serde_json::json!({"error": format!("bad request: {e}")}))?; out.flush()?; continue; }
        };
        let ids: Vec<usize> = req["ids"].as_array().map(|a| a.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect()).unwrap_or_default();
        let max_new = req["max_new"].as_u64().unwrap_or(16) as usize;
        let stop: Vec<usize> = req["stop"].as_array().map(|a| a.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect()).unwrap_or_default();
        // stop on the stop_count-th stop token (Solar: <|end|> closes the think block first, then the answer)
        let stop_count = req["stop_count"].as_u64().unwrap_or(1).max(1) as usize;
        if ids.is_empty() || ids.len() + max_new > CTX || ids.iter().any(|&t| t >= VOCAB) {
            writeln!(out, "{}", serde_json::json!({"error": format!("need 1..{CTX} tokens incl. max_new and ids < {VOCAB}")}))?; out.flush()?; continue;
        }
        m.pos = 0; m.hits = 0; m.misses = 0; m.prof.clear();
        let t0 = Instant::now();
        let mut tok = 0;
        for &id in &ids { tok = m.step(id)?; }
        let prefill = t0.elapsed().as_secs_f64();
        writeln!(out, "{}", serde_json::json!({"tok": tok}))?; out.flush()?;
        let t1 = Instant::now();
        let mut n = 1;
        let mut seen = stop.contains(&tok) as usize;
        while n < max_new && seen < stop_count {
            tok = m.step(tok)?; n += 1;
            seen += stop.contains(&tok) as usize;
            writeln!(out, "{}", serde_json::json!({"tok": tok}))?; out.flush()?;
        }
        let dec = t1.elapsed().as_secs_f64();
        writeln!(out, "{}", serde_json::json!({"done": true, "n": n, "prompt": ids.len(), "prefill_s": prefill,
            "decode_ms_per_tok": if n > 1 { dec * 1e3 / (n - 1) as f64 } else { 0.0 },
            "hit_rate": m.hits as f64 / (m.hits + m.misses).max(1) as f64}))?;
        out.flush()?;
    }
    Ok(())
}

fn main() -> eyre::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let data = &args[1];
    // `solar_engine <data_dir> serve`: persistent server, one JSON request per stdin line (see serve())
    let serve_mode = args[2] == "serve";
    let n_new: usize = if serve_mode { 0 } else { args[2].parse()? };
    let n_layers: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(L);
    if let Ok(n) = std::env::var("MMIO_RT").map(|v| v.parse::<i32>().unwrap_or(8)) {
        unsafe { host::ck_mmio_set_threads(n) };
        let _ = furiosa_opt_rt::MMIO_READ_HOOK.set(chip::mmio::hook);
    }
    let cpu_threads: usize = std::env::var("CPU_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(18);
    set_threads(cpu_threads);
    let pj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{data}/prompt.json"))?)?;
    let ids: Vec<usize> = pj["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let t0 = Instant::now();
    let w = Weights::open(pj["model_dir"].as_str().unwrap())?;
    eprintln!("[engine] weights mapped in {:.1}s", t0.elapsed().as_secs_f64());
    // placement: [layer][chip] -> expert ids in slot order
    let use_chips: usize = std::env::var("CHIPS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let pl: Vec<Vec<Vec<usize>>> = serde_json::from_str(&std::fs::read_to_string(format!("{data}/placement.json"))?)?;
    let mut slot_of = vec![vec![None; E]; n_layers];
    let mut chips = vec![];
    if use_chips > 0 {
        let (rtx, rrx) = std::sync::mpsc::channel();
        for c in 0..use_chips {
            let tables: Vec<Vec<chip::Ex>> = (0..n_layers).map(|l| pl[l][c].iter().map(|&e| w.expert(l, e)).collect()).collect();
            for l in 0..n_layers { for (s, &e) in pl[l][c].iter().enumerate() { slot_of[l][e] = Some((c, s)); } }
            let dense = if c == 0 {
                Some(chip::DenseW {
                    layers: (0..n_layers).map(|l| chip::DenseLayer {
                        sg: w.layer(l, "mlp.shared_experts.gate_proj.weight"), su: w.layer(l, "mlp.shared_experts.up_proj.weight"),
                        sd: w.layer(l, "mlp.shared_experts.down_proj.weight"), wr: w.layer(l, "mlp.gate.weight"),
                        w_in: w.layer(l, "input_layernorm.weight"), wq: w.layer(l, "self_attn.q_proj.weight"),
                        wk: w.layer(l, "self_attn.k_proj.weight"), wv: w.layer(l, "self_attn.v_proj.weight"),
                        wo: w.layer(l, "self_attn.o_proj.weight"), w_post: w.layer(l, "post_attention_layernorm.weight") }).collect(),
                    lm: w.bf("lm_head.weight"), w_final: w.bf("model.norm.weight"),
                })
            } else if std::env::var("LM_ON_B").map(|v| v == "1").unwrap_or(false) || std::env::var("LM_SPLIT").map(|v| v == "1").unwrap_or(false) {
                Some(chip::DenseW { layers: vec![], lm: w.bf("lm_head.weight"), w_final: w.bf("model.norm.weight") })
            } else { None };
            chips.push(chip::Chip::spawn(c, tables, dense, rtx.clone()));
        }
        for _ in 0..use_chips { eyre::ensure!(rrx.recv()? != usize::MAX, "chip init failed"); }
        // main thread + its OpenMP team (cold experts) on the cores the chip threads don't use
        let cc: usize = std::env::var("CHIP_CORES").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
        if cc > 0 { pin_current(cc, 23); }
        eprintln!("[engine] {use_chips} chips ready ({:.0}s)", t0.elapsed().as_secs_f64());
    }
    let mut m = Model {
        w, cos: read_bf16_file(&format!("{data}/rope_cos.bin"))?, sin: read_bf16_file(&format!("{data}/rope_sin.bin"))?,
        n_layers, kc: vec![vec![0u16; MAXS * KVD]; n_layers], vc: vec![vec![0u16; MAXS * KVD]; n_layers], pos: 0,
        prof: Default::default(), chips, slot_of, hits: 0, misses: 0, bias: vec![], cold_store: vec![], cold_q8: vec![],
        tail_skip: std::env::var("TAIL_SKIP").ok().and_then(|s| s.parse().ok()).unwrap_or(TOPK), skipped: 0, spos: [0, 0], no_ea: std::env::var("NO_EA").is_ok(), lm_on_b: std::env::var("LM_ON_B").map(|v| v == "1").unwrap_or(false),
        lm_split: std::env::var("LM_SPLIT").map(|v| v == "1").unwrap_or(false),
        balance: std::env::var("BALANCE").map(|v| v == "1").unwrap_or(false),
        bal_cpu: std::env::var("BAL_CPU").ok().and_then(|v| v.parse().ok()).unwrap_or(0.205),
        bal_bfix: std::env::var("BAL_BFIX").ok().and_then(|v| v.parse().ok()).unwrap_or(0.20),
        bal_bper: std::env::var("BAL_BPER").ok().and_then(|v| v.parse().ok()).unwrap_or(0.03),
        moved: 0, b_expert: pl.iter().take(n_layers).map(|x| x.get(1).cloned().unwrap_or_default()).collect(),
        prefetch_n: std::env::var("PREFETCH_N").ok().and_then(|s| s.parse().ok()).unwrap_or(0), pf_hit: 0, pf_issued: 0, res: std::env::var("RES").map(|v| v == "1" || v == "attn").unwrap_or(false) || std::env::var("URES").map(|v| v == "1").unwrap_or(false),
    };
    if std::env::var("COLD_COPY").map(|v| v != "0").unwrap_or(true) && !m.chips.is_empty() {
        // copy cold (non-resident) experts into fresh anonymous memory: under `numactl --interleave=all` it is
        // spread over all NUMA nodes (the shared page cache of the safetensors files is not).
        let t1 = Instant::now();
        let w = &m.w; let so = &m.slot_of;
        // NUMA-local placement: each page first-touched by the OpenMP thread that reads it in ck_moe3
        let balance = std::env::var("BALANCE").map(|v| v == "1").unwrap_or(false);
        let store: Vec<Vec<Option<Box<[u16]>>>> = (0..n_layers).map(|l| (0..E).map(|e| if matches!(so[l][e], Some((0, _))) || (so[l][e].is_some() && !balance) { None } else {
            let (g, u, d) = w.expert(l, e);
            Some(place_expert(g, u, d))
        }).collect()).collect();
        m.cold_store = store;
        if std::env::var("COLD_INT8").map(|v| v == "1").unwrap_or(false) {
            let t2 = Instant::now();
            m.cold_q8 = (0..n_layers).map(|l| (0..E).map(|e| if so[l][e].is_some() { None } else {
                let (g, u, d) = w.expert(l, e);
                Some(place_expert_q8(g, u, d))
            }).collect()).collect();
            eprintln!("[engine] COLD_INT8: int8 cold experts placed in {:.0}s (approximate mode)", t2.elapsed().as_secs_f64());
        }
        eprintln!("[engine] cold experts copied to process memory in {:.0}s", t1.elapsed().as_secs_f64());
    }
    m.bias = (0..n_layers).map(|l| m.w.f32v(&format!("model.layers.{l}.mlp.gate.e_score_correction_bias"))).collect();
    if std::env::var("MMIO_RT").is_ok() { chip::mmio::start_serving(); }
    if serve_mode { return serve(&mut m); }
    if std::env::var("NSTREAM").map(|v| v == "2").unwrap_or(false) {
        // two concurrent streams (same prompt): aggregate throughput, exact per stream
        let repeat: usize = std::env::var("REPEAT").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
        let mut rounds = vec![];
        let mut outs: [Vec<usize>; 2] = [vec![], vec![]];
        for r in 0..repeat {
            m.spos = [0, 0];
            let t0 = Instant::now();
            let mut tk = [0usize; 2];
            for &id in &ids { tk = m.step_pair([id, id])?; }
            let tp = t0.elapsed().as_secs_f64();
            m.prof.clear(); m.hits = 0; m.misses = 0; m.skipped = 0;
            outs = [vec![tk[0]], vec![tk[1]]];
            let t0 = Instant::now();
            for _ in 1..n_new { tk = m.step_pair(tk)?; outs[0].push(tk[0]); outs[1].push(tk[1]); }
            let td = t0.elapsed().as_secs_f64();
            let steps = (n_new - 1).max(1) as f64;
            rounds.push(td * 1e3 / steps);
            eprintln!("[round {r}] prefill {:.2}s; pair step {:.1} ms -> per-stream {:.2} tok/s, aggregate {:.2} tok/s",
                      tp, td * 1e3 / steps, steps / td, 2.0 * steps / td);
        }
        let mut sr = rounds.clone(); sr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = sr[sr.len() / 2];
        println!("TOKENS0 {}", serde_json::to_string(&outs[0])?);
        println!("TOKENS1 {}", serde_json::to_string(&outs[1])?);
        println!("NSTREAM=2: pair step {:.1} ms (median of {}: {:?}) -> per-stream {:.2} tok/s, AGGREGATE {:.2} tok/s",
                 med, rounds.len(), rounds.iter().map(|v| (v * 10.0).round() / 10.0).collect::<Vec<_>>(), 1e3 / med, 2e3 / med);
        let nd = (n_new - 1).max(1) as f64;
        let prof: Vec<String> = m.prof.iter().map(|(k, v)| format!("{k}={:.1}ms", v * 1e3 / nd)).collect();
        println!("per-pair-step: {}", prof.join(" "));
        println!("hit_rate {:.3}", m.hits as f64 / (m.hits + m.misses).max(1) as f64);
        return Ok(());
    }
    let repeat: usize = std::env::var("REPEAT").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut rounds = vec![];
    let mut out = vec![];
    let mut tp = 0.0;
    let mut nd = 1.0;
    for r in 0..repeat {
        m.pos = 0;
        let t0 = Instant::now();
        let mut tok = 0;
        for &id in &ids { tok = m.step(id)?; }
        tp = t0.elapsed().as_secs_f64();
        m.prof.clear(); m.hits = 0; m.misses = 0; m.skipped = 0; m.pf_hit = 0; m.pf_issued = 0; m.moved = 0;
        for ch in m.chips.iter() { let (t, rr) = chip::rx(); ch.send(chip::Cmd::Stats { reply: t }); let _ = rr.recv(); }
        out = vec![tok];
        // FORCE=<json file with a token list>: teacher forcing — feed the given tokens, record argmax predictions
        let forced: Option<Vec<usize>> = std::env::var("FORCE").ok().map(|f| serde_json::from_str(&std::fs::read_to_string(f).unwrap()).unwrap());
        let t0 = Instant::now();
        for i in 1..n_new {
            let inp = match &forced { Some(fv) if i - 1 < fv.len() => fv[i - 1], _ => tok };
            tok = m.step(inp)?; out.push(tok);
        }
        let td = t0.elapsed().as_secs_f64();
        nd = (n_new - 1).max(1) as f64;
        rounds.push(td * 1e3 / nd);
        eprintln!("[round {r}] decode {:.1} ms/tok", td * 1e3 / nd);
    }
    let mut sr = rounds.clone(); sr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sr[sr.len() / 2];
    println!("TOKENS {}", serde_json::to_string(&out)?);
    println!("prefill {} tok in {:.2}s; decode {:.1} ms/tok ({:.2} tok/s) [median of {} rounds: {:?}]", ids.len(), tp, med, 1e3 / med,
             rounds.len(), rounds.iter().map(|v| (v * 10.0).round() / 10.0).collect::<Vec<_>>());
    let prof: Vec<String> = m.prof.iter().map(|(k, v)| format!("{k}={:.1}ms", v * 1e3 / nd)).collect();
    let _ = &prof;
    println!("per-token: {}", prof.join(" "));
    println!("balance: moved {} chip-B experts to CPU", m.moved);
    println!("hit_rate {:.3} (tail-skipped cold experts: {}); cpu-prefetch: issued {} useful {} of {} cold", m.hits as f64 / (m.hits + m.misses).max(1) as f64, m.skipped, m.pf_issued, m.pf_hit, m.misses);
    for (c, ch) in m.chips.iter().enumerate() {
        let (t, r) = chip::rx(); ch.send(chip::Cmd::Stats { reply: t });
        println!("chip{c}: {}", r.recv()?.unwrap_or_default());
    }
    Ok(())
}
