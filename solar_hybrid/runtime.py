"""Solar-Open-100B hybrid runtime (track 1): hot experts resident on RNGD, cold experts on host CPU.

Per layer, T tokens (T padded to a bucket in {1,8,16}):
  NPU  G_qkv  : h=rmsnorm(x); qkv = h @ Wqkv^T                         (launch 1)
  host        : rope, KV append, attention -> a
  NPU  G_post : x2 = x + a @ Wo^T; h2 = rmsnorm(x2); y = x2 + shared(h2)   (launch 2)
  host        : router(h2) fp32 -> top-k ; split hits (resident) / misses
  NPU  G_exp  : sum_k G[:,k] * FFN_k(h2) over K-bucket slots   (launch 3..)  ||  CPU: misses
  host        : x = y + npu_exp + cpu_exp
mode="cpu" runs everything on host (oracle, same math path).

  python3 runtime.py --layers 4 --mode both --prompt "..." --n 4      # verify NPU vs host oracle
"""
import argparse, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import BF, Weights, rmsnorm, swiglu_ffn, route, Rotary, rotate_half, attention

EDF_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "edf")
CPU_KERNEL = os.environ.get("CPU_KERNEL", "1") == "1"
USE_MESH = os.environ.get("USE_MESH", "0") == "1"
STAGE_COPY = os.environ.get("STAGE_COPY", "1") == "1"
MAIN_THREADS = int(os.environ.get("MAIN_THREADS", "4"))
T_PROF = os.environ.get("T_PROF", "0") == "1"
T_BUCKETS = (1, 8)   # T=16 x K=24 gather graph fails to compile (O543 flattening)
K_BUCKETS = (8, 16, 32)


def bucket(n, bs):
    for b in bs:
        if n <= b:
            return b
    return bs[-1]


# ---------------- NPU graphs (shared by all layers; weights are runtime inputs) ----------------
class GQkv(torch.nn.Module):
    def __init__(s, eps): super().__init__(); s.eps = eps
    def forward(s, x, nw, wqkv):
        xf = x.float(); xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + s.eps)
        return (nw * xf.to(x.dtype)) @ wqkv.t()


class GPost(torch.nn.Module):
    def __init__(s, eps, F): super().__init__(); s.eps = eps; s.F = F
    def forward(s, x, a, wo, nw, sgu, sd):
        x2 = x + a @ wo.t()
        xf = x2.float(); xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + s.eps)
        h2 = nw * xf.to(x2.dtype)
        g = h2 @ sgu.t()
        y = x2 + (torch.nn.functional.silu(g[:, :s.F]) * g[:, s.F:]) @ sd.t()
        return y, h2


class GExp(torch.nn.Module):
    def __init__(s, K, F): super().__init__(); s.K = K; s.F = F
    def forward(s, h, G, *ws):
        out = None
        for i in range(s.K):
            g = h @ ws[2 * i].t()
            y = ((torch.nn.functional.silu(g[:, :s.F]) * g[:, s.F:]) @ ws[2 * i + 1].t()) * G[:, i:i + 1]
            out = y if out is None else out + y
        return out


_FUSION_SET = False
_COMPILE_LOCK = threading.Lock()   # torch.export is not thread-safe


class GExpG(torch.nn.Module):
    """Device-side gather from a per-layer resident table (vendor-style W[ids_int32])."""
    def __init__(s, F): super().__init__(); s.F = F
    def forward(s, h, G, ids, Wgu, Wd):
        gu = torch.matmul(h.unsqueeze(0), Wgu[ids].transpose(1, 2))          # [K,T,2F]
        a = torch.nn.functional.silu(gu[..., :s.F]) * gu[..., s.F:]
        y = torch.matmul(a, Wd[ids].transpose(1, 2))                          # [K,T,H]
        return (y * G.t().unsqueeze(-1)).sum(0)


class Npu:
    def __init__(self, cfg, dev, fusion=True):
        global _FUSION_SET
        import furiosa.torch  # noqa
        if fusion and not _FUSION_SET:
            from furiosa.torch import native_device as nd
            nd.set_fusion(8)
            if USE_MESH:      # mesh(2) => 2 logical chips => all graph I/O in the small DRAM_SHARED pool (OOM, slow copy_)
                nd.set_mesh_kind(nd.MeshKind.Mesh(2))
            _FUSION_SET = True
        from furiosa.torch.custom_ops import CompileModule
        self.CM = CompileModule; self.dev = dev; self.cfg = cfg; self.g = {}

    def stage(self, name, t):
        """Copy host tensor into a persistent device buffer (allocated once per name/shape/dtype).
        Avoids per-call small allocations that fragment the device first-fit allocator (DRAM_SHARED OOM)."""
        if not STAGE_COPY:      # measured: copy_ into an existing device tensor costs ~1-3 ms vs ~85 us for .to()
            return t.to(self.dev)
        if not hasattr(self, "_bufs"):
            self._bufs = {}
        k = (name, tuple(t.shape), t.dtype)
        b = self._bufs.get(k)
        if b is None:
            b = self._bufs[k] = t.contiguous().to(self.dev)
        else:
            b.copy_(t)
        return b

    def put(self, t):
        return t.to(self.dev)

    def graph(self, kind, T, K=0):
        key = (kind, T, K)
        if key in self.g:
            return self.g[key]
        with _COMPILE_LOCK:
            return self._compile(kind, T, K)

    def _compile(self, kind, T, K):
        """Compile once per (kind,T,K) shape, persist EDF to disk, load per device (no recompile)."""
        import artifact_cache as AC
        key = (kind, T, K)
        if key in self.g:
            return self.g[key]
        c = self.cfg; H, F = c.H, c.F
        z = lambda *s: torch.zeros(*s, dtype=BF)

        def compile_fn():
            if kind == "qkv":
                return self.CM.from_module(GQkv(c.eps), (z(T, H), z(H), z(c.nh * c.hd + 2 * c.nkv * c.hd, H)))
            if kind == "post":
                return self.CM.from_module(GPost(c.eps, F), (z(T, H), z(T, c.nh * c.hd), z(H, c.nh * c.hd), z(H),
                                                             z(2 * F, H), z(H, F)))
            if kind == "expg":
                k_, S_ = K
                return self.CM.from_module(GExpG(F), (z(T, H), z(T, k_), torch.zeros(k_, dtype=torch.int32),
                                                      z(S_, 2 * F, H), z(S_, H, F)))
            # distinct placeholder tensors per slot: aliased inputs get deduplicated by the tracer
            ws = [t for _ in range(K) for t in (z(2 * F, H), z(H, F))]
            return self.CM.from_module(GExp(K, F), (z(T, H), z(T, K), *ws))

        ck = AC.cache_key(kind=kind, T=T, K=K, H=H, F=F, nh=c.nh, nkv=c.nkv, hd=c.hd, eps=c.eps, fusion=8, mesh=2 if USE_MESH else 0, v=1)
        m, how, dt = AC.get_or_compile(EDF_CACHE, ck, compile_fn)
        if how == "compile":   # fresh compile object: reload from disk so every device gets its own EdfModule
            m = AC.load(AC.path_for(EDF_CACHE, ck))
        if self.dev is not None:
            m.to(self.dev)
        self.g[key] = m
        print(f"[npu:{self.dev}] {how} {key} in {dt:.1f}s", flush=True)
        return self.g[key]


class Runtime:
    def __init__(self, W, n_layers=None, mode="hybrid", resident=None, devs=("rngd:0",), cpu_threads=12,
                 max_ctx=4096, slots=None, srv_chip=None):
        """slots: per-device #experts per layer -> device-gather tables (resident = {(l,e): (dev, local_idx)})."""
        self.W = W; c = self.c = W.cfg; self.mode = mode; self.slots = slots
        self.nL = n_layers or c.L
        self.rot = Rotary(c.raw)
        self.embed = W.get("model.embed_tokens.weight")
        self.norm = W.get("model.norm.weight")
        self.lm = W.get("lm_head.weight")
        self.cpu_threads = cpu_threads
        # main thread runs many tiny host ops: with torch's default (all 24 cores) they are ~100x slower
        # (oversubscription vs the CPU-expert worker). Keep the main pool small.
        torch.set_num_threads(MAIN_THREADS)
        # persistent workers: a fresh Python thread per layer spins up a fresh OpenMP team every time
        # core split (SNC4, 24 cores): main + NPU-driver threads on cores 0-5 (NUMA nodes 0/1 = NPU-local),
        # CPU-expert OpenMP team on cores 6-23. Run the process under `numactl --interleave=all`.
        main_cores, cpu_cores = set(range(0, 6)), set(range(6, 6 + cpu_threads))
        try: os.sched_setaffinity(0, main_cores)
        except Exception: pass
        def _cpu_init():
            try: os.sched_setaffinity(0, cpu_cores)
            except Exception: pass
            torch.set_num_threads(cpu_threads)
            if CPU_KERNEL:
                from cpu_moe_ext import ext
                ext.set_threads(cpu_threads)
        self.cpu_pool = ThreadPoolExecutor(1, initializer=_cpu_init)
        self.dev_pools = [ThreadPoolExecutor(1) for _ in devs] if mode == "hybrid" else []
        self.srv = None; self.srv_chip = srv_chip
        self.srv_di = len(devs)      # device index used for experts owned by the furiosa-opt server
        t0 = time.time()
        self.dense = [W.layer_dense(l) for l in range(self.nL)]
        # host expert table (all experts; misses read from here)
        self.hx = [[W.expert(l, e) for e in range(c.E)] for l in range(self.nL)]
        print(f"[rt] host weights loaded ({self.nL} layers) {time.time()-t0:.0f}s", flush=True)
        self.resident = {k: v for k, v in (resident or {}).items() if k[0] < self.nL}  # {(l,e): dev_idx}
        self.npu = None
        if mode == "hybrid":
            self.npus = [Npu(c, d) for d in devs]
            self.npu = self.npus[0]       # dense part lives on device 0
            # compile/load + execute every graph once BEFORE the big uploads: the DRAM_SHARED pool used for
            # graph outputs/stacks (mesh mode) otherwise ends up with capacity 0 when HBM is already full.
            self.precompile(warm=True)
            t0 = time.time()
            P = self.npu.put
            self.dd = [dict(in_norm=P(d["in_norm"]), post_norm=P(d["post_norm"]), wqkv=P(d["wqkv"]),
                            wo=P(d["wo"]), sh_gu=P(d["sh_gu"]), sh_d=P(d["sh_d"])) for d in self.dense]
            if slots:
                # per-layer, per-device stacked tables [S, 2F, H] / [S, H, F]
                self.tab = [[None] * self.nL for _ in devs]
                if srv_chip is not None:
                    import opt_client as OC
                    assign = [[e for e, li in sorted(((e, li) for (ll, e), (dd, li) in self.resident.items()
                                                       if ll == l and dd == self.srv_di), key=lambda t: t[1])]
                              for l in range(self.nL)]
                    assert all(len(a) == OC.E_SRV for a in assign), [len(a) for a in assign][:4]
                    wdir = "/dev/shm/solar_srv_w"
                    t1 = time.time(); OC.export_tables(self.hx, assign, self.nL, wdir)
                    print(f"[rt] exported expert_srv tables in {time.time()-t1:.0f}s", flush=True)
                    self.srv = OC.OptExpertServer(wdir, self.nL, chip=srv_chip,
                                                  log=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                        "..", "logs", "expert_srv.log"), "w"))
                    print(f"[rt] expert_srv ready on npu{srv_chip} ({self.srv.t_load:.0f}s)", flush=True)
                    self.dev_pools.append(ThreadPoolExecutor(1))
                for l in range(self.nL):
                    for di in range(len(devs)):
                        es = sorted((e, li) for (ll, e), (dd, li) in self.resident.items() if ll == l and dd == di)
                        es = [e for e, li in sorted(es, key=lambda t: t[1])]
                        gu = torch.stack([self.hx[l][e][0] for e in es]); dn = torch.stack([self.hx[l][e][1] for e in es])
                        self.tab[di][l] = (self.npus[di].put(gu), self.npus[di].put(dn))
                n_up = len(self.resident)
            else:
                self.dx = {(l, e): (self.npus[di].put(self.hx[l][e][0]), self.npus[di].put(self.hx[l][e][1]))
                           for (l, e), di in sorted(self.resident.items())}
                n_up = len(self.dx)
            print(f"[rt] device upload: dense {self.nL}L + {n_up} experts in {time.time()-t0:.0f}s", flush=True)
        S = max_ctx
        cos, sin = self.rot(torch.arange(S))          # precomputed rotary table
        self.cos_t, self.sin_t = cos, sin
        self.kc = [torch.zeros(S, c.nkv, c.hd, dtype=BF) for _ in range(self.nL)]
        self.vc = [torch.zeros(S, c.nkv, c.hd, dtype=BF) for _ in range(self.nL)]
        self.n_past = 0
        self._h2_dev = None
        self.prof = {}
        self.stats = dict(hit=0, miss=0, t_npu=0.0, t_cpu=0.0, t_wait=0.0, swaps=0, t_swap=0.0)
        # online expert swap (PolyStrata follow(): LFU with exponential decay)
        self.swap_per_tok = 0           # 0 = static placement only
        self.recent = torch.zeros(self.nL, c.E)
        self.n_dec = 0

    def precompile(self, warm=False):
        t0 = time.time(); c = self.c; H, F = c.H, c.F
        z = lambda dev, *s, dt=BF: torch.zeros(*s, dtype=dt).to(dev)
        for T in T_BUCKETS:
            g = self.npu.graph("qkv", T)
            if warm:
                d = self.npu.dev
                g(z(d, T, H), z(d, H), z(d, c.nh * c.hd + 2 * c.nkv * c.hd, H)).cpu()
            g = self.npu.graph("post", T)
            if warm:
                g(z(d, T, H), z(d, T, c.nh * c.hd), z(d, H, c.nh * c.hd), z(d, H), z(d, 2 * F, H), z(d, H, F))[0].cpu()
            for di, npu in enumerate(self.npus):
                if self.slots:
                    S = self.slots[di]
                    tab = (z(npu.dev, S, 2 * F, H), z(npu.dev, S, H, F)) if warm else None
                    for K in self.kbuckets(di):
                        g = npu.graph("expg", T, (K, S))
                        if warm:
                            g(z(npu.dev, T, H), z(npu.dev, T, K), z(npu.dev, K, dt=torch.int32), *tab).cpu()
                    del tab
                else:
                    for K in K_BUCKETS:
                        npu.graph("exp", T, K)
        if warm:   # create every persistent staging buffer before HBM fills up (DRAM_SHARED capacity issue)
            for T in T_BUCKETS:
                self.npu.stage("x", torch.zeros(T, H, dtype=BF)); self.npu.stage("a", torch.zeros(T, c.nh * c.hd, dtype=BF))
                for di, npu in enumerate(self.npus):
                    npu.stage("h2", torch.zeros(T, H, dtype=BF))
                    if self.slots:
                        for K in self.kbuckets(di):
                            npu.stage("G", torch.zeros(T, K, dtype=BF)); npu.stage("ids", torch.zeros(K, dtype=torch.int32))
        print(f"[rt] precompiled{' + warmed' if warm else ''} all graphs in {time.time()-t0:.0f}s", flush=True)

    SWAP_HALF = 128   # halve recent counts every N decode tokens
    SWAP_MIN = 2      # swap when in.recent >= 2*out.recent + SWAP_MIN

    def _write_row(self, di, l, li, e):
        """Overwrite row li of device di's layer-l table with host expert e (in place, no recompile)."""
        Wgu, Wd = self.tab[di][l]
        gu, dn = self.hx[l][e]
        Wgu[li].copy_(gu); Wd[li].copy_(dn)

    def follow(self):
        """After a decode token: swap up to swap_per_tok hot non-resident experts in for cold resident ones."""
        self.n_dec += 1
        if self.n_dec % self.SWAP_HALF == 0:
            self.recent *= 0.5
        if not (self.mode == "hybrid" and self.slots and self.swap_per_tok):
            return
        cands = []
        for l in range(self.nL):
            res = {e: v for (ll, e), v in self.resident.items() if ll == l and v[0] != self.srv_di}
            r = self.recent[l]
            nonres = [e for e in torch.argsort(r, descending=True).tolist()[:8]
                      if (l, e) not in self.resident and r[e] > 0]
            if not nonres: continue
            e_in = nonres[0]
            e_out = min(res, key=lambda e: float(r[e]))
            if r[e_in] >= 2 * r[e_out] + self.SWAP_MIN:
                cands.append((float(r[e_in] - r[e_out]), l, e_in, e_out))
        cands.sort(reverse=True)
        for gain, l, e_in, e_out in cands[:self.swap_per_tok]:
            t0 = time.perf_counter()
            di, li = self.resident.pop((l, e_out))   # e_out -> CPU from now on
            self._write_row(di, l, li, e_in)
            self.resident[(l, e_in)] = (di, li)      # e_in -> NPU only after the write completes
            self.stats["swaps"] += 1; self.stats["t_swap"] += time.perf_counter() - t0

    def kbuckets(self, di):
        # K must factor nicely (K=26 -> "O666 flattening failed"); K may exceed S (padded ids, zero gate)
        S = self.slots[di]; ks = (8, 16, 24, 32, 48, 64)
        return tuple([k for k in ks if k < S] + [min(k for k in ks if k >= S)])

    def reset(self):
        self.n_past = 0

    # ---- expert part ----
    def _cpu_experts(self, l, h, jobs, out):
        t0 = time.perf_counter()
        if CPU_KERNEL:   # fused AVX-512 BF16 kernel: all of this layer's CPU experts in one call
            from cpu_moe_ext import ext
            gus = [self.hx[l][e][0] for e, _, _ in jobs]; ds = [self.hx[l][e][1] for e, _, _ in jobs]
            out.append(ext.cpu_moe(h.contiguous(), gus, ds, [r.long() for _, r, _ in jobs],
                                   [w.float() for _, _, w in jobs]))
            self.stats["t_cpu"] += time.perf_counter() - t0
            return
        acc = torch.zeros(h.shape[0], self.c.H, dtype=torch.float32)
        for e, rows, w in jobs:
            gu, d = self.hx[l][e]
            acc[rows] += swiglu_ffn(h[rows], gu, d).float() * w[:, None]
        out.append(acc)
        self.stats["t_cpu"] += time.perf_counter() - t0

    def _moe(self, l, h2, Tb, T):
        c = self.c
        tr = time.perf_counter()
        idx, w = route(c, h2[:T], self.dense[l]["router"], self.dense[l]["bias"])
        tr = self._tick("moe.route", tr)
        # group token rows by expert
        per = {}
        for t in range(T):
            for j in range(c.topk):
                per.setdefault(int(idx[t, j]), []).append((t, float(w[t, j])))
        if T <= 8:                       # decode-like steps drive the online cache
            for e, rows in per.items():
                self.recent[l, e] += len(rows)
        hits = [e for e in per if (l, e) in self.resident] if self.mode == "hybrid" else []
        miss = [e for e in per if e not in set(hits)]
        self.stats["hit"] += sum(len(per[e]) for e in hits); self.stats["miss"] += sum(len(per[e]) for e in miss)
        jobs = [(e, torch.tensor([r for r, _ in per[e]]), torch.tensor([x for _, x in per[e]])) for e in miss]
        tr = self._tick("moe.group", tr)
        cpu_out = []
        th = None
        if jobs:
            th = self.cpu_pool.submit(self._cpu_experts, l, h2[:T], jobs, cpu_out)
        npu_out = torch.zeros(T, c.H, dtype=torch.float32)
        if hits:
            t0 = time.perf_counter()
            outs = []
            def run_dev_g(di, dh):
                npu = self.npus[di]; S = self.slots[di]
                t_ = time.perf_counter()
                hd = self._h2_dev if (di == 0 and self._h2_dev is not None) else npu.stage("h2", h2)
                self._tick(f"dev{di}.h2", t_)
                K = bucket(len(dh), self.kbuckets(di))
                G = torch.zeros(Tb, K, dtype=BF); ids = torch.zeros(K, dtype=torch.int32)
                for i, e in enumerate(dh):
                    ids[i] = self.resident[(l, e)][1]
                    for r, x in per[e]:
                        G[r, i] = x
                ids[len(dh):] = ids[0]
                Wgu, Wd = self.tab[di][l]
                ta = time.perf_counter()
                Gd, idd = npu.stage("G", G), npu.stage("ids", ids)
                tb = self._tick(f"dev{di}.stage", ta)
                yd = npu.graph("expg", Tb, (K, S))(hd, Gd, idd, Wgu, Wd)
                tc = self._tick(f"dev{di}.graph", tb)
                outs.append(yd.cpu()[:T].float())
                self._tick(f"dev{di}.d2h", tc)

            def run_dev(di, dh):
                npu = self.npus[di]; hd = npu.put(h2); acc = torch.zeros(T, c.H, dtype=torch.float32)
                for s in range(0, len(dh), K_BUCKETS[-1]):
                    chunk = dh[s:s + K_BUCKETS[-1]]
                    K = bucket(len(chunk), K_BUCKETS)
                    G = torch.zeros(Tb, K, dtype=BF); ws = []
                    for i, e in enumerate(chunk):
                        for r, x in per[e]:
                            G[r, i] = x
                        ws += list(self.dx[(l, e)])
                    pad = self.dx[(l, chunk[0])]
                    for i in range(len(chunk), K):
                        ws += list(pad)  # zero gate -> no contribution
                    acc += npu.graph("exp", Tb, K)(hd, npu.put(G), *ws).cpu()[:T].float()
                outs.append(acc)
            by_dev = {}
            for e in hits:
                r_ = self.resident[(l, e)]
                by_dev.setdefault(r_[0] if self.slots else r_, []).append(e)
            def run_srv(di, dh):
                acc = torch.zeros(T, c.H, dtype=torch.float32)
                for t in range(T):
                    es = [(e, x) for e in dh for r, x in per[e] if r == t]
                    if not es: continue
                    ys = self.srv.ffn(l, h2[t], [self.resident[(l, e)][1] for e, _ in es])
                    acc[t] += (ys * torch.tensor([x for _, x in es])[:, None]).sum(0)
                outs.append(acc)

            fn0 = run_dev_g if self.slots else run_dev
            fn = lambda di, dh: (run_srv if (self.srv is not None and di == self.srv_di) else fn0)(di, dh)
            futs = [self.dev_pools[di].submit(fn, di, dh) for di, dh in by_dev.items()]
            for f_ in futs:
                f_.result()          # re-raises worker exceptions
            for o in outs: npu_out += o
            self.stats["t_npu"] += time.perf_counter() - t0
        if th:
            t0 = time.perf_counter(); th.result(); self.stats["t_wait"] += time.perf_counter() - t0
            npu_out += cpu_out[0]
        return npu_out

    # ---- layer ----
    def _tick(self, name, t0):
        t = time.perf_counter()
        if T_PROF:
            self.prof[name] = self.prof.get(name, 0.0) + (t - t0)
        return t

    def layer(self, l, x, pos, Tb, T):
        c = self.c; d = self.dense[l]
        t0 = time.perf_counter()
        if self.mode == "hybrid":
            dd = self.dd[l]
            qkv = self.npu.graph("qkv", Tb)(self.npu.stage("x", x), dd["in_norm"], dd["wqkv"]).cpu()[:T]
        else:
            qkv = rmsnorm(x[:T], d["in_norm"], c.eps) @ d["wqkv"].t()
        t0 = self._tick("qkv", t0)
        nq, nk = c.nh * c.hd, c.nkv * c.hd
        q = qkv[:, :nq].view(T, c.nh, c.hd); k = qkv[:, nq:nq + nk].view(T, c.nkv, c.hd)
        v = qkv[:, nq + nk:].view(T, c.nkv, c.hd)
        cos, sin = self.cos_t[pos], self.sin_t[pos]
        t0 = self._tick("rotary", t0)
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        self.kc[l][self.n_past:self.n_past + T] = k; self.vc[l][self.n_past:self.n_past + T] = v
        a = attention(c, q, self.kc[l], self.vc[l], self.n_past, T)
        t0 = self._tick("attn", t0)
        if self.mode == "hybrid":
            ap = torch.zeros(Tb, a.shape[1], dtype=BF); ap[:T] = a
            y, h2 = self.npu.graph("post", Tb)(self.npu.stage("x", x), self.npu.stage("a", ap), dd["wo"], dd["post_norm"],
                                                dd["sh_gu"], dd["sh_d"])
            self._h2_dev = h2
            t0 = self._tick("post.graph", t0)
            y = y.cpu(); h2 = h2.cpu()
        else:
            x2 = x[:T] + a @ d["wo"].t()
            h2 = rmsnorm(x2, d["post_norm"], c.eps)
            y = x2 + swiglu_ffn(h2, d["sh_gu"], d["sh_d"])
            yp = torch.zeros(Tb, c.H, dtype=BF); yp[:T] = y; y = yp
            hp = torch.zeros(Tb, c.H, dtype=BF); hp[:T] = h2; h2 = hp
        t0 = self._tick("post", t0)
        moe = self._moe(l, h2, Tb, T)
        t0 = self._tick("moe", t0)
        out = torch.zeros(Tb, c.H, dtype=BF)
        out[:T] = (y[:T].float() + moe).to(BF)
        self._tick("combine", t0)
        return out

    def forward(self, ids):
        """ids: list[int] (T <= 16). returns logits [T, vocab] fp32 for the new tokens."""
        T = len(ids); Tb = bucket(T, T_BUCKETS)
        assert T <= T_BUCKETS[-1]
        x = torch.zeros(Tb, self.c.H, dtype=BF); x[:T] = self.embed[torch.tensor(ids)]
        pos = torch.arange(self.n_past, self.n_past + T)
        for l in range(self.nL):
            x = self.layer(l, x, pos, Tb, T)
        self.n_past += T
        t0 = time.perf_counter()
        h = rmsnorm(x[:T], self.norm, self.c.eps)
        lg = (h @ self.lm.t()).float()
        self._tick("lm_head", t0)
        return lg

    def prefill(self, ids):
        out = None
        for s in range(0, len(ids), T_BUCKETS[-1]):
            out = self.forward(ids[s:s + T_BUCKETS[-1]])
        return out

    def generate(self, ids, n):
        self.reset(); out = []
        lg = self.prefill(ids)
        tok = int(lg[-1].argmax()); out.append(tok)
        self.prof = {}                       # decode-only profile from here
        times = []
        for _ in range(n - 1):
            t0 = time.perf_counter()
            lg = self.forward([tok]); tok = int(lg[-1].argmax()); out.append(tok)
            self.follow()
            times.append(time.perf_counter() - t0)
        return out, times


def load_resident(path, nL, budget_experts, E=128, caps=None):
    """Rank (l,e) by profile counts (routes.pt), take top budget; assign to devices alternately
    (balances each layer's hits across cards) until a device's capacity (in experts) is full."""
    keys = _rank(path, nL, budget_experts, E)
    caps = list(caps or [len(keys)])
    out, d = {}, 0
    for k in keys:
        for _ in range(len(caps)):
            if caps[d] > 0: break
            d = (d + 1) % len(caps)
        if caps[d] <= 0: break
        out[k] = d; caps[d] -= 1; d = (d + 1) % len(caps)
    return out


def per_layer_slots(path, nL, slots, E=128):
    """Top sum(slots) experts per layer by profile counts; alternate devices -> {(l,e): (dev, local_idx)}."""
    c = _counts(path, nL, E)
    out = {}
    for l in range(nL):
        order = torch.argsort(c[l], descending=True).tolist()
        fill = [0] * len(slots); di = 0
        for e in order:
            if sum(fill) >= sum(slots): break
            while fill[di] >= slots[di]: di = (di + 1) % len(slots)
            out[(l, e)] = (di, fill[di]); fill[di] += 1; di = (di + 1) % len(slots)
    return out


def _counts(path, nL, E=128):
    d = torch.load(path)
    c = torch.zeros(nL, E)
    for r in d["recs"]:
        for key in ("prefill", "decode"):
            x = r[key]
            if x is None: continue
            for l in range(nL):
                c[l] += torch.bincount(x[l].flatten().long(), minlength=E).float()
    return c


def _rank(path, nL, budget_experts, E=128):
    if path and os.path.exists(path):
        d = torch.load(path)
        c = torch.zeros(nL, E)
        for r in d["recs"]:
            for key in ("prefill", "decode"):
                x = r[key]
                if x is None: continue
                for l in range(nL):
                    c[l] += torch.bincount(x[l].flatten().long(), minlength=E).float()
        order = torch.argsort(c.flatten(), descending=True)[:budget_experts]
        return [(int(i) // E, int(i) % E) for i in order]
    per = budget_experts // nL
    return [(l, e) for e in range(per) for l in range(nL)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--mode", default="both", choices=["cpu", "hybrid", "both"])
    ap.add_argument("--budget", type=float, default=0.4, help="fraction of routed experts resident")
    ap.add_argument("--profile", default=os.path.join(os.path.dirname(__file__), "..", "p0", "results", "routes.pt"))
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--cpu-threads", type=int, default=12)
    ap.add_argument("--devs", default="rngd:0")
    ap.add_argument("--caps", default="", help="experts per device, e.g. 1150,1400")
    ap.add_argument("--slots", default="", help="device-gather mode: experts per layer per device, e.g. 20,26")
    ap.add_argument("--srv", type=int, default=None, help="chip index for the furiosa-opt expert server (32 experts/layer)")
    ap.add_argument("--swap", type=int, default=0, help="online expert swaps per decode token (gather mode)")
    ap.add_argument("--chat", action="store_true", help="apply chat template to prompt")
    ap.add_argument("--warmup", type=int, default=1, help="untimed generate runs first (compiles graphs)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    W = Weights(); c = W.cfg
    nL = a.layers or c.L
    tk = AutoTokenizer.from_pretrained(W.dir)
    if a.chat:
        ids = tk.apply_chat_template([{"role": "user", "content": a.prompt}], add_generation_prompt=True)
        ids = list(ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids)
    else:
        ids = tk.encode(a.prompt, add_special_tokens=False)
    print(f"prompt tokens: {len(ids)}", flush=True)
    res = {}
    for mode in (["cpu", "hybrid"] if a.mode == "both" else [a.mode]):
        devs = a.devs.split(",")
        caps = [int(x) for x in a.caps.split(",")] if a.caps else None
        slots = [int(x) for x in a.slots.split(",")] if a.slots else None
        if mode != "hybrid":
            resident = None
        elif slots:
            # the expert server is the last "device" with 32 slots
            resident = per_layer_slots(a.profile, nL, slots + ([32] if a.srv is not None else []), c.E)
        else:
            resident = load_resident(a.profile, nL, int(a.budget * nL * c.E), c.E, caps)
        rt = Runtime(W, nL, mode, resident, devs=devs, cpu_threads=a.cpu_threads, slots=slots,
                     srv_chip=a.srv if mode == "hybrid" else None)
        rt.swap_per_tok = a.swap
        for _ in range(a.warmup if mode == "hybrid" else 0):
            rt.generate(ids, a.n)
        rt.stats = dict(hit=0, miss=0, t_npu=0.0, t_cpu=0.0, t_wait=0.0, swaps=0, t_swap=0.0)
        rt.prof = {}
        toks, times = rt.generate(ids, a.n)
        if T_PROF:
            tot = sum(v for k, v in rt.prof.items() if "." not in k)
            print(f"[prof] (decode only, {len(times)} tok) " + " ".join(f"{k}={v:.2f}s" for k, v in
                  sorted(rt.prof.items(), key=lambda kv: -kv[1])) + f" | sum_top={tot:.2f}s", flush=True)
        st = rt.stats
        res[mode] = toks
        ms = 1e3 * sum(times) / max(1, len(times))
        print(f"[{mode}] tokens={toks} text={tk.decode(toks)!r} decode {ms:.0f} ms/tok "
              f"hit={st['hit']} miss={st['miss']} t_npu={st['t_npu']:.2f}s t_cpu={st['t_cpu']:.2f}s "
              f"wait={st['t_wait']:.2f}s swaps={st.get('swaps',0)} t_swap={st.get('t_swap',0):.2f}s "
              f"hit_rate={st['hit']/max(1,st['hit']+st['miss']):.3f}", flush=True)
        if getattr(rt, "srv", None) is not None:
            print(f"[rt] expert_srv calls={rt.srv.calls} avg_wait={1e6*rt.srv.t_wait/max(1,rt.srv.calls):.0f}us", flush=True)
            rt.srv.close()
        del rt
    if len(res) == 2:
        print("MATCH" if res["cpu"] == res["hybrid"] else "MISMATCH", res)
