"""Drive a data-plane engine binary in serve mode (`<engine> <data_dir> serve`).

Protocol (stdin/stdout JSON lines): request {"ids": [...], "max_new": N, "stop": [...]} -> {"tok": t} per token,
then {"done": true, "n", "prefill_s", "decode_ms_per_tok", "hit_rate"} or {"error": ...}. One stream at a time.
"""
import json, os, subprocess, threading, time

# engine settings that reproduced the best exact decode (PLAN.md 7.3-7.4, experiments/2026-10-16-*)
SOLAR_ENV = dict(PIN_BOUNCE="1", MMIO_VEC="2", MMIO_LIMIT="32768", SQPOLL_OFF="1", DMA_IDLE_US="50", URES="1",
                 LM_ON_B="1", OMP_WAIT_POLICY="passive", GOMP_SPINCOUNT="100000", CHIP_CORES="7", CPU_THREADS="17",
                 OMP_PLACES="{7}:17:1", OMP_PROC_BIND="close", FURIOSA_VISIBLE_CHIPS="0,1")


class EngineError(RuntimeError):
    pass


class EngineProcess:
    def __init__(self, binary, data_dir, env=None, log_path=None, ready_timeout=900):
        self.lock = threading.Lock()
        e = dict(os.environ); e.update(env or {})
        self.log = open(log_path, "ab") if log_path else subprocess.DEVNULL
        self.p = subprocess.Popen([binary, data_dir, "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.log, env=e, text=True, bufsize=1)
        t0 = time.time()
        line = self.p.stdout.readline()
        if line.strip() != "READY":
            raise EngineError(f"engine did not start (got {line!r}); see {log_path}")
        self.load_s = time.time() - t0

    def generate(self, ids, max_new=64, stop=(), stop_count=1):
        """Yields token ids; the final value of the generator's StopIteration is the 'done' record."""
        with self.lock:
            self.p.stdin.write(json.dumps({"ids": list(ids), "max_new": int(max_new), "stop": list(stop),
                                           "stop_count": int(stop_count)}) + "\n")
            self.p.stdin.flush()
            while True:
                line = self.p.stdout.readline()
                if not line: raise EngineError("engine exited")
                r = json.loads(line)
                if "tok" in r: yield r["tok"]
                elif r.get("done"): return r
                elif "error" in r: raise EngineError(r["error"])

    def run(self, ids, max_new=64, stop=(), stop_count=1):
        g = self.generate(ids, max_new, stop, stop_count); toks = []
        while True:
            try: toks.append(next(g))
            except StopIteration as s: return toks, s.value

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.close()
            try: self.p.wait(30)
            except subprocess.TimeoutExpired: self.p.kill()


class MockEngine:
    """NPU-free stand-in with the same interface (tests, gateway demos)."""
    def __init__(self, ms_per_tok=1.0, hit_rate=0.85, vocab=196608):
        self.ms, self.hit, self.vocab, self.lock, self.load_s = ms_per_tok, hit_rate, vocab, threading.Lock(), 0.0

    def generate(self, ids, max_new=64, stop=(), stop_count=1):
        with self.lock:
            t, seen = ids[-1], 0
            for i in range(max_new):
                time.sleep(self.ms / 1000); t = (t * 31 + 7) % self.vocab
                yield t
                seen += t in stop
                if seen >= stop_count: break
            hit = self.hit
            if isinstance(hit, dict):   # class-dependent (mock workers with different placements)
                from .llmd.affinity import classify
                hit = hit.get(classify(bytes(t for t in ids if t < 256).decode("utf-8", "ignore")), 0.8)
            return {"done": True, "n": i + 1, "prompt": len(ids), "prefill_s": 0.0, "decode_ms_per_tok": self.ms, "hit_rate": hit}

    def run(self, ids, max_new=64, stop=(), stop_count=1):
        g = self.generate(ids, max_new, stop, stop_count); toks = []
        while True:
            try: toks.append(next(g))
            except StopIteration as s: return toks, s.value

    def close(self): pass
