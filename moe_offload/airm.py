"""ai-rm protocol v0.1 adapter (rm-core contracts) for npu-offload.

Registered as the workload runner `npu-offload` (entry-point group `airm.workloads`):

    from rm_core import registry, Workload
    run = registry.load("airm.workloads", "npu-offload")        # no side effects, no device access
    run(Workload(kind="serve", model="solar-open-100b"), engine=..., data=..., hf_dir=..., port=8101)

Expert offloading splits weights and activations across tiers, not KV cache, so the runner uses no KVTransport
(`kv_transport` must be empty). It declares the memory tiers it uses (`device-hbm` per RNGD, `host-dram`) in its
start log and on /metrics, and adds the shared airm_* series (tok_s, tpot_p50_ms, tpot_p90_ms, power_w) to the
worker's /metrics. Configuration keys: README, section "ai-rm".
"""
import json, os, re, socket, subprocess, sys, threading, time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

from rm_core import ContractError, Measurement, MemoryTier, Workload, prometheus_name

from . import siteenv

NAME = "npu-offload"
KINDS = ("serve", "decode")


def node_name():
    """Node part of accelerator ids (`<node>/furiosa:<i>`): $NODE_NAME (k8s downward API), else the hostname."""
    n = re.sub(r"[^a-z0-9._-]", "-", (os.environ.get("NODE_NAME") or socket.gethostname()).lower()).lstrip("._-")
    return n or "node"


def _smi(cmd):
    try:
        out = subprocess.run(["furiosa-smi", cmd, "--format", "json"], capture_output=True, text=True, timeout=10)
        return json.loads(out.stdout) if out.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def memory_tiers(node=None):
    """Tiers the engine uses, with capacities read from the machine: one `device-hbm` per RNGD (furiosa-smi) and
    `host-dram` (physical memory). No RNGD found -> host tier only."""
    node = node or node_name()
    tiers = [MemoryTier(f"rngd{i}-hbm", "device-hbm", int(d["memory"]["total_size"]), f"{node}/furiosa:{i}").validate()
             for i, d in enumerate(_smi("status") or [])]
    host = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return tiers + [MemoryTier("host-dram", "host-dram", host).validate()]


def card_power(node=None):
    """{accelerator_id: watts} from furiosa-smi; {} when unavailable."""
    node = node or node_name()
    out = {}
    for i, d in enumerate(_smi("info") or []):
        m = re.match(r"\s*([\d.]+)\s*W", str(d.get("power", "")))
        if m: out[f"{node}/furiosa:{i}"] = float(m.group(1))
    return out


def _pct(xs, q):
    """Nearest-rank percentile."""
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, -(-len(xs) * q // 100) - 1))]


def decode_measurements(samples, tags):
    """[(tokens, decode ms/token), ...] per request -> tok_s (tokens per decode second), tpot_p50_ms, tpot_p90_ms."""
    if not samples: return []
    ms = [m for _, m in samples]
    tok_s = sum(n for n, _ in samples) / (sum(n * m for n, m in samples) / 1000)
    return [Measurement("tok_s", round(tok_s, 3), tags).validate(),
            Measurement("tpot_p50_ms", round(_pct(ms, 50), 3), tags).validate(),
            Measurement("tpot_p90_ms", round(_pct(ms, 90), 3), tags).validate()]


def prometheus_lines(measurements):
    return "".join(f"{prometheus_name(m.name)}{{{','.join(f'{k}={json.dumps(v)}' for k, v in sorted(m.tags.items()))}}} {m.value}\n"
                   for m in measurements)


def tier_lines(tiers):
    return "".join(f'moe_offload_memory_tier_capacity_bytes{{tier="{t.name}",kind="{t.kind}",accelerator="{t.accelerator_id}"}} '
                   f"{t.capacity_bytes}\n" for t in tiers)


class AirmMetrics:
    """/metrics hook for a worker: memory tiers + airm_* series. Card power is cached for `power_every` seconds."""
    def __init__(self, worker, tiers, model, power_every=5.0):
        self.w, self.tiers, self.power_every = worker, tiers, power_every
        self.tags = {"component": NAME, "model": model, "workload": "serve"}
        self.lock = threading.Lock(); self.power, self.power_t = {}, 0.0

    def _power(self):
        with self.lock:
            if time.time() - self.power_t > self.power_every:
                self.power, self.power_t = card_power(), time.time()
            return self.power

    def __call__(self):
        with self.w.st.lock: samples = list(self.w.st.recent)
        ms = decode_measurements(samples, self.tags)
        ms += [Measurement("power_w", w, {"component": NAME, "scope": "card", "accelerator": a}).validate()
               for a, w in sorted(self._power().items())]
        return tier_lines(self.tiers) + prometheus_lines(ms)


def _log(msg):
    print(f"[moe_offload.airm] {msg}", flush=True)


class Runner:
    """Workload runner (PROTOCOL.md §5.3). Construction is free of side effects; calling it starts the work."""
    name = NAME

    def __call__(self, workload, *, engine=None, data=None, hf_dir=None, role="worker", host="127.0.0.1", port=8101,
                 metrics_port=None, gateway=None, wid=None, env=None, log="engine_serve.log", priors=None,
                 mock=False, mock_ms=1.0, mock_hit="0.85", kv_transport="", prompt=None, max_new=32, repeat=1,
                 block=True):
        if not isinstance(workload, Workload): raise ContractError("npu-offload runner needs an rm_core.Workload")
        workload.validate()
        if workload.kind not in KINDS:
            raise ContractError(f"npu-offload runs Workload.kind in {KINDS}, not {workload.kind!r}")
        if kv_transport:
            raise ContractError(f"npu-offload moves expert weights/activations, not KV: no KV transport "
                                f"(got kv_transport={kv_transport!r})")
        if role == "gateway":
            if workload.kind != "serve": raise ContractError("role=gateway needs Workload.kind 'serve'")
            return self._gateway(host, port, block)
        if role != "worker": raise ContractError(f"role must be 'worker' or 'gateway', not {role!r}")
        if not mock and not (engine and data and hf_dir):
            raise ContractError("npu-offload needs engine=, data= and hf_dir= (or mock=True)")
        tiers = memory_tiers()
        for t in tiers: _log("memory tier " + json.dumps(vars(t)))
        args = SimpleNamespace(engine=engine, data=data, model=hf_dir, model_name=workload.model, mock=mock,
                               mock_ms=float(mock_ms), mock_hit=str(mock_hit), log=log,
                               wid=wid or socket.gethostname() + "-moe0",
                               env=[f"{k}={v}" for k, v in (env or {}).items()])
        from . import worker as wk
        w = wk.build(args)
        if workload.kind == "decode":
            return self._decode(w, workload, prompt, int(max_new), int(repeat))
        return self._serve(wk, w, workload, tiers, host, int(port), metrics_port, gateway, priors, block)

    def _serve(self, wk, w, workload, tiers, host, port, metrics_port, gateway, priors, block):
        w.metric_hooks.append(AirmMetrics(w, tiers, workload.model))
        srvs = [ThreadingHTTPServer((host, port), wk.make_handler(w))]
        if metrics_port is not None and (int(metrics_port) == 0 or int(metrics_port) != port):   # 0: any free port
            srvs.append(ThreadingHTTPServer((host, int(metrics_port)), wk.make_handler(w)))
        for s in srvs: threading.Thread(target=s.serve_forever, daemon=True).start()
        if gateway is None:
            siteenv.load(); gateway = os.environ.get("LLMD_GATEWAY_URL", "")
        if gateway:
            w.attach(gateway, f"http://{host}:{srvs[0].server_address[1]}",
                     priors=json.loads(priors) if isinstance(priors, str) else priors)
        _log(f"{w.wid} serving {w.model} on http://{host}:{srvs[0].server_address[1]}"
             + (f", metrics on :{srvs[1].server_address[1]}" if len(srvs) > 1 else ""))
        if not block: return SimpleNamespace(worker=w, servers=srvs)
        while True: time.sleep(3600)

    def _gateway(self, host, port, block):
        from .llmd import gateway as gw          # imports rngd_llmd ($RNGD_LLMD) only when a gateway is asked for
        srv = gw.serve(gw.Gateway(), host, int(port))
        _log(f"gateway on http://{host}:{srv.server_address[1]}")
        if not block: return SimpleNamespace(servers=[srv])
        while True: time.sleep(3600)

    def _decode(self, w, workload, prompt, max_new, repeat):
        """One prompt `repeat` times through the engine; returns the measurements and prints report.py args."""
        ids = list(prompt) if isinstance(prompt, (list, tuple)) else w._encode(prompt or "Hello")
        samples = []
        for _ in range(repeat):
            toks, rec = w.eng.run(ids, max_new)
            samples.append((len(toks), rec["decode_ms_per_tok"]))
        ms = decode_measurements(samples, {"component": NAME, "model": workload.model, "workload": "decode"})
        _log("report.py " + " ".join(f"--metric {m.name}={m.value}" for m in ms))
        return ms


def serve(workload=None, **config):
    """Entry point `airm.workloads: npu-offload`. Without a workload (what `registry.load` does) it returns the
    runner; with one it runs it: `serve(Workload(kind="serve", model=...), engine=..., ...)`."""
    r = Runner()
    return r if workload is None else r(workload, **config)


def main(argv=None):
    """moe-offload airm --kind serve|decode --model NAME [key=value ...]: run the runner from the command line."""
    import argparse
    ap = argparse.ArgumentParser(prog="moe-offload airm")
    ap.add_argument("--kind", default="serve", choices=KINDS); ap.add_argument("--model", default="solar-open-100b")
    ap.add_argument("config", nargs="*", help="runner configuration as key=value (JSON values allowed)")
    a = ap.parse_args(argv)
    cfg = {}
    for kv in a.config:
        k, v = kv.split("=", 1)
        try: cfg[k] = json.loads(v)
        except ValueError: cfg[k] = v
    out = serve(Workload(kind=a.kind, model=a.model), **cfg)
    if a.kind == "decode": print(json.dumps([vars(m) for m in out]))


if __name__ == "__main__":
    sys.exit(main())
