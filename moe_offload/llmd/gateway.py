"""MoE-aware gateway on top of rngd_llmd's registry/EPP: routes OpenAI requests to offloading workers by
expected expert residency (per-class hit-rate EWMA, seeded by placement priors) minus load, and proxies them
over HTTP (the HTTP dispatcher rngd_llmd's serving_app left as a deployment blueprint).

  python3 -m moe_offload.llmd.gateway --port 8080
Workers (moe_offload.worker --gateway http://127.0.0.1:8080) register themselves.

Endpoints: POST /v1/completions, /v1/chat/completions (proxied), /register, /heartbeat, /moe_event, /kv_event;
GET /metrics, /healthz, /workers.
"""
import argparse, json, os, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .affinity import AffinityTable, classify, score
from .. import siteenv

RNGD_LLMD = siteenv.require("RNGD_LLMD", "path to an rngd_llmd checkout; its gateway/ package is imported")
sys.path.insert(0, os.path.join(RNGD_LLMD, "gateway"))
from registry import WorkerRegistry, Role  # noqa: E402  (rngd_llmd)
from kv_events import KVIndex  # noqa: E402
from epp import EndpointPicker, RouteDecision  # noqa: E402


class MoEAffinityPicker(EndpointPicker):
    """EndpointPicker with an expert-affinity scorer; falls back to rngd_llmd's KV/least-load policy on ties."""
    def __init__(self, registry, kv_index, affinity, block_size=16):
        super().__init__(registry, kv_index, block_size)
        self.aff = affinity

    def route_text(self, text, token_ids=None):
        cls = classify(text)
        cand = self.reg.alive(Role.DECODE)
        if not cand: raise RuntimeError("no live decode-capable worker")
        scored = sorted(((score(self.aff.get(w.wid, cls), w.inflight), w.wid) for w in cand), reverse=True)
        best_s, best = scored[0]
        reason = f"class={cls} affinity " + ", ".join(f"{wid}:{s:.3f}" for s, wid in scored)
        return RouteDecision("replica", decode_wid=best, prefill_wid=best, kv_hit_blocks=0, reason=reason), cls


class Gateway:
    def __init__(self, timeout=600):
        self.reg = WorkerRegistry(stale_after=30.0)
        self.kv = KVIndex()
        self.aff = AffinityTable()
        self.epp = MoEAffinityPicker(self.reg, self.kv, self.aff)
        self.timeout = timeout
        self.lock = threading.Lock()
        self.routed = {}

    def register(self, b):
        w = self.reg.register(b["wid"], b.get("role", "both"), b.get("chip", 0), b.get("endpoint", ""))
        for cls, hit in (b.get("priors") or {}).items(): self.aff.set_prior(b["wid"], cls, float(hit))
        return {"ok": True, "wid": w.wid}

    def heartbeat(self, b):
        self.reg.heartbeat(b["wid"], b.get("inflight")); return {"ok": True}

    def moe_event(self, b):
        self.aff.observe(b["wid"], b.get("class", "en"), float(b.get("hit_rate", 0.0))); return {"ok": True}

    def proxy(self, path, body):
        text = body.get("prompt") or " ".join(m.get("content", "") for m in body.get("messages", []) if m.get("role") == "user")
        d, cls = self.epp.route_text(text if isinstance(text, str) else "")
        w = self.reg.get(d.decode_wid)
        self.reg.incr(w.wid, 1)
        try:
            req = urllib.request.Request(w.endpoint.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r: out = json.loads(r.read())
        finally:
            self.reg.incr(w.wid, -1)
        with self.lock:
            k = (w.wid, cls); self.routed[k] = self.routed.get(k, 0) + 1
        out["routing"] = {"worker": w.wid, "class": cls, "reason": d.reason}
        return out

    def metrics(self):
        with self.lock: routed = dict(self.routed)
        L = [f'moe_gateway_routed_total{{worker="{w}",class="{c}"}} {n}' for (w, c), n in sorted(routed.items())]
        for w in self.reg.alive():
            L.append(f'moe_gateway_worker_inflight{{worker="{w.wid}"}} {w.inflight}')
            for c in ("ko", "code", "en"):
                L.append(f'moe_gateway_affinity{{worker="{w.wid}",class="{c}"}} {self.aff.get(w.wid, c):.4f}')
        return "\n".join(L) + "\n"


def make_handler(g):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        def _send(self, code, obj, ctype="application/json"):
            data = obj.encode() if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            p = self.path.rstrip("/")
            if p == "/metrics": self._send(200, g.metrics(), "text/plain; version=0.0.4")
            elif p == "/healthz": self._send(200, {"ok": True})
            elif p == "/workers": self._send(200, [vars(w) | {"role": w.role.value} for w in g.reg.alive()])
            else: self._send(404, {"error": "not found"})
        def do_POST(self):
            p = self.path.rstrip("/")
            n = int(self.headers.get("Content-Length", 0))
            try:
                b = json.loads(self.rfile.read(n) or b"{}")
                if p in ("/v1/completions", "/v1/chat/completions"): out = g.proxy(p, b)
                elif p == "/register": out = g.register(b)
                elif p == "/heartbeat": out = g.heartbeat(b)
                elif p == "/moe_event": out = g.moe_event(b)
                elif p == "/kv_event": out = {"ok": True}
                else: return self._send(404, {"error": "not found"})
                self._send(200, out)
            except Exception as e:
                self._send(502, {"error": f"{type(e).__name__}: {e}"})
    return H


def serve(g, host="127.0.0.1", port=8080):
    srv = ThreadingHTTPServer((host, port), make_handler(g))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    serve(Gateway(), a.host, a.port)
    print(f"[moe_offload.gateway] http://{a.host}:{a.port} (rngd_llmd from {RNGD_LLMD})", flush=True)
    while True: time.sleep(3600)


if __name__ == "__main__":
    main()
