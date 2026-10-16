"""OpenAI-compatible HTTP worker around one offloading engine, speaking the rngd_llmd gateway protocol.

  POST /v1/completions        {"prompt": str | "prompt_token_ids"/"tokens": [...], "max_tokens"/"max_new": N}
  POST /v1/chat/completions   {"messages": [...], "max_tokens": N}  (reasoning split like vLLM's reasoning parser)
  GET  /v1/models, /health, /healthz
  GET  /metrics               vLLM-compatible gauges (scraped by llm-d's EPP) + moe_offload_* series
With --gateway URL the worker registers (role "both"), heartbeats its in-flight count, and posts a
/moe_event {wid, class, hit_rate} after every request for the gateway's expert-affinity scorer.

  python3 -m moe_offload.worker --engine track2_opt/solar_engine/target/release/solar_engine \
      --data track2_opt/solar_engine/data --model <hf dir> --port 8101 [--gateway http://127.0.0.1:8080]
  python3 -m moe_offload.worker --mock --port 8101            # NPU-free
"""
import argparse, json, os, socket, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import siteenv
from .engine import EngineProcess, MockEngine, SOLAR_ENV
from .llmd.affinity import classify

THINK, END, BEGIN, CONTENT = 22, 21, 20, 23   # Solar special tokens


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = self.waiting = 0
        self.requests = self.tokens = 0
        self.hit_ewma = None; self.ms_ewma = None; self.ctx = 0
        self.cls = {}

    def done(self, cls, rec, n_out):
        with self.lock:
            self.requests += 1; self.tokens += n_out
            h, ms = rec.get("hit_rate", 0.0), rec.get("decode_ms_per_tok", 0.0)
            self.hit_ewma = h if self.hit_ewma is None else 0.7 * self.hit_ewma + 0.3 * h
            if ms: self.ms_ewma = ms if self.ms_ewma is None else 0.7 * self.ms_ewma + 0.3 * ms
            c = self.cls.setdefault(cls, [0, 0.0]); c[0] += 1; c[1] = h if c[0] == 1 else 0.7 * c[1] + 0.3 * h

    def prometheus(self, model, ctx_max):
        with self.lock:
            L = [f'vllm:num_requests_running{{model_name="{model}"}} {self.running}',
                 f'vllm:num_requests_waiting{{model_name="{model}"}} {self.waiting}',
                 f'vllm:kv_cache_usage_perc{{model_name="{model}"}} {self.ctx / ctx_max:.4f}',
                 f'vllm:gpu_cache_usage_perc{{model_name="{model}"}} {self.ctx / ctx_max:.4f}',
                 f"moe_offload_requests_total {self.requests}",
                 f"moe_offload_generation_tokens_total {self.tokens}",
                 f"moe_offload_expert_hit_rate {self.hit_ewma or 0.0:.4f}",
                 f"moe_offload_decode_ms_per_token {self.ms_ewma or 0.0:.2f}"]
            L += [f'moe_offload_class_hit_rate{{class="{c}"}} {v[1]:.4f}' for c, v in self.cls.items()]
            return "\n".join(L) + "\n"


class Worker:
    def __init__(self, engine, tokenizer, model_name, wid, ctx_max=8192):
        self.eng, self.tk, self.model, self.wid, self.ctx_max = engine, tokenizer, model_name, wid, ctx_max
        self.st = Stats()
        self.gateway = None
        self.gen_lock = threading.Lock()      # one decode stream per engine: later requests queue here

    # ---- generation ----
    def _gen(self, ids, max_new, stop=(), stop_count=1, text_for_class=""):
        cls = classify(text_for_class)
        with self.st.lock: self.st.waiting += 1
        t0 = time.time()
        with self.gen_lock:
            with self.st.lock: self.st.waiting -= 1; self.st.running += 1; self.st.ctx = len(ids)
            try:
                toks, rec = self.eng.run(ids, max_new, stop, stop_count)
            finally:
                with self.st.lock: self.st.running -= 1; self.st.ctx = 0
        self.st.done(cls, rec, len(toks))
        if self.gateway:
            _post(self.gateway + "/moe_event", {"wid": self.wid, "class": cls, "hit_rate": rec.get("hit_rate", 0.0),
                                                "decode_ms_per_tok": rec.get("decode_ms_per_tok", 0.0)})
        rec["wall_s"] = time.time() - t0
        return toks, rec, cls

    def completions(self, body):
        if "prompt_token_ids" in body or "tokens" in body:
            ids = list(body.get("prompt_token_ids") or body.get("tokens")); text = self._decode(ids)
        else:
            text = body["prompt"]; ids = self._encode(text)
        max_new = int(body.get("max_tokens", body.get("max_new", 16)))
        toks, rec, cls = self._gen(ids, max_new, text_for_class=text)
        return {"object": "text_completion", "model": self.model,
                "choices": [{"index": 0, "text": self._decode(toks), "tokens": toks,
                             "finish_reason": "length" if rec["n"] >= max_new else "stop"}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(toks), "total_tokens": len(ids) + len(toks)},
                "moe_offload": {"class": cls, **rec}}

    def chat(self, body):
        msgs = body["messages"]
        ids = self._chat_ids(msgs)
        max_new = int(body.get("max_tokens", 512))
        user_text = " ".join(m.get("content", "") for m in msgs if m.get("role") == "user")
        toks, rec, cls = self._gen(ids, max_new, stop=(END,), stop_count=2, text_for_class=user_text)
        reasoning, content = split_reasoning(toks)
        return {"object": "chat.completion", "model": self.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": self._decode(content),
                                                     "reasoning_content": self._decode(reasoning)},
                             "finish_reason": "stop" if toks and toks[-1] == END else "length"}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(toks), "total_tokens": len(ids) + len(toks)},
                "moe_offload": {"class": cls, **rec}}

    # ---- tokenizer helpers ----
    def _encode(self, text):
        return self.tk.encode(text, add_special_tokens=False) if self.tk else list(text.encode())[:4096] or [0]

    def _decode(self, ids):
        return self.tk.decode(ids, skip_special_tokens=True) if self.tk else ""

    def _chat_ids(self, msgs):
        if not self.tk: return [1, 2, 3]
        r = self.tk.apply_chat_template(msgs, add_generation_prompt=True)
        return list(r["input_ids"] if hasattr(r, "keys") else r)

    # ---- gateway ----
    def attach(self, gateway, endpoint, chip=0, every=5.0, priors=None):
        self.gateway = gateway.rstrip("/")
        _post(self.gateway + "/register", {"wid": self.wid, "role": "both", "chip": chip, "endpoint": endpoint,
                                           "priors": priors or {}})
        def beat():
            while True:
                time.sleep(every)
                with self.st.lock: inflight = self.st.running + self.st.waiting
                _post(self.gateway + "/heartbeat", {"wid": self.wid, "inflight": inflight})
        threading.Thread(target=beat, daemon=True).start()


def split_reasoning(toks):
    """Solar: <|think|> R <|end|><|begin|>assistant<|content|> C <|end|> -> (R, C) token lists."""
    if not toks or toks[0] != THINK: return [], toks
    try: e = toks.index(END)
    except ValueError: return toks[1:], []
    rest = toks[e + 1:]
    c = rest.index(CONTENT) + 1 if CONTENT in rest else 0
    content = rest[c:]
    if content and content[-1] == END: content = content[:-1]
    return toks[1:e], content


def _post(url, obj, timeout=5):
    try:
        req = urllib.request.Request(url, json.dumps(obj).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read() or b"{}")
    except Exception as e:
        return {"error": str(e)}


def make_handler(w):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        def _send(self, code, obj, ctype="application/json"):
            data = obj.encode() if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            p = self.path.rstrip("/")
            if p == "/metrics": self._send(200, w.st.prometheus(w.model, w.ctx_max), "text/plain; version=0.0.4")
            elif p in ("/health", "/healthz"): self._send(200, {"ok": True, "wid": w.wid})
            elif p == "/v1/models": self._send(200, {"object": "list", "data": [{"id": w.model, "object": "model"}]})
            else: self._send(404, {"error": "not found"})
        def do_POST(self):
            p = self.path.rstrip("/")
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                if p == "/v1/completions": out = w.completions(body)
                elif p == "/v1/chat/completions": out = w.chat(body)
                else: return self._send(404, {"error": "not found"})
                self._send(200, out)
            except Exception as e:
                self._send(400, {"error": f"{type(e).__name__}: {e}"})
    return H


def build(args):
    if args.mock:
        hit = json.loads(args.mock_hit) if args.mock_hit.strip().startswith("{") else float(args.mock_hit)
        return Worker(MockEngine(ms_per_tok=args.mock_ms, hit_rate=hit), None, args.model_name, args.wid)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(args.model)
    env = dict(SOLAR_ENV); env.update(dict(kv.split("=", 1) for kv in args.env))
    eng = EngineProcess(args.engine, args.data, env, log_path=args.log)
    return Worker(eng, tk, args.model_name, args.wid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine"); ap.add_argument("--data"); ap.add_argument("--model")
    ap.add_argument("--model-name", default="solar-open-100b"); ap.add_argument("--wid", default=socket.gethostname() + "-moe0")
    ap.add_argument("--port", type=int, default=8101); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--gateway", default=None, help="gateway URL (default: $LLMD_GATEWAY_URL, else none)"); ap.add_argument("--log", default="engine_serve.log")
    ap.add_argument("--env", nargs="*", default=[], help="extra engine env KEY=VALUE")
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--mock-ms", type=float, default=1.0)
    ap.add_argument("--mock-hit", default="0.85", help="hit rate, or JSON per class for the mock engine")
    ap.add_argument("--priors", default="", help='JSON {"ko":0.89,...}: placement-predicted hit per class')
    a = ap.parse_args()
    if a.gateway is None:
        siteenv.load(); a.gateway = os.environ.get("LLMD_GATEWAY_URL", "")
    w = build(a)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(w))
    if a.gateway: w.attach(a.gateway, f"http://{a.host}:{a.port}", priors=json.loads(a.priors) if a.priors else None)
    print(f"[moe_offload.worker] {w.wid} serving {w.model} on http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
