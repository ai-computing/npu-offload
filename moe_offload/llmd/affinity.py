"""Expert-affinity routing signal — the MoE analogue of llm-d's prefix-cache-aware scorer.

An offloading worker is fast when the request's routed experts are resident in its accelerators (tier hit rate).
Different workers can carry different placements (e.g. one planned from a Korean profile, one from code), so the
gateway should send each request where its experts are likely resident. The gateway cannot run the router, so it
scores with a per-(worker, request-class) EWMA of the hit rate the workers report after each request, seeded by the
hit rate the worker's placement predicts for that class's profile.
"""
import re, threading

CLASSES = ("ko", "code", "en")
_HANGUL = re.compile(r"[가-힣㄰-㆏]")
_CODE = re.compile(r"```|\bdef \w+\(|\bclass \w+|#include|;\s*$|\bfunction\b|=>|\bimport \w+", re.M)


def classify(text):
    if not text: return "en"
    if len(_HANGUL.findall(text)) / max(1, len(text)) > 0.15: return "ko"
    if _CODE.search(text): return "code"
    return "en"


class AffinityTable:
    """hit-rate EWMA per (worker, class); unknown pairs fall back to the worker's prior, then to `default`."""
    def __init__(self, alpha=0.3, default=0.8):
        self.a, self.default = alpha, default
        self.h, self.prior = {}, {}
        self.lock = threading.Lock()

    def set_prior(self, wid, cls, hit):
        with self.lock: self.prior[(wid, cls)] = hit

    def observe(self, wid, cls, hit):
        with self.lock:
            k = (wid, cls)
            old = self.h.get(k, self.prior.get(k, self.default))
            self.h[k] = (1 - self.a) * old + self.a * hit

    def get(self, wid, cls):
        with self.lock:
            return self.h.get((wid, cls), self.prior.get((wid, cls), self.default))


def expected_ms_per_tok(hit, ms_hit_floor, ms_per_miss_rate):
    """crude latency model: decode time grows linearly with the cold (host) fraction."""
    return ms_hit_floor + ms_per_miss_rate * (1.0 - hit)


def score(hit, inflight, kv_hit_blocks=0, w_hit=1.0, w_load=0.15, w_kv=0.02):
    """higher is better: expert residency, prefix-cache reuse, minus queueing."""
    return w_hit * hit + w_kv * kv_hit_blocks - w_load * inflight
