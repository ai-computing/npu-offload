"""NPU-free regression tests:  python3 -m moe_offload.tests.run"""
import json, threading, time, urllib.request
from http.server import ThreadingHTTPServer

from moe_offload import placement, budget, airm
from moe_offload.spec import ModelSpec
from moe_offload.worker import Worker, make_handler, split_reasoning
from moe_offload.engine import MockEngine
from moe_offload.llmd import gateway as gw
from moe_offload.llmd.affinity import classify

class Skip(Exception):
    """Raised by a test that cannot run here; reported as SKIP, not as a failure."""


SOLAR = ModelSpec("solar_open", 4096, 48, 64, 8, 128, 128, 8, 1280, 1, 196608)


def test_placement():
    counts = [[float(128 - e) for e in range(128)] for _ in range(3)]       # rank == expert id
    p = placement.plan(counts, [24, 32], "mix:4")
    assert p[0][0] == [0, 1, 2, 3] + list(range(36, 56)) and p[0][1] == list(range(4, 36))
    p = placement.plan(counts, [24, 32], "top")
    assert p[0][0] == list(range(24)) and p[0][1] == list(range(24, 56))
    h = placement.hit_rates(counts, p)
    tot = 3 * sum(range(1, 129))
    assert abs(h[0] - 3 * sum(range(105, 129)) / tot) < 1e-9 and abs(h[1] - 3 * sum(range(73, 105)) / tot) < 1e-9
    assert abs(sum(h) - 1) < 1e-9
    pri = placement.class_priors(p, {"en": counts})
    assert abs(pri["en"] - (h[0] + h[1])) < 1e-3


def test_budget():
    rows, host = budget.report(SOLAR, budget.rngd_solar_2chip())
    assert all(r["fits"] for r in rows) and rows[0]["max_slots"] >= 24 and rows[1]["max_slots"] >= 32
    assert host > 0


def test_reasoning_split():
    r, c = split_reasoning([22, 5, 6, 21, 20, 999, 23, 7, 8, 21])
    assert r == [5, 6] and c == [7, 8]
    assert split_reasoning([7, 8]) == ([], [7, 8])


def test_classify():
    assert classify("김치찌개 끓이는 법 알려줘") == "ko"
    assert classify("def foo(x):\n    return x") == "code"
    assert classify("Summarize the French revolution") == "en"


def _post(url, obj):
    req = urllib.request.Request(url, json.dumps(obj).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def test_gateway_affinity():
    g = gw.Gateway()
    gsrv = gw.serve(g, port=0)
    gurl = f"http://127.0.0.1:{gsrv.server_address[1]}"
    hits = {"wA": {"ko": 0.95, "en": 0.70, "code": 0.72}, "wB": {"ko": 0.70, "en": 0.93, "code": 0.90}}
    srvs = []
    for wid, h in hits.items():
        w = Worker(MockEngine(ms_per_tok=0.2, hit_rate=h), None, "mock", wid)
        s = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(w)); threading.Thread(target=s.serve_forever, daemon=True).start()
        w.attach(gurl, f"http://127.0.0.1:{s.server_address[1]}", every=0.5)
        srvs.append(s)
    ko, en = "한국어로 김치찌개 끓이는 법 알려줘", "Explain how hash tables handle collisions"
    for _ in range(4):                      # warm-up: unknown affinities -> load/tie order explores both
        _post(gurl + "/v1/completions", {"prompt": ko, "max_tokens": 2})
        _post(gurl + "/v1/completions", {"prompt": en, "max_tokens": 2})
    # make sure each worker has been observed on each class (the EWMA learns from reports)
    for wid in hits:
        for cls, h in hits[wid].items(): g.aff.observe(wid, cls, h)
    r_ko = [_post(gurl + "/v1/completions", {"prompt": ko, "max_tokens": 2})["routing"]["worker"] for _ in range(5)]
    r_en = [_post(gurl + "/v1/completions", {"prompt": en, "max_tokens": 2})["routing"]["worker"] for _ in range(5)]
    assert r_ko.count("wA") == 5, r_ko
    assert r_en.count("wB") == 5, r_en
    m = urllib.request.urlopen(gurl + "/metrics").read().decode()
    assert 'moe_gateway_routed_total{worker="wA",class="ko"}' in m
    # priors on register seed the table before any traffic
    g.register({"wid": "wC", "endpoint": "http://127.0.0.1:1", "priors": {"code": 0.99}})
    assert g.aff.get("wC", "code") == 0.99
    for s in srvs: s.shutdown()
    gsrv.shutdown()


def test_airm_runner():
    from rm_core import ContractError, Workload, registry
    if "npu-offload" not in registry.available("airm.workloads"):
        raise Skip("airm.workloads:npu-offload entry point not registered; run `pip install -e .`")
    run = registry.load("airm.workloads", "npu-offload")
    tiers = airm.memory_tiers()
    assert tiers[-1].kind == "host-dram" and all(t.capacity_bytes > 0 for t in tiers)
    for bad in (Workload(kind="train", model="m"), Workload(kind="decode", model="m")):
        try: run(bad, mock=True, kv_transport="p2p" if bad.kind == "decode" else "")
        except ContractError: continue
        raise AssertionError(f"runner accepted {bad}")
    ms = run(Workload(kind="decode", model="mock"), mock=True, mock_ms=2.0, prompt=[1, 2, 3], max_new=4, repeat=3)
    assert {m.name for m in ms} == {"tok_s", "tpot_p50_ms", "tpot_p90_ms"} and abs(ms[0].value - 500) < 1e-6
    h = run(Workload(kind="serve", model="mock"), mock=True, mock_ms=0.2, port=0, metrics_port=0, gateway="", block=False)
    url = f"http://127.0.0.1:{h.servers[0].server_address[1]}"
    _post(url + "/v1/completions", {"prompt": "hello", "max_tokens": 3})
    m = urllib.request.urlopen(f"http://127.0.0.1:{h.servers[1].server_address[1]}/metrics").read().decode()
    for k in ("airm_tok_s{", "airm_tpot_p50_ms{", "airm_tpot_p90_ms{", 'moe_offload_memory_tier_capacity_bytes{tier="host-dram"'):
        assert k in m, (k, m)
    for s in h.servers: s.shutdown()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    skipped = 0
    for t in tests:
        t0 = time.time()
        try: t()
        except Skip as e:
            skipped += 1; print(f"SKIP {t.__name__}: {e}"); continue
        print(f"PASS {t.__name__} ({time.time() - t0:.2f}s)")
    print(f"{len(tests) - skipped}/{len(tests)} passed" + (f", {skipped} skipped" if skipped else ""))
