"""P0 expert-activation profile for Solar-Open-100B (bf16, CPU, HF transformers).

Records per-layer top-k expert ids for every prefill and decode token over a diverse prompt set,
then (analyze) computes held-out hit rate vs HBM expert budget for a frequency-ranked static
placement (PolyStrata-style), and expert-union growth vs batch size.

  python3 expert_profile.py record --n-new 48 --out results/routes.pt
  python3 expert_profile.py analyze --routes results/routes.pt
"""
import argparse, json, os, sys, time
import torch

MODEL = "upstage/Solar-Open-100B"
PROMPTS = [
    # (domain, text)
    ("ko_chat", "서울에서 주말에 가볼 만한 곳을 다섯 군데 추천하고 각각 이유를 설명해줘."),
    ("ko_tech", "NPU와 GPU의 메모리 계층 구조 차이를 대학원생 수준으로 자세히 설명해줘."),
    ("ko_write", "봄을 주제로 짧은 시를 한 편 쓰고, 그 시의 표현 기법을 분석해줘."),
    ("en_chat", "What are some good strategies for staying productive when working from home?"),
    ("en_sci", "Explain how mixture-of-experts language models route tokens and why load balancing matters."),
    ("en_hist", "Summarize the causes and consequences of the fall of the Western Roman Empire."),
    ("code_py", "Write a Python function that parses a CSV file and returns the top 5 rows sorted by a given column, with error handling."),
    ("code_cpp", "Implement a thread-safe LRU cache in C++17 and explain the design choices."),
    ("math", "Solve step by step: a train travels 300 km at 80 km/h and returns at 120 km/h. What is the average speed?"),
    ("ja", "日本の四季について、それぞれの季節の特徴と代表的な行事を説明してください。"),
    ("ko_code", "파이썬으로 이진 탐색 트리를 구현하고 삽입, 삭제, 탐색 연산의 시간 복잡도를 설명해줘."),
    ("en_biz", "Draft a short email to a client explaining a two-week delay in a software project delivery."),
]


def record(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers.models.solar_open.modeling_solar_open as M
    torch.set_num_threads(a.threads)
    tk = AutoTokenizer.from_pretrained(MODEL)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu").eval()
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    # assign layer index to each MoE module
    moes = [m for m in model.modules() if isinstance(m, M.SolarOpenMoE)]
    assert len(moes) == model.config.num_hidden_layers, len(moes)
    buf = []  # filled per forward: list of [T, k] per layer in order
    orig = M.SolarOpenMoE.route_tokens_to_experts

    def hooked(self, router_logits):
        idx, w = orig(self, router_logits)
        buf.append(idx.detach().to(torch.int16).cpu())
        return idx, w
    M.SolarOpenMoE.route_tokens_to_experts = hooked

    recs = []
    for pi, (dom, text) in enumerate(PROMPTS):
        ids = tk.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                     return_tensors="pt")
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        P = ids.shape[1]
        buf.clear(); t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=a.n_new, do_sample=False)
        dt = time.time() - t0
        L = len(moes)
        # buf: step0 = prefill (L entries of [P,k]), then each decode step L entries of [1,k]
        steps = len(buf) // L
        pre = torch.stack([buf[l] for l in range(L)], 0)                       # [L,P,k]
        dec = torch.stack([torch.cat([buf[s * L + l] for s in range(1, steps)], 0)
                           for l in range(L)], 0) if steps > 1 else None      # [L,D,k]
        gen = tk.decode(out[0, P:], skip_special_tokens=True)
        recs.append(dict(domain=dom, prompt_len=P, prefill=pre, decode=dec, text=gen))
        print(f"[{pi}] {dom} P={P} D={0 if dec is None else dec.shape[1]} {dt:.0f}s :: {gen[:80]!r}", flush=True)
        torch.save(dict(model=MODEL, n_expert=model.config.n_routed_experts, recs=recs), a.out)
    print("saved", a.out)


def analyze(a):
    d = torch.load(a.routes)
    E = d["n_expert"]; recs = d["recs"]
    L = recs[0]["prefill"].shape[0]
    total = L * E
    # 2-fold: profile on even prompts, eval decode tokens of odd prompts, and vice versa
    def counts(rs, use_prefill=True, use_decode=True):
        c = torch.zeros(L, E)
        for r in rs:
            for key, on in (("prefill", use_prefill), ("decode", use_decode)):
                x = r[key]
                if on and x is not None:
                    for l in range(L):
                        c[l] += torch.bincount(x[l].flatten().long(), minlength=E).float()
        return c
    budgets = [0.1, 0.2, 0.3, 0.39, 0.5, 0.6, 0.75, 1.0]
    out = []
    for prof_src in ("prefill+decode", "prefill_only"):
        res = {b: [0, 0, 0, 0] for b in budgets}  # hits, total, tokens_allhit, tokens
        for fold in (0, 1):
            tr = [r for i, r in enumerate(recs) if i % 2 == fold]
            te = [r for i, r in enumerate(recs) if i % 2 != fold]
            c = counts(tr, True, prof_src == "prefill+decode")
            order = torch.argsort(c.flatten(), descending=True)   # global knapsack (equal sizes)
            for b in budgets:
                n = int(b * total)
                res_mask = torch.zeros(L * E, dtype=torch.bool); res_mask[order[:n]] = True
                res_mask = res_mask.view(L, E)
                for r in te:
                    x = r["decode"]
                    if x is None: continue
                    hit = torch.stack([res_mask[l][x[l].long()] for l in range(L)])  # [L,D,k]
                    res[b][0] += hit.sum().item(); res[b][1] += hit.numel()
                    res[b][2] += hit.all(-1).sum().item(); res[b][3] += hit.shape[0] * hit.shape[1]
        for b in budgets:
            h, t, ah, tt = res[b]
            out.append(dict(profile=prof_src, budget=b, hit_rate=h / t, layer_all8_hit=ah / tt))
    # uniform baseline: budget b -> expected hit b
    # union growth: distinct experts per layer touched by a batch of B decode tokens (random mix)
    allde = torch.cat([r["decode"] for r in recs if r["decode"] is not None], 1)  # [L,N,k]
    N = allde.shape[1]; g = torch.Generator().manual_seed(0)
    union = []
    for B in [1, 2, 4, 8, 16, 32, 64, 128]:
        if B > N: break
        us = []
        for _ in range(20):
            sel = torch.randperm(N, generator=g)[:B]
            us.append(sum(len(torch.unique(allde[l, sel])) for l in range(L)) / L)
        union.append(dict(batch=B, mean_distinct_experts_per_layer=sum(us) / len(us)))
    # skew: share of activations covered by top-x% experts (all data)
    c = counts(recs).flatten(); cs = torch.sort(c, descending=True)[0].cumsum(0) / c.sum()
    skew = {f"top{int(p*100)}%": cs[int(p * len(cs)) - 1].item() for p in (0.1, 0.2, 0.39, 0.5)}
    report = dict(n_prompts=len(recs), n_decode_tokens=N, hit=out, union=union, skew=skew)
    print(json.dumps(report, indent=1))
    with open(a.routes.replace(".pt", "_analysis.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("record"); r.add_argument("--n-new", type=int, default=48)
    r.add_argument("--threads", type=int, default=24)
    r.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "routes.pt"))
    z = sp.add_parser("analyze"); z.add_argument("--routes", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "routes.pt"))
    a = ap.parse_args()
    record(a) if a.cmd == "record" else analyze(a)
