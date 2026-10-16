"""moe-offload CLI:  plan | budget | worker | gateway | selftest"""
import argparse, json, sys


def _plan(a):
    from .spec import ModelSpec
    from .profile import counts_from_routes, counts_by_class
    from .placement import plan, hit_rates, class_priors, save
    m = ModelSpec.from_hf_config(a.model)
    c = counts_from_routes(a.profile, m.layers, m.experts)
    p = plan(c, [int(s) for s in a.slots.split(",")], a.policy)
    save(p, a.out)
    print(json.dumps({"out": a.out, "tier_fractions": [round(x, 4) for x in hit_rates(c, p)],
                      "class_priors": class_priors(p, counts_by_class(a.profile, m.layers, m.experts))}))


def _budget(a):
    from .spec import ModelSpec
    from . import budget
    m = ModelSpec.from_hf_config(a.model)
    rows, host = budget.report(m, budget.rngd_solar_2chip(lm_on_b=not a.lm_on_a), a.ctx)
    for r in rows: print(json.dumps(r))
    print(json.dumps({"host_expert_gib": round(host / budget.GiB, 1)}))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="moe-offload")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="tiered expert placement from a routing profile")
    p.add_argument("--model", required=True); p.add_argument("--profile", required=True)
    p.add_argument("--slots", default="24,32"); p.add_argument("--policy", default="mix:4"); p.add_argument("--out", required=True)
    b = sub.add_parser("budget", help="HBM accounting per accelerator (Solar layout on 2x RNGD)")
    b.add_argument("--model", required=True); b.add_argument("--ctx", type=int, default=8192)
    b.add_argument("--lm-on-a", action="store_true")
    sub.add_parser("worker", help="OpenAI-compatible worker (args: see moe_offload.worker)", add_help=False)
    sub.add_parser("gateway", help="expert-affinity gateway (args: see moe_offload.llmd.gateway)", add_help=False)
    sub.add_parser("selftest", help="NPU-free regression tests")
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("worker", "gateway"):
        sys.argv = [f"moe-offload {argv[0]}"] + argv[1:]
        if argv[0] == "worker":
            from .worker import main as m
        else:
            from .llmd.gateway import main as m
        return m()
    a = ap.parse_args(argv)
    if a.cmd == "plan": _plan(a)
    elif a.cmd == "budget": _budget(a)
    elif a.cmd == "selftest":
        from .tests import run
        tests = [v for k, v in sorted(vars(run).items()) if k.startswith("test_")]
        for t in tests: t(); print("PASS", t.__name__)
        print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
