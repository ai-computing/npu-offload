"""FuriosaAI RNGD kernels generated per shape (furiosa-opt 0.9).

  plan(H, F) / generate / check     routed-expert FFN kernel + CPU-emulation check + expert server (template/)
  plan_gemv(rows, cols) / generate_gemv   weight-streaming GEMV (attention projections, router, LM head)
  build_server(plan, emulate)       build the expert server binary (`cargo furiosa-opt build` or emulator)

A new shape is supported by computing a plan, not by hand-editing kernels. The expert server speaks the stdin/stdout
protocol documented in template/src/bin/server.rs.tpl (LOAD / RUN / UPDATE / RUN_WSUM).
"""
from .build import build_server, cache_root  # noqa: F401
from .gemv import GemvPlan, generate_gemv, plan_gemv  # noqa: F401
from .generate import TilePlan, check, crate_name, generate, plan  # noqa: F401
