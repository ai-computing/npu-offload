import pytest
from npu_kernels.rngd import plan, plan_gemv


def test_expert_plans():
    assert (lambda p: (p.A, p.R, p.HR, p.FC))(plan(4096, 1280)) == (2, 320, 128, 80)     # Solar-Open
    assert (lambda p: (p.A, p.R, p.HR, p.FC))(plan(2048, 768)) == (4, 96, 64, 48)        # Qwen3-30B-A3B
    for H, F in ((2880, 2880), (4096, 100), (256, 512)):
        with pytest.raises(ValueError):
            plan(H, F)


def test_gemv_plans():
    p = plan_gemv(8192, 4096)                         # Solar q projection = the layer engine's constants
    assert (p.F, p.NC, p.A, p.R) == (2048, 4, 2, 512)
    assert plan_gemv(1024, 2048).R == 128 and plan_gemv(128, 2048).R == 16
    p = plan_gemv(151936, 2048)                       # Qwen3 LM head: padded, chunked
    assert p.ROWS_P >= 151936 and p.R % 8 == 0
