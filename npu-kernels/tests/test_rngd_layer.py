import json
import os
import tempfile

import pytest
from npu_kernels.rngd.layer import (QWEN3_30B_A3B, SOLAR_OPEN_100B, LayerGeometry, generate_layer, geometry_from_hf,
                                    plan_layer)


def test_solar_plan_reproduces_engine_constants():
    """the hand-written Solar-Open engine (track2_opt/solar_engine/src/attn): RW 2048 in 4 q chunks, 512 rows x 32 cols
    per slice (RS = RW / 512 % 2, H / 32), one kv head per 32 slices, 4 kv heads per cluster, Wo in 4 chunks of 1024
    rows with 512 rows per slice (OS = Q / 32)."""
    p = plan_layer(SOLAR_OPEN_100B)
    assert (p.RW, p.Qc, p.A, p.RPS) == (2048, 4, 2, 512)
    assert (p.SPN, p.NPC) == (32, 4)
    assert (p.Hr, p.Hc, p.Ao, p.RPSo) == (1024, 4, 1, 512)
    assert (p.Q, p.P, p.N, p.G, p.D, p.T, p.Cm) == (8192, 1024, 8, 8, 128, 512, 16)


def test_qwen3_plan():
    p = plan_layer(QWEN3_30B_A3B)
    assert (p.RW, p.Qc, p.A, p.RPS) == (1024, 4, 4, 128)       # both clusters, 256 slices per q/k/v chunk
    assert (p.SPN, p.SPH, p.NPC) == (64, 8, 2)
    assert (p.Hr, p.Hc, p.Ao, p.RPSo) == (1024, 2, 2, 256)
    assert p.RR == 16 and p.LF * p.LNC >= 151936 and p.LR % 8 == 0
    ep = p.expert_plan()
    assert (ep.H, ep.F, ep.E, ep.K) == (2048, 768, 64, 8)


def test_geometry_from_hf_config():
    cfg = {"model_type": "qwen3_moe", "hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 4,
           "head_dim": 128, "rms_norm_eps": 1e-6, "rope_theta": 10000000, "num_experts": 128,
           "num_experts_per_tok": 8, "moe_intermediate_size": 768, "vocab_size": 151936, "num_hidden_layers": 48,
           "norm_topk_prob": True}
    assert geometry_from_hf(cfg) == QWEN3_30B_A3B


def test_unsupported_geometries():
    with pytest.raises(ValueError):          # head_dim 64
        plan_layer(LayerGeometry(2048, 32, 4, 64, True, 1e-6, 1e6, 128, 8, 768, 0, 1000, 2))
    with pytest.raises(ValueError):          # hidden does not divide 8192
        plan_layer(LayerGeometry(2880, 64, 8, 128, False, 1e-5, 1e6, 128, 4, 2880, 0, 1000, 2))
    with pytest.raises(ValueError):          # 2 kv heads
        plan_layer(LayerGeometry(2048, 16, 2, 128, True, 1e-6, 1e6, 128, 8, 768, 0, 1000, 2))


def test_generate_crate():
    with tempfile.TemporaryDirectory() as d:
        for g in (QWEN3_30B_A3B, SOLAR_OPEN_100B):
            out = generate_layer(plan_layer(g), os.path.join(d, g.name))
            src = open(os.path.join(out, "src", "attn", "mod.rs")).read()
            assert "${" not in src and "attn_block_c4" in src
            assert ("head_norm" in src) == g.qk_norm
            assert json.load(open(os.path.join(out, "plan.json")))["RW"] == 2 * g.kv_heads * g.head_dim
