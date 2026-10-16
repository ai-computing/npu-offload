"""moe_offload — control plane for MoE expert offloading across accelerator HBM and host DRAM.

Model-agnostic pieces (spec, profile, placement, budget, worker, llm-d integration) live here; the
model-specific data plane is a separate engine binary (e.g. track2_opt/solar_engine) driven through
`engine.EngineProcess`.
"""
__all__ = ["spec", "profile", "placement", "engine", "worker"]
