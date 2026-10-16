"""Build/load the AVX-512 BF16 CPU MoE kernel (cpu_moe.cpp) via torch cpp_extension (cached build)."""
import os
from torch.utils.cpp_extension import load
_here = os.path.dirname(os.path.abspath(__file__))
ext = load(name="cpu_moe_ext", sources=[os.path.join(_here, "cpu_moe.cpp")],
           extra_cflags=["-O3", "-march=sapphirerapids", "-fopenmp"], extra_ldflags=["-fopenmp"],
           build_directory=os.path.join(_here, "..", "cache", "cpu_moe_build"), verbose=False) \
    if os.makedirs(os.path.join(_here, "..", "cache", "cpu_moe_build"), exist_ok=True) is None else None
