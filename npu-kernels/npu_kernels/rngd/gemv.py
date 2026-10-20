#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Generic weight-streaming GEMV for RNGD, y[ROWS] = W[ROWS, COLS] @ x[COLS] (bf16), generated per shape.

The attention projections (q, k/v, o), the MoE router and the LM head of a layer engine are all GEMVs of this form;
the tiling is the expert kernel's gate/up rule: the 256 slices of a cluster are (A = 8192 / COLS row groups) x
(COLS / 32 column blocks), the two clusters split the rows of a chunk, each slice holds R = F / (2A) rows x 32
columns. Rows are streamed in chunks of F rows sized to the data memory (rows padded to a multiple of F).
"""
import os
import shutil
from dataclasses import asdict, dataclass
from string import Template

_TPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemv_template")
_CHUNK_BYTES = 16 << 20          # weight bytes per chunk (the layer engine streams ~16.8 MB chunks)


@dataclass(frozen=True)
class GemvPlan:
    ROWS: int
    H: int                       # columns
    F: int                       # rows per chunk
    NC: int                      # chunks
    A: int
    R: int

    @property
    def ROWS_P(self):
        return self.F * self.NC

    def subs(self):
        d = asdict(self)
        d.update(ROWS_P=self.ROWS_P, FH=self.F // 2, NH32=self.H // 32, R8=self.R // 8)
        return {k: str(v) for k, v in d.items()}


def plan_gemv(rows: int, cols: int, chunk_bytes: int = _CHUNK_BYTES) -> GemvPlan:
    if cols % 32 or 8192 % cols:
        raise ValueError(f"cols={cols}: needs cols | 8192 and cols % 32 == 0")
    A = 8192 // cols
    unit = 2 * A * 8                                  # rows per chunk must give R % 8 == 0
    f_max = max(unit, (chunk_bytes // (2 * cols)) // unit * unit)
    rows_u = -(-rows // unit) * unit
    nc = -(-rows_u // f_max)
    F = -(-rows_u // nc // unit) * unit               # balance chunks
    return GemvPlan(rows, cols, F, nc, A, F // (2 * A))


def crate_name(p: GemvPlan) -> str:
    return f"rngd_gemv_r{p.ROWS}_c{p.H}"


def _chunk_block(p: GemvPlan, c: int) -> str:
    s = p.subs()
    wt = f"m![1 # {p.NC}, F / {p.R}, H / 32, F / 8 % {s['R8']}, H / 4 % 8, F % 8, H % 4]"
    return f"""    {{
        let w: DmTensor<bf16, Chip, GC, GS, GElem> = w.view().tile::<m![NC], 1, {wt}>({c}).to_dm(&mut device.tdma);
        let y = gemv_g(device, &w, &x_trf);
        y.view().to_hbm_view(&mut device.tdma, out.view_mut().tile::<m![NC], 1, m![1 #{{!}} {p.NC}, F]>({c}));
    }}"""


def generate_gemv(p: GemvPlan, out_dir: str, crate: str = None) -> str:
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    subs = p.subs()
    subs["CRATE"] = crate or crate_name(p)
    subs["CHUNKS"] = "\n".join(_chunk_block(p, c) for c in range(p.NC))
    for root, _, files in os.walk(_TPL):
        for f in files:
            src = os.path.join(root, f)
            rel = os.path.relpath(src, _TPL)
            dst = os.path.join(out_dir, rel[:-4] if rel.endswith(".tpl") else rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            text = open(src).read()
            open(dst, "w").write(Template(text).substitute(subs) if src.endswith(".tpl") else text)
    return out_dir
