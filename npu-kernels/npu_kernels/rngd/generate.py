#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Tiling rule and crate generator for the RNGD expert FFN kernel (one chip, 8 PEs = 2 clusters x 256 slices).

gate/up GEMV (F rows x H cols): the 2 clusters split F in halves; inside a cluster the 256 slices are
  (A row groups) x (H/32 column blocks), A = 8192 / H, so each slice owns R = F / (2A) rows x 32 cols.
down GEMV (H rows x F cols): the 2 clusters split H; inside a cluster the 256 slices are (C row groups of HR rows) x
  (256 / C column blocks of FC cols), FC = F * C / 256. HR is chosen so FC is a multiple of 16 (packet size).
Solar-Open (H=4096, F=1280): A=2, R=320, HR=128, FC=80 - the hand-written kernel in npu-offload.
"""
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from string import Template

_TPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template")


@dataclass(frozen=True)
class TilePlan:
    H: int
    F: int
    E: int          # experts per resident table
    K: int          # experts per launch
    A: int
    R: int
    HR: int
    C: int
    FC: int

    def subs(self):
        d = asdict(self)
        d.update(FH=self.F // 2, HH=self.H // 2, NH32=self.H // 32, R8=self.R // 8, R4=self.R // 4,
                 NFC=self.F // self.FC, HR8=self.HR // 8, FC4=self.FC // 4, FC16=self.FC // 16,
                 NR=self.F // self.R, NHR=self.H // self.HR)
        return {k: str(v) for k, v in d.items()}


def plan(H: int, F: int, E: int = 32, K: int = 8) -> TilePlan:
    if H % 32 or 8192 % H:
        raise ValueError(f"H={H}: needs H | 8192 and H % 32 == 0 (gate/up slices are 32 columns wide)")
    A = 8192 // H
    if F % (2 * A) or (F // (2 * A)) % 8:
        raise ValueError(f"F={F}: rows per slice F/(2*{A}) must be a multiple of 8")
    R = F // (2 * A)
    for HR in (128, 64, 256, 32):
        if (H // 2) % HR: continue
        C = (H // 2) // HR
        if 256 % C: continue
        nfc = 256 // C
        if F % nfc: continue
        FC = F // nfc
        if FC % 16 == 0 and HR % 8 == 0:
            return TilePlan(H, F, E, K, A, R, HR, C, FC)
    raise ValueError(f"H={H} F={F}: no down-proj split with 16-aligned column blocks")


def _slot_block(p: TilePlan, k: int) -> str:
    s = p.subs()
    wg = f"m![1 # {p.E}, F / {p.R}, H / 32, F / 8 % {s['R8']}, H / 4 % 8, F % 8, H % 4]"
    wd = f"m![1 # {p.E}, H / {p.HR}, F / {p.FC}, H / 8 % {s['HR8']}, F / 4 % {s['FC4']}, H % 8, F % 4]"
    return f"""    if {k}usize < active {{
        let e = ((packed / {64 ** k}u64) % 64) as usize;
        let wg = wg_t.view().tile::<m![E], 1, {wg}>(e);
        let wu = wu_t.view().tile::<m![E], 1, {wg}>(e);
        let wd = wd_t.view().tile::<m![E], 1, {wd}>(e);
        let o = out.view_mut().tile::<m![K], 1, m![1 #{{!}} {p.K}, H]>({k});
        expert(device, &x_trf, wg, wu, wd, o);
    }}"""


def crate_name(p: TilePlan) -> str:
    """distinct per plan, so plans built in a shared target directory never reuse each other's artifacts."""
    return f"rngd_expert_h{p.H}_f{p.F}_e{p.E}_k{p.K}"


def generate(p: TilePlan, out_dir: str, crate: str = None) -> str:
    """write the crate for plan `p` into out_dir (overwritten)."""
    crate = crate or crate_name(p)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    subs = p.subs(); subs["CRATE"] = crate
    subs["SLOTS"] = "\n".join(_slot_block(p, k) for k in range(p.K))
    for root, _, files in os.walk(_TPL):
        for f in files:
            src = os.path.join(root, f)
            rel = os.path.relpath(src, _TPL)
            dst = os.path.join(out_dir, rel[:-4] if rel.endswith(".tpl") else rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            text = open(src).read()
            open(dst, "w").write(Template(text).substitute(subs) if src.endswith(".tpl") else text)
    return out_dir


def check(crate_dir: str, timeout: int = 3600) -> str:
    """build and run the CPU-emulation correctness check; returns its output (raises on failure)."""
    r = subprocess.run(["cargo", "run", "--release", "--bin", "eval", "--", "check"], cwd=crate_dir,
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"rngd kernel check failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r.stdout
