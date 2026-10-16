# lm_head split across both chips

- **Date**: 2026-10-16
- **Task**: Solar decode latency
- **Host / device**: etri-ESC8000-E11, 2x RNGD (driver 2026.4, furiosa-opt 0.9.0), Xeon 4410Y x2
- **Commit**: bb532fd

## Question
Splitting the 196608-row head (chip A rows 0..98303, chip B the rest) should save ~0.5 ms/tok.

## Setup
`LM_SPLIT=1`: new kernel `dense/lm_half.rs` (12 chunks); chip A releases its resident DM slots for the head, re-allocates and prefetches layer 0 after.

## Command
```bash
LM_SPLIT=1 PIN_BOUNCE=1 MMIO_VEC=2 MMIO_LIMIT=32768 SQPOLL_OFF=1 DMA_IDLE_US=50 URES=1 LM_ON_B=1 OMP_WAIT_POLICY=passive GOMP_SPINCOUNT=100000 CHIP_CORES=7 CPU_THREADS=17 OMP_PLACES="{7}:17:1" OMP_PROC_BIND=close REPEAT=5 FURIOSA_VISIBLE_CHIPS=0,1 ./target/release/solar_engine data_p2 32
```

## Results
| config | decode ms/tok | layerA ms/tok | lm_head ms/tok |
|---|---:|---:|---:|
| LM_ON_B (baseline) | 28.7 | ~10.3 | 1.6 |
| LM_SPLIT=1 | 48.8 | 28.0 | 1.6 |

## Conclusion
Exact but much slower: every chip-A layer got ~350 µs slower after the head ran there (re-allocated resident slots / program switch suspected, not yet diagnosed). Left off.
