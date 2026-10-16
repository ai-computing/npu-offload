# Page-locked bounce buffer for small DMA reads

- **Date**: 2026-10-16
- **Task**: Solar decode latency
- **Host / device**: etri-ESC8000-E11, 2x RNGD (driver 2026.4, furiosa-opt 0.9.0), Xeon 4410Y x2
- **Commit**: bb532fd

## Question
Does pinning (dma_heap, now world-writable) cut DMA read latency for 8 KB downloads?

## Setup
`PIN_BOUNCE=1`: reads <= 64 KiB DMA into a per-thread `Pinned` buffer, then memcpy out (no per-call page pinning).

## Command
```bash
PIN_BOUNCE=1 MMIO_VEC=2 MMIO_LIMIT=32768 SQPOLL_OFF=1 DMA_IDLE_US=50 URES=1 LM_ON_B=1 OMP_WAIT_POLICY=passive GOMP_SPINCOUNT=100000 CHIP_CORES=7 CPU_THREADS=17 OMP_PLACES="{7}:17:1" OMP_PROC_BIND=close REPEAT=5 FURIOSA_VISIBLE_CHIPS=0,1 ./target/release/solar_engine data_p2 32   # with and without PIN_BOUNCE=1, interleaved, load < 3
```

## Results
| config | decode ms/tok | rounds | L.dl µs | B.dl µs |
|---|---:|---|---:|---:|
| PIN_BOUNCE=1 (clean run) | **28.7** | 29.1 28.4 28.3 28.7 29.1 | 30 | 41 |
| baseline (cleanest run) | 31.3 | 33.9 31.3 33.2 30.5 30.6 | 53 | 62 |

Other runs in the batch were contaminated (my own cargo build, external root python using 15-22 cores). Tokens identical.

## Conclusion
~1.5-2.5 ms/tok gain, stable rounds; adopted (34.9 tok/s). Needs a cleaner repeat to pin down the delta.
