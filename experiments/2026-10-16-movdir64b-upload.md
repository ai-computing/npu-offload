# MOVDIR64B uploads through the BAR4 window

- **Date**: 2026-10-16
- **Task**: Solar decode latency
- **Host / device**: etri-ESC8000-E11, 2x RNGD (driver 2026.4, furiosa-opt 0.9.0), Xeon 4410Y x2
- **Commit**: bb532fd (measured before the repo existed; code identical)

## Question
Can host->device uploads skip the ~25 µs DMA setup by writing through the BAR4 window with wide stores?

## Setup
Runtime posted writes <= 2 KiB through BAR4 with 8-byte volatile stores; larger writes used DMA. Added 64-byte stores
(AVX-512 `MMIO_VEC=1` or MOVDIR64B `MMIO_VEC=2`) and `MMIO_LIMIT` (window used up to N bytes).

## Command
```bash
MMIO_VEC={0,1,2} MMIO_LIMIT=32768 SQPOLL_OFF=1 DMA_IDLE_US=50 URES=1 LM_ON_B=1 ... REPEAT=3 ./target/release/solar_engine data_p2 32
```

## Results
| config | decode ms/tok | chip A upload µs | chip B upload µs | tokens |
|---|---:|---:|---:|---|
| stock (8-byte, 2 KiB) | 31.6 | 39 | 77 | ref |
| AVX-512 64B, 32 KiB | 31.5 | 56 | 61 | same |
| MOVDIR64B, 32 KiB | 29.8 | 27 | 50 | same |

## Conclusion
MOVDIR64B wins (UC memory splits plain AVX-512 stores). Adopted as default config.
