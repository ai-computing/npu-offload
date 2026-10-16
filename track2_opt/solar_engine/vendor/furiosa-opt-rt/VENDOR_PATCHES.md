# Local modifications to furiosa-opt-rt 0.9.0 (Apache-2.0)

Copied from crates.io `furiosa-opt-rt-0.9.0` and changed for the Solar offload engine. Restore
`prebuilt/` (firmware and bootloader images, not tracked here) from the crates.io package:
`cp ~/.cargo/registry/src/*/furiosa-opt-rt-0.9.0/prebuilt/* prebuilt/`.

All changes are opt-in through environment variables; with none set the runtime behaves as upstream.

| env | file | change |
|---|---|---|
| `SQPOLL_OFF=1` | device/dma | io_uring without SQPOLL |
| `DMA_IDLE_US` | device/dma | DMA worker idle spin before parking |
| `MMIO_VEC=1/2`, `MMIO_LIMIT` | device/engine/mmio.rs, mod.rs | BAR4 window writes with AVX-512 or MOVDIR64B 64-byte stores; window used up to `MMIO_LIMIT` bytes (upstream: 8-byte stores, 2 KiB) |
| `MMIO_RLIMIT` + `MMIO_READ_HOOK` | lib.rs, device/engine/mod.rs | small reads served by a caller-registered hook reading the whole-DRAM BAR mapping |
| `PIN_BOUNCE=1` | device/engine/mod.rs | small DMA reads go through a per-thread page-locked bounce buffer |
