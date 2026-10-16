"""Client for track2_opt/expert_srv (furiosa-opt hot-expert server on its own RNGD chip).

The server keeps E=32 hot experts per layer resident and runs the hand-written ffn_k8u kernel
(runtime expert ids, ~250 us for 8 experts) per request. We talk over a shared-memory mailbox;
the client spins with sleep(0) so it releases the GIL while waiting.
"""
import mmap, os, struct, subprocess, time
import numpy as np
import torch

SRV_BIN = os.path.expanduser("~/workspace/npu_offload/track2_opt/expert_srv/target/release/expert_srv")
E_SRV, K, H = 32, 8, 4096
OFF_X, OFF_OUT = 64, 16384
SHM_BYTES = OFF_OUT + K * H * 2 + 4096


def export_tables(hx, assign, n_layers, wdir):
    """hx[l][e] = (gu [2F,H], d [H,F]) bf16. assign[l] = list of E_SRV expert ids (slot order).
    Writes wdir/layer_<l>.bin = per slot [Wg][Wu][Wd] raw row-major bf16."""
    os.makedirs(wdir, exist_ok=True)
    for l in range(n_layers):
        p = os.path.join(wdir, f"layer_{l}.bin")
        parts = []
        for e in assign[l]:
            gu, d = hx[l][e]
            parts.append(gu.contiguous().view(torch.int16).numpy().ravel())   # Wg rows then Wu rows
            parts.append(d.contiguous().view(torch.int16).numpy().ravel())
        np.concatenate(parts).tofile(p)


class OptExpertServer:
    def __init__(self, wdir, n_layers, chip=0, shm_path="/dev/shm/solar_expert_srv", log=None):
        self.shm_path = shm_path
        with open(shm_path, "wb") as f:
            f.write(b"\0" * SHM_BYTES)
        self.f = open(shm_path, "r+b")
        self.mm = mmap.mmap(self.f.fileno(), SHM_BYTES)
        self.buf = memoryview(self.mm)
        self.x = np.frombuffer(self.mm, dtype=np.uint16, count=H, offset=OFF_X)
        self.out = np.frombuffer(self.mm, dtype=np.uint16, count=K * H, offset=OFF_OUT)
        env = dict(os.environ, FURIOSA_OPT_NPUS=str(chip), FURIOSA_VISIBLE_CHIPS=str(chip))
        self.proc = subprocess.Popen([SRV_BIN, shm_path, wdir, str(n_layers)], env=env,
                                     stdout=log or subprocess.DEVNULL, stderr=log or subprocess.STDOUT)
        self.seq = 0
        t0 = time.time()
        while struct.unpack_from("<I", self.mm, 36)[0] != 1:
            if self.proc.poll() is not None:
                raise RuntimeError(f"expert_srv exited rc={self.proc.returncode}")
            time.sleep(0.2)
        self.t_load = time.time() - t0
        self.calls = 0; self.t_wait = 0.0

    def ffn(self, layer, x_bf16, slots):
        """x_bf16: [H] torch bf16; slots: list of <=8 local slot ids. Returns [len(slots), H] fp32 (unweighted)."""
        n = len(slots)
        ids = list(slots) + [slots[0]] * (K - n)        # unguarded kernel always runs 8
        packed = K << 56
        for k, e in enumerate(ids):
            packed += e << (6 * k)
        self.x[:] = x_bf16.contiguous().view(torch.int16).numpy().view(np.uint16)
        struct.pack_into("<IIQ", self.mm, 16, 1, layer, packed)
        self.seq += 1
        struct.pack_into("<Q", self.mm, 0, self.seq)    # publish (x86 TSO keeps store order)
        t0 = time.perf_counter()
        while struct.unpack_from("<Q", self.mm, 8)[0] != self.seq:
            time.sleep(0)                               # yields GIL
        self.t_wait += time.perf_counter() - t0; self.calls += 1
        st = struct.unpack_from("<I", self.mm, 32)[0]
        if st != 0:
            raise RuntimeError(f"expert_srv status {st}")
        o = torch.from_numpy(self.out.copy().view(np.int16)).view(torch.bfloat16).view(K, H)
        return o[:n].float()

    def last_us(self):
        return struct.unpack_from("<d", self.mm, 40)[0]

    def close(self):
        try:
            struct.pack_into("<IIQ", self.mm, 16, 3, 0, 0)
            self.seq += 1
            struct.pack_into("<Q", self.mm, 0, self.seq)
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
