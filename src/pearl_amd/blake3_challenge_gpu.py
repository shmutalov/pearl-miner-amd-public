"""OpenCL solver for the pearl/v1 stratum ``pearl.challenge`` BLAKE3 PoW.

The kernel itself lives in ``src/kernels/blake3_challenge.cl``. Each work-item
hashes one nonce; the host launches batches of (typically) 16-64M nonces at a
time until a winner appears. On an RX 570 we expect well into the hundreds of
megahashes per second, so difficulty=32 (expected 2^32 hashes) finishes in
seconds rather than the ~5 minutes a 12-core CPU search takes.
"""
from __future__ import annotations

import os
import struct
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pyopencl as cl

from .device import find_gpu


_KERNEL_REL = Path(__file__).resolve().parent.parent / "kernels" / "blake3_challenge.cl"


class GpuChallengeSolver:
    """Reusable PyOpenCL context + kernel for the pearl.challenge gate.

    Construct once, call :meth:`solve` per challenge. The context, queue and
    program are kept alive between solves so we don't pay JIT cost twice.
    """

    def __init__(self, batch_size: int | None = None,
                 device: cl.Device | None = None) -> None:
        if device is None:
            device = find_gpu()
        self.device = device
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(device=device, context=self.context)
        src = _KERNEL_REL.read_text(encoding="utf-8")
        self.program = cl.Program(self.context, src).build()
        self.kernel = self.program.blake3_challenge_search

        # Default batch: a few * max_compute_units * 1024. The RX 570 has
        # 32 CUs; 32 * 1024 = 32768 per "wave"; we want enough to hide kernel-
        # launch overhead but not so much that we overshoot the winner by far.
        # 16 Mi (≈ 2^24) is a good balance: each batch takes ~0.1s on a 570
        # and the win arrives within at most ~2x the median.
        cu = device.max_compute_units
        self.batch_size = batch_size or max(1 << 20, cu * (1 << 16))

        # Reusable device buffers.
        self._seed_buf = cl.Buffer(self.context,
                                    cl.mem_flags.READ_ONLY,
                                    size=8 * 4)
        self._found_flag = cl.Buffer(self.context,
                                      cl.mem_flags.READ_WRITE,
                                      size=4)
        self._found_nonce = cl.Buffer(self.context,
                                       cl.mem_flags.READ_WRITE,
                                       size=8)
        self._zero_u32 = np.zeros(1, dtype=np.uint32)
        self._zero_u64 = np.zeros(1, dtype=np.uint64)

    def solve(self, seed_hex: str, difficulty: int,
              progress_cb: Callable[[float, int, int], None] | None = None,
              ) -> tuple[int, float, int]:
        """Search for a nonce. Returns (nonce, seconds, hashes_tried)."""
        seed_bytes = bytes.fromhex(seed_hex)
        if len(seed_bytes) != 32:
            raise ValueError(f"seed must be 32 bytes, got {len(seed_bytes)}")
        seed_u32 = np.frombuffer(seed_bytes, dtype=np.uint32).copy()
        cl.enqueue_copy(self.queue, self._seed_buf, seed_u32)

        kernel = self.kernel
        nonce_base = 0
        batch = int(self.batch_size)
        t0 = time.time()
        hashes_tried = 0
        last_report = t0

        # Host-side wrapper for kernel.set_args + enqueue.
        while True:
            # Reset found_flag = 0 on each iteration; the in-kernel atomic_or
            # check needs a clean zero at the start of every batch.
            cl.enqueue_copy(self.queue, self._found_flag, self._zero_u32)
            cl.enqueue_copy(self.queue, self._found_nonce, self._zero_u64)

            kernel.set_args(
                self._seed_buf,
                np.uint32(difficulty),
                np.uint64(nonce_base),
                self._found_flag,
                self._found_nonce,
            )
            cl.enqueue_nd_range_kernel(self.queue, kernel, (batch,), None)
            self.queue.finish()

            flag_host = np.zeros(1, dtype=np.uint32)
            cl.enqueue_copy(self.queue, flag_host, self._found_flag)
            self.queue.finish()
            hashes_tried += batch

            if flag_host[0] != 0:
                nonce_host = np.zeros(1, dtype=np.uint64)
                cl.enqueue_copy(self.queue, nonce_host, self._found_nonce)
                self.queue.finish()
                dt = time.time() - t0
                return int(nonce_host[0]), dt, hashes_tried

            now = time.time()
            if progress_cb and now - last_report >= 1.0:
                progress_cb(now - t0, hashes_tried, batch)
                last_report = now

            nonce_base += batch


# ---------------------------------------------------------------------------- #
# Sanity check                                                                 #
# ---------------------------------------------------------------------------- #

def selftest_correctness() -> None:
    """Verify the GPU kernel against the reference ``blake3`` Python library
    by searching for a trivial difficulty=1 challenge and confirming the
    resulting hash actually starts with a zero bit.
    """
    import blake3

    seed = bytes(range(32))
    seed_hex = seed.hex()
    solver = GpuChallengeSolver(batch_size=1 << 16)  # 65k probes
    nonce, dt, tried = solver.solve(seed_hex, difficulty=8)
    digest = blake3.blake3(seed + struct.pack("<Q", nonce)).digest()
    print(f"selftest: difficulty=8 -> nonce=0x{nonce:016x} hash={digest.hex()} "
          f"({dt*1000:.1f}ms, tried={tried})")
    if digest[0] != 0:
        raise AssertionError("GPU result fails difficulty=8 (first byte non-zero)")
    print(f"  device: {solver.device.name} ({solver.device.max_compute_units} CUs)")


if __name__ == "__main__":
    selftest_correctness()
