"""OpenCL host for GPU-accelerated noise derivation.

Replaces the CPU paths in ``jackpot.generate_uniform_random_matrix`` and
``jackpot.generate_permutation_matrix``:

  - ``uniform(seed, key, num_rows, num_cols)``  → (num_rows, num_cols) int8
    in [-32, 31]. One BLAKE3-keyed single-block compress per 32-byte
    output chunk; ``num_cols`` must be a multiple of 32. The kernel writes
    sign-extended bytes directly.

  - ``perm(seed, key, k, noise_rank)``         → (k, 2) uint32. One
    compress per 8 output rows; XOR/mul_hi rule per ``pearl_noise.rs``.

At pool shape (m=n=131072, k=4096, r=128) the CPU path takes ~5 s for
all four matrices (e_al, e_ar_t, e_bl, e_br_t). The GPU path needs ~1 M
single-block compressions and lands in tens of milliseconds.

Bit-identical to the CPU oracle in ``jackpot.py``.

A constructor accepts an optional ``(context, queue)`` pair so callers
(e.g. ``JackpotGpu``) can share OpenCL state and pass device buffers
directly without a host round-trip. If no context is supplied, a fresh
one is created against the first GPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyopencl as cl

from .device import find_gpu

_KERNEL_PATH = Path(__file__).resolve().parent.parent / "kernels" / "pearl_noise.cl"

BLAKE3_DIGEST_SIZE = 32


class PearlNoiseGpu:
    """GPU equivalent of the CPU noise generators in ``jackpot.py``.

    Compile the kernel once on construction; ``uniform`` / ``perm`` are
    cheap per-call. Each call allocates a fresh output buffer; callers
    that need to reuse VRAM can pass ``out_buf=`` to write into an
    existing device buffer.
    """

    def __init__(self, context: cl.Context | None = None,
                 queue: cl.CommandQueue | None = None,
                 device: cl.Device | None = None) -> None:
        if context is None:
            self.device = device or find_gpu()
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context, device=self.device)
        else:
            self.context = context
            self.queue = queue or cl.CommandQueue(context)
            self.device = device or self.queue.device

        src = _KERNEL_PATH.read_text(encoding="utf-8")
        self.program = cl.Program(self.context, src).build()
        self.k_uniform = self.program.pearl_noise_uniform_int8
        self.k_perm = self.program.pearl_noise_perm_u32

    # ------------------------------------------------------------------- #
    # Uniform                                                             #
    # ------------------------------------------------------------------- #

    def uniform(self, seed: bytes, key: bytes, num_rows: int, num_cols: int,
                *, out_buf: cl.Buffer | None = None,
                read_back: bool = True) -> tuple[np.ndarray | None, cl.Buffer]:
        """Compute ``(num_rows, num_cols)`` int8 in [-32, 31]."""
        if len(seed) != 32 or len(key) != 32:
            raise ValueError("seed and key must both be 32 bytes")
        if num_cols % BLAKE3_DIGEST_SIZE != 0:
            raise ValueError(
                f"num_cols must be a multiple of {BLAKE3_DIGEST_SIZE}, got {num_cols}")
        if num_rows <= 0 or num_cols <= 0:
            raise ValueError("num_rows and num_cols must be positive")

        ctx, mf = self.context, cl.mem_flags
        seed_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                             hostbuf=np.frombuffer(seed, dtype=np.uint32).copy())
        key_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                            hostbuf=np.frombuffer(key, dtype=np.uint32).copy())
        n_bytes = num_rows * num_cols
        if out_buf is None:
            out_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n_bytes)

        hashes_per_row = num_cols // BLAKE3_DIGEST_SIZE
        total_hashes = num_rows * hashes_per_row
        self.k_uniform.set_args(
            seed_buf, key_buf, out_buf,
            np.uint32(num_rows), np.uint32(num_cols),
        )
        cl.enqueue_nd_range_kernel(self.queue, self.k_uniform,
                                   (total_hashes,), None)

        if not read_back:
            return None, out_buf

        host = np.empty((num_rows, num_cols), dtype=np.int8)
        cl.enqueue_copy(self.queue, host, out_buf)
        self.queue.finish()
        return host, out_buf

    # ------------------------------------------------------------------- #
    # Permutation                                                         #
    # ------------------------------------------------------------------- #

    def perm(self, seed: bytes, key: bytes, k: int, noise_rank: int,
             *, out_buf: cl.Buffer | None = None,
             read_back: bool = True) -> tuple[np.ndarray | None, cl.Buffer]:
        """Compute ``(k, 2)`` uint32 sparse permutation rows."""
        if len(seed) != 32 or len(key) != 32:
            raise ValueError("seed and key must both be 32 bytes")
        if noise_rank <= 0 or (noise_rank & (noise_rank - 1)) != 0:
            raise ValueError("noise_rank must be a power of two")
        if k <= 0:
            raise ValueError("k must be positive")

        ctx, mf = self.context, cl.mem_flags
        seed_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                             hostbuf=np.frombuffer(seed, dtype=np.uint32).copy())
        key_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                            hostbuf=np.frombuffer(key, dtype=np.uint32).copy())
        n_bytes = k * 2 * 4
        if out_buf is None:
            out_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n_bytes)

        lines_per_hash = 8
        n_blocks = (k + lines_per_hash - 1) // lines_per_hash
        self.k_perm.set_args(
            seed_buf, key_buf, out_buf,
            np.uint32(k), np.uint32(noise_rank),
        )
        cl.enqueue_nd_range_kernel(self.queue, self.k_perm,
                                   (n_blocks,), None)

        if not read_back:
            return None, out_buf

        host = np.empty((k, 2), dtype=np.uint32)
        cl.enqueue_copy(self.queue, host, out_buf)
        self.queue.finish()
        return host, out_buf


# ---------------------------------------------------------------------------- #
# Selftest                                                                     #
# ---------------------------------------------------------------------------- #

def _selftest_against_cpu() -> None:
    import time

    from .jackpot import (
        SEED_LABEL_A, SEED_LABEL_B,
        generate_permutation_matrix, generate_uniform_random_matrix,
    )

    gpu = PearlNoiseGpu()
    print(f"  device: {gpu.device.name} ({gpu.device.max_compute_units} CUs)")

    # Reproducible seed/key pair — neither needs to be a real BLAKE3 hash for
    # the unit test.
    key = bytes(range(32))

    # Uniform: a few shapes.
    print("\n  -- uniform --")
    for num_rows, num_cols, seed in [
        (1, 32, SEED_LABEL_A),
        (4, 32, SEED_LABEL_A),
        (4, 128, SEED_LABEL_B),
        (256, 128, SEED_LABEL_A),
        (1024, 128, SEED_LABEL_B),
    ]:
        cpu = generate_uniform_random_matrix(seed, key, list(range(num_rows)), num_cols)
        gpu_out, _ = gpu.uniform(seed, key, num_rows, num_cols)
        if not np.array_equal(cpu, gpu_out):
            diffs = np.argwhere(cpu != gpu_out)[:5]
            raise AssertionError(
                f"uniform mismatch at ({num_rows}, {num_cols}) seed={seed[:8]!r}: "
                f"first diffs={diffs.tolist()}\n"
                f"  cpu[0,:8]={cpu.ravel()[:8].tolist()}\n"
                f"  gpu[0,:8]={gpu_out.ravel()[:8].tolist()}")
        print(f"    {num_rows:>5}x{num_cols:<4} seed={seed[:1]}... → {gpu_out.ravel()[:6].tolist()}... ✓")

    # Perm: a few k/rank pairs.
    print("\n  -- perm --")
    for k, r, seed in [
        (8, 64, SEED_LABEL_A),
        (256, 64, SEED_LABEL_A),
        (256, 128, SEED_LABEL_B),
        (4096, 128, SEED_LABEL_A),
    ]:
        cpu = generate_permutation_matrix(seed, key, k, r)
        gpu_out, _ = gpu.perm(seed, key, k, r)
        if not np.array_equal(cpu, gpu_out):
            diffs = np.argwhere(cpu != gpu_out)[:5]
            raise AssertionError(
                f"perm mismatch at (k={k}, r={r}) seed={seed[:8]!r}: "
                f"first diffs={diffs.tolist()}\n"
                f"  cpu[:3]={cpu[:3].tolist()}\n"
                f"  gpu[:3]={gpu_out[:3].tolist()}")
        print(f"    k={k:>5} r={r:<4} seed={seed[:1]}... → first={tuple(gpu_out[0])} last={tuple(gpu_out[-1])} ✓")


def _bench_pool_shape() -> None:
    import time

    from .jackpot import (
        SEED_LABEL_A, SEED_LABEL_B,
        generate_permutation_matrix, generate_uniform_random_matrix,
    )

    m, n, k, r = 131072, 131072, 4096, 128
    a_noise_seed = bytes(range(32))
    b_noise_seed = bytes(range(31, -1, -1))

    gpu = PearlNoiseGpu()
    print(f"  device: {gpu.device.name} ({gpu.device.max_compute_units} CUs)")
    print(f"  shape: m=n={m} k={k} r={r}")

    # Warm-up
    gpu.uniform(SEED_LABEL_A, a_noise_seed, 64, 128)

    # ---- GPU ----
    print("\n  -- GPU --")
    t0 = time.time()
    e_al, _ = gpu.uniform(SEED_LABEL_A, a_noise_seed, m, r)
    t1 = time.time()
    print(f"    e_al    ({m}x{r}): {(t1 - t0) * 1000:6.0f}ms")
    t0 = time.time()
    e_ar_t, _ = gpu.perm(SEED_LABEL_A, a_noise_seed, k, r)
    t1 = time.time()
    print(f"    e_ar_t  ({k}x2):   {(t1 - t0) * 1000:6.0f}ms")
    t0 = time.time()
    e_bl, _ = gpu.perm(SEED_LABEL_B, b_noise_seed, k, r)
    t1 = time.time()
    print(f"    e_bl    ({k}x2):   {(t1 - t0) * 1000:6.0f}ms")
    t0 = time.time()
    e_br_t, _ = gpu.uniform(SEED_LABEL_B, b_noise_seed, n, r)
    t1 = time.time()
    print(f"    e_br_t  ({n}x{r}): {(t1 - t0) * 1000:6.0f}ms")

    # ---- CPU oracle (only first few rows for speed) ----
    print("\n  -- CPU (full pool shape, all four matrices) --")
    t0 = time.time()
    cpu_e_al = generate_uniform_random_matrix(
        SEED_LABEL_A, a_noise_seed, list(range(m)), r)
    cpu_e_ar_t = generate_permutation_matrix(SEED_LABEL_A, a_noise_seed, k, r)
    cpu_e_bl = generate_permutation_matrix(SEED_LABEL_B, b_noise_seed, k, r)
    cpu_e_br_t = generate_uniform_random_matrix(
        SEED_LABEL_B, b_noise_seed, list(range(n)), r)
    t1 = time.time()
    print(f"    all four matrices: {(t1 - t0) * 1000:.0f}ms")

    assert np.array_equal(cpu_e_al, e_al), "e_al mismatch at pool shape"
    assert np.array_equal(cpu_e_ar_t, e_ar_t), "e_ar_t mismatch at pool shape"
    assert np.array_equal(cpu_e_bl, e_bl), "e_bl mismatch at pool shape"
    assert np.array_equal(cpu_e_br_t, e_br_t), "e_br_t mismatch at pool shape"
    print("\n  all four matrices bit-identical to CPU ✓")


if __name__ == "__main__":
    print("== correctness ==")
    _selftest_against_cpu()
    print("\n== pool-shape bench ==")
    _bench_pool_shape()
