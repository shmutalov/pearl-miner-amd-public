"""OpenCL host for GPU-accelerated ``derive_matrix``.

Replaces the CPU BLAKE3 XOF expansion (~5 s/matrix at pool shape) with a
single kernel launch: one work-item per 64-byte output block, each doing
one re-compression of the BLAKE3 root with the counter set to its output
index. The kernel folds the full 16-word state to 64 bytes, sign-extends
each byte to int8 in [-64, 63], and writes directly into the device-side
output buffer. The host then copies the int8 bytes back and reshapes.

For the typical caller path (``derive_AB_from_seed(seed, m, n, k)``), the
input to BLAKE3 is ``domain_tag || seed`` which is well under 64 bytes —
so we run the entire "absorb" stage on the CPU once (one compress) and
hand the kernel an 8-word CV (= IV in default mode) plus the 16-word
message block and flags. The kernel never touches the input bytes again.

Multi-block inputs (64 < len ≤ 1024) are also supported by absorbing all
but the last block on the host. Inputs over 1024 bytes would need a
multi-chunk path, which is not currently exercised and not implemented.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pyopencl as cl

from .device import find_gpu

_KERNEL_PATH = Path(__file__).resolve().parent.parent / "kernels" / "blake3_xof.cl"

CHUNK_LEN = 1024
BLOCK_LEN = 64

_IV = (
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
)

_FLAG_CHUNK_START = 0x01
_FLAG_CHUNK_END = 0x02
_FLAG_ROOT = 0x08


def _rotr32(x: int, n: int) -> int:
    x &= 0xFFFFFFFF
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF


_PERM = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8),
    (3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1),
    (10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6),
    (12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4),
    (9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7),
    (11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13),
)


def _g(v: list[int], a: int, b: int, c: int, d: int, mx: int, my: int) -> None:
    v[a] = (v[a] + v[b] + mx) & 0xFFFFFFFF
    v[d] = _rotr32(v[d] ^ v[a], 16)
    v[c] = (v[c] + v[d]) & 0xFFFFFFFF
    v[b] = _rotr32(v[b] ^ v[c], 12)
    v[a] = (v[a] + v[b] + my) & 0xFFFFFFFF
    v[d] = _rotr32(v[d] ^ v[a], 8)
    v[c] = (v[c] + v[d]) & 0xFFFFFFFF
    v[b] = _rotr32(v[b] ^ v[c], 7)


def _compress_to_cv(cv: list[int], m: list[int], counter: int,
                    block_len: int, flags: int) -> list[int]:
    """Reference BLAKE3 compress used host-side to absorb leading blocks of
    multi-block inputs. Returns the next 8-word chaining value."""
    counter_lo = counter & 0xFFFFFFFF
    counter_hi = (counter >> 32) & 0xFFFFFFFF
    v = [
        cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
        _IV[0], _IV[1], _IV[2], _IV[3],
        counter_lo, counter_hi, block_len, flags,
    ]
    for perm in _PERM:
        m2 = [m[perm[i]] for i in range(16)]
        _g(v, 0, 4, 8, 12, m2[0], m2[1])
        _g(v, 1, 5, 9, 13, m2[2], m2[3])
        _g(v, 2, 6, 10, 14, m2[4], m2[5])
        _g(v, 3, 7, 11, 15, m2[6], m2[7])
        _g(v, 0, 5, 10, 15, m2[8], m2[9])
        _g(v, 1, 6, 11, 12, m2[10], m2[11])
        _g(v, 2, 7, 8, 13, m2[12], m2[13])
        _g(v, 3, 4, 9, 14, m2[14], m2[15])
    return [(v[i] ^ v[i + 8]) & 0xFFFFFFFF for i in range(8)]


def _block_words(block: bytes) -> list[int]:
    padded = block + b"\x00" * (BLOCK_LEN - len(block))
    return list(struct.unpack("<16I", padded))


def _prepare_root(input_data: bytes) -> tuple[list[int], list[int], int, int]:
    """Compute (cv, m_words, block_len, base_flags) for the root re-compress.

    Returns the arguments the GPU kernel needs to produce the BLAKE3 XOF
    stream of `input_data` in default (un-keyed) mode. ROOT is added by
    the kernel itself.
    """
    if len(input_data) > CHUNK_LEN:
        raise NotImplementedError(
            "derive_matrix_gpu currently assumes input fits in one chunk "
            f"(≤ {CHUNK_LEN} bytes); got {len(input_data)}")

    cv = list(_IV)
    if len(input_data) <= BLOCK_LEN:
        m_words = _block_words(input_data)
        block_len = len(input_data)
        flags = _FLAG_CHUNK_START | _FLAG_CHUNK_END
        return cv, m_words, block_len, flags

    # Multi-block single chunk: absorb all but the last block on the host.
    n_blocks = (len(input_data) + BLOCK_LEN - 1) // BLOCK_LEN
    for i in range(n_blocks - 1):
        block = input_data[i * BLOCK_LEN:(i + 1) * BLOCK_LEN]
        m_words = _block_words(block)
        flags = _FLAG_CHUNK_START if i == 0 else 0
        cv = _compress_to_cv(cv, m_words, counter=0,
                             block_len=BLOCK_LEN, flags=flags)
    last = input_data[(n_blocks - 1) * BLOCK_LEN:]
    m_words = _block_words(last)
    block_len = len(last)
    flags = _FLAG_CHUNK_END
    if n_blocks == 1:
        flags |= _FLAG_CHUNK_START
    return cv, m_words, block_len, flags


class DeriveMatrixGpu:
    """Reusable context + program for GPU BLAKE3-XOF matrix derivation.

    Construct once, call :meth:`derive` per matrix. The kernel is JIT-built
    on first use and cached for the instance's lifetime.

    Pass ``context`` / ``queue`` to share OpenCL state with sibling classes
    (``MerkleGpu``, ``JackpotGpu``, …) so output buffers stay in the same
    address space and can be consumed in-place without PCIe round-trips.
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
        self.kernel = self.program.blake3_xof_derive_int8

    def derive(self, seed: bytes, rows: int, cols: int,
               domain_tag: bytes = b"PEARL_MATRIX",
               *, out_buf: cl.Buffer | None = None,
               read_back: bool = True) -> tuple[np.ndarray | None, cl.Buffer]:
        """GPU equivalent of ``proof_builder.derive_matrix``.

        Returns ``(host_array_or_None, device_buffer)``. The host array is
        ``(rows, cols)`` int8 in [-64, 63] when ``read_back=True``; pass
        ``read_back=False`` to keep the output on device only. Pass
        ``out_buf`` to write into an existing device buffer.
        """
        if len(seed) != 32:
            raise ValueError(f"seed must be 32 bytes, got {len(seed)}")
        if rows <= 0 or cols <= 0:
            raise ValueError(f"rows and cols must be positive, got ({rows}, {cols})")

        input_data = domain_tag + seed
        cv, m_words, block_len, base_flags = _prepare_root(input_data)

        n_bytes = rows * cols
        n_blocks = (n_bytes + BLOCK_LEN - 1) // BLOCK_LEN

        ctx, mf = self.context, cl.mem_flags
        cv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                           hostbuf=np.array(cv, dtype=np.uint32))
        m_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                          hostbuf=np.array(m_words, dtype=np.uint32))
        if out_buf is None:
            out_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n_bytes)

        self.kernel.set_args(
            cv_buf, m_buf,
            np.uint32(block_len),
            np.uint32(base_flags),
            out_buf,
            np.uint64(n_bytes),
        )
        cl.enqueue_nd_range_kernel(self.queue, self.kernel, (n_blocks,), None)

        if not read_back:
            return None, out_buf

        out = np.empty(n_bytes, dtype=np.int8)
        cl.enqueue_copy(self.queue, out, out_buf)
        self.queue.finish()
        return out.reshape(rows, cols), out_buf

    def derive_AB(self, miner_seed: bytes, m: int, n: int, k: int,
                  *, read_back: bool = True
                  ) -> tuple[np.ndarray | None, np.ndarray | None,
                             cl.Buffer, cl.Buffer]:
        """Convenience: derive both A (m×k) and B (n×k). Returns
        ``(A_host, B_host, A_buf, B_buf)``. Host arrays are ``None`` when
        ``read_back=False`` (device-only path)."""
        A, A_buf = self.derive(miner_seed, m, k, domain_tag=b"PEARL_MATRIX_A",
                               read_back=read_back)
        B, B_buf = self.derive(miner_seed, n, k, domain_tag=b"PEARL_MATRIX_B",
                               read_back=read_back)
        return A, B, A_buf, B_buf


# ---------------------------------------------------------------------------- #
# Selftest                                                                     #
# ---------------------------------------------------------------------------- #

def _selftest_against_cpu() -> None:
    import time

    from .proof_builder import derive_matrix

    gpu = DeriveMatrixGpu()
    print(f"  device: {gpu.device.name} ({gpu.device.max_compute_units} CUs)")

    cases = [
        # (rows, cols, domain_tag)
        (1, 1, b"X"),
        (1, 64, b"PEARL_MATRIX_A"),
        (1, 65, b"PEARL_MATRIX_A"),
        (4, 256, b"PEARL_MATRIX_B"),
        (64, 256, b"PEARL_MATRIX_A"),
        (256, 1024, b"PEARL_MATRIX_B"),
        (1024, 1024, b"PEARL_MATRIX_A"),
    ]
    seed = bytes(range(32))
    for rows, cols, tag in cases:
        cpu = derive_matrix(seed, rows, cols, domain_tag=tag)
        out, _ = gpu.derive(seed, rows, cols, domain_tag=tag)
        if not np.array_equal(cpu, out):
            diff = np.argwhere(cpu != out)[:5]
            raise AssertionError(
                f"mismatch at rows={rows} cols={cols} tag={tag!r}: "
                f"first diffs at {diff.tolist()}\n"
                f"  cpu[0,:8]={cpu.ravel()[:8].tolist()}\n"
                f"  gpu[0,:8]={out.ravel()[:8].tolist()}")
        print(f"  {rows:>5}x{cols:<6} tag={tag.decode():<14} → {out.ravel()[:6].tolist()}... ✓")


def _bench_pool_shape() -> None:
    import time

    from .proof_builder import derive_AB_from_seed

    gpu = DeriveMatrixGpu()
    miner_seed = bytes.fromhex("c0ffee" * 10 + "abcd")
    assert len(miner_seed) == 32

    m, n, k = 131072, 131072, 4096
    n_bytes_per = m * k
    print(f"  shape: m=n={m} k={k} ({n_bytes_per // (1024 * 1024)} MiB per matrix)")

    # Warm-up (small)
    gpu.derive(miner_seed, 64, 256, b"WARMUP")

    print("  GPU derive A: ", end="", flush=True)
    t0 = time.time()
    A_gpu, _ = gpu.derive(miner_seed, m, k, b"PEARL_MATRIX_A")
    t1 = time.time()
    print(f"{(t1 - t0) * 1000:.0f}ms  ({n_bytes_per / (t1 - t0) / 1e9:.1f} GB/s)")

    print("  GPU derive B: ", end="", flush=True)
    t0 = time.time()
    B_gpu, _ = gpu.derive(miner_seed, n, k, b"PEARL_MATRIX_B")
    t1 = time.time()
    print(f"{(t1 - t0) * 1000:.0f}ms  ({n_bytes_per / (t1 - t0) / 1e9:.1f} GB/s)")

    print("  CPU derive A: ", end="", flush=True)
    t0 = time.time()
    A_cpu, B_cpu = derive_AB_from_seed(miner_seed, m, n, k)
    t1 = time.time()
    print(f"{(t1 - t0) * 1000:.0f}ms  (both A and B together)")

    assert np.array_equal(A_cpu, A_gpu), "A mismatch at pool shape"
    assert np.array_equal(B_cpu, B_gpu), "B mismatch at pool shape"
    print("  A and B both match CPU bit-for-bit at pool shape ✓")


if __name__ == "__main__":
    print("== correctness ==")
    _selftest_against_cpu()
    print("\n== pool-shape bench ==")
    _bench_pool_shape()
