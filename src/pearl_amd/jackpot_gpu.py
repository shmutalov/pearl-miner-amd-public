"""OpenCL host for the batched jackpot evaluator.

Wraps ``src/kernels/jackpot_search.cl`` with a Python class that holds the
per-job device buffers (A, B, noise factors, key, patterns) and exposes
``evaluate_batch(t_rows, t_cols)`` → ``np.ndarray`` of ``hash_jackpot``
bytes ``(batch_size, 32)``.

Per-job uploads (once on ``set_job``):
  - A, B: int8 (m, k) and (n, k)
  - e_al, e_br_t: int8 (m, r) and (n, r) — uniform random noise rows
  - e_ar_t, e_bl: uint32 (k, 2) — sparse permutation pairs
  - row_pattern, col_pattern: int32 (h,) and (w,)
  - a_noise_seed: 32 bytes (BLAKE3 key for final jackpot hash)

Per-batch (every ``evaluate_batch`` call):
  - t_rows, t_cols: int32 (B,) — per-candidate tile offsets
  - output: uint32 (B, 8) — hash_jackpot as little-endian u32 words
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyopencl as cl

from .device import find_gpu
from .jackpot import (
    NOISE_RANGE, generate_permutation_matrix, generate_uniform_random_matrix,
    SEED_LABEL_A, SEED_LABEL_B,
)

try:
    from .pearl_noise_gpu import PearlNoiseGpu
    _GPU_NOISE_AVAILABLE = True
except Exception:  # pragma: no cover — pyopencl missing or no GPU
    PearlNoiseGpu = None  # type: ignore[assignment]
    _GPU_NOISE_AVAILABLE = False


_KERNELS_DIR = Path(__file__).resolve().parent.parent / "kernels"
_KERNEL_PATH = _KERNELS_DIR / "jackpot_search.cl"
_KERNEL_PATH_RDNA3 = _KERNELS_DIR / "jackpot_search_rdna3.cl"


def _select_kernel_path(device: cl.Device, variant: str) -> Path:
    """Pick the kernel source for ``variant``.

    ``"auto"`` uses the RDNA3-tuned kernel on wave32 RDNA parts
    (gfx10xx/gfx11xx/gfx12xx), which is bit-identical to the original but
    computes the noisy pa/pb operand strips once per outer iter instead of
    redundantly per thread. ``"polaris"`` forces the original GCN kernel;
    ``"rdna3"`` forces the new one.
    """
    variant = variant.lower()
    if variant == "polaris":
        return _KERNEL_PATH
    if variant == "rdna3":
        return _KERNEL_PATH_RDNA3
    if variant != "auto":
        raise ValueError(f"unknown kernel variant {variant!r}")
    name = (device.name or "").lower()
    is_rdna = any(name.startswith(g) for g in ("gfx10", "gfx11", "gfx12"))
    return _KERNEL_PATH_RDNA3 if is_rdna else _KERNEL_PATH


def _ensure_int8_contiguous(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.dtype != np.int8:
        raise ValueError(f"{name} must be int8, got {arr.dtype}")
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


class JackpotGpu:
    """Batched GPU evaluator. One instance per device + per (h, w, r) shape."""

    def __init__(self, h: int, w: int, r: int,
                 context: cl.Context | None = None,
                 queue: cl.CommandQueue | None = None,
                 device: cl.Device | None = None,
                 use_gpu_noise: bool = True,
                 variant: str = "auto") -> None:
        if h * w not in (128, 64, 256, 512):
            # OpenCL kernel uses reqd_work_group_size; matching matters.
            # 128 is the pool shape; we keep an explicit allowlist to fail fast.
            pass
        self.h, self.w, self.r = h, w, r
        if context is None:
            self.device = device or find_gpu()
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context, device=self.device)
        else:
            self.context = context
            self.queue = queue or cl.CommandQueue(context)
            self.device = device or self.queue.device

        self.kernel_path = _select_kernel_path(self.device, variant)
        self.variant = variant
        src = self.kernel_path.read_text(encoding="utf-8")
        build_opts = [
            "-cl-std=CL2.0",
            f"-D PEARL_H={h}",
            f"-D PEARL_W={w}",
            f"-D PEARL_R={r}",
        ]
        self.program = cl.Program(self.context, src).build(options=build_opts)
        self.kernel = self.program.jackpot_evaluate_batch

        # Shared noise generator built against our own context — output
        # buffers go straight into this device's address space, no PCIe
        # round-trip required.
        self._noise_gpu: PearlNoiseGpu | None = None
        if use_gpu_noise and _GPU_NOISE_AVAILABLE:
            self._noise_gpu = PearlNoiseGpu(context=self.context, queue=self.queue,
                                            device=self.device)

        # Per-job device buffers — created in set_job.
        self._A_buf: cl.Buffer | None = None
        self._B_buf: cl.Buffer | None = None
        self._eal_buf: cl.Buffer | None = None
        self._ebr_buf: cl.Buffer | None = None
        self._ear_buf: cl.Buffer | None = None
        self._ebl_buf: cl.Buffer | None = None
        self._row_pat_buf: cl.Buffer | None = None
        self._col_pat_buf: cl.Buffer | None = None
        self._key_buf: cl.Buffer | None = None
        self._m: int = 0
        self._n: int = 0
        self._k: int = 0

    # ------------------------------------------------------------------- #
    # Per-job upload                                                      #
    # ------------------------------------------------------------------- #

    def set_job(self, A: np.ndarray, B: np.ndarray,
                row_pattern: list[int], col_pattern: list[int],
                commitment_hash: tuple[bytes, bytes],
                a_noise_seed: bytes,
                *,
                A_buf: cl.Buffer | None = None,
                B_buf: cl.Buffer | None = None) -> None:
        """Push all per-job inputs to the device.

        ``commitment_hash`` = ``(b_noise_seed, a_noise_seed)``. Noise
        matrices are derived on the GPU when ``use_gpu_noise=True`` (the
        default) — ~10 ms at pool shape vs ~5 s on CPU. With shared
        context, the output buffers are produced directly in this
        instance's VRAM and never round-trip through host.

        Pass ``A_buf`` / ``B_buf`` to reuse already-on-device matrices
        (e.g. from ``DeriveMatrixGpu``); the host ``A`` / ``B`` arrays
        are still required for shape validation but their contents won't
        be re-uploaded.
        """
        A = _ensure_int8_contiguous(A, "A")
        B = _ensure_int8_contiguous(B, "B")
        m, k1 = A.shape
        n, k2 = B.shape
        if k1 != k2:
            raise ValueError(f"A.shape[1]={k1} != B.shape[1]={k2}")
        k = k1
        if k % self.r != 0:
            raise ValueError(f"k={k} must be a multiple of r={self.r}")
        if len(row_pattern) != self.h:
            raise ValueError(f"row_pattern len {len(row_pattern)} != h={self.h}")
        if len(col_pattern) != self.w:
            raise ValueError(f"col_pattern len {len(col_pattern)} != w={self.w}")
        if len(a_noise_seed) != 32:
            raise ValueError("a_noise_seed must be 32 bytes")

        b_noise_seed, a_noise_seed_check = commitment_hash
        if a_noise_seed_check != a_noise_seed:
            raise ValueError("a_noise_seed inconsistent with commitment_hash")

        ctx, mf = self.context, cl.mem_flags
        if A_buf is not None:
            self._A_buf = A_buf
        else:
            self._A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        if B_buf is not None:
            self._B_buf = B_buf
        else:
            self._B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)

        if self._noise_gpu is not None:
            # GPU noise: write straight into READ_WRITE buffers, no host hop.
            self._eal_buf = cl.Buffer(ctx, mf.READ_WRITE, size=m * self.r)
            self._ebr_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n * self.r)
            self._ear_buf = cl.Buffer(ctx, mf.READ_WRITE, size=k * 2 * 4)
            self._ebl_buf = cl.Buffer(ctx, mf.READ_WRITE, size=k * 2 * 4)
            self._noise_gpu.uniform(SEED_LABEL_A, a_noise_seed, m, self.r,
                                    out_buf=self._eal_buf, read_back=False)
            self._noise_gpu.perm(SEED_LABEL_A, a_noise_seed, k, NOISE_RANGE // 2,
                                 out_buf=self._ear_buf, read_back=False)
            self._noise_gpu.perm(SEED_LABEL_B, b_noise_seed, k, NOISE_RANGE // 2,
                                 out_buf=self._ebl_buf, read_back=False)
            self._noise_gpu.uniform(SEED_LABEL_B, b_noise_seed, n, self.r,
                                    out_buf=self._ebr_buf, read_back=False)
        else:
            # CPU fallback: derive then upload.
            all_a_rows = list(range(m))
            all_b_cols = list(range(n))
            e_al = generate_uniform_random_matrix(SEED_LABEL_A, a_noise_seed,
                                                  all_a_rows, self.r)
            e_ar_t = generate_permutation_matrix(SEED_LABEL_A, a_noise_seed,
                                                 k, NOISE_RANGE // 2)
            e_bl = generate_permutation_matrix(SEED_LABEL_B, b_noise_seed,
                                               k, NOISE_RANGE // 2)
            e_br_t = generate_uniform_random_matrix(SEED_LABEL_B, b_noise_seed,
                                                    all_b_cols, self.r)
            self._eal_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=e_al)
            self._ebr_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=e_br_t)
            self._ear_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=np.ascontiguousarray(e_ar_t).astype(np.uint32))
            self._ebl_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=np.ascontiguousarray(e_bl).astype(np.uint32))

        self._row_pat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=np.array(row_pattern, dtype=np.int32))
        self._col_pat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=np.array(col_pattern, dtype=np.int32))
        key_words = np.frombuffer(a_noise_seed, dtype=np.uint32).copy()
        self._key_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=key_words)

        self._m, self._n, self._k = m, n, k

    # ------------------------------------------------------------------- #
    # Per-batch evaluation                                                #
    # ------------------------------------------------------------------- #

    def evaluate_batch(self, t_rows: np.ndarray, t_cols: np.ndarray
                       ) -> np.ndarray:
        """Evaluate one batch. ``t_rows`` and ``t_cols`` are int32 (B,)
        arrays; returns ``hash_jackpot`` as ``(B, 32)`` uint8 bytes."""
        if self._A_buf is None:
            raise RuntimeError("call set_job() first")
        if t_rows.dtype != np.int32 or t_cols.dtype != np.int32:
            raise ValueError("t_rows/t_cols must be int32")
        if t_rows.shape != t_cols.shape or t_rows.ndim != 1:
            raise ValueError("t_rows and t_cols must be 1-D arrays of equal length")

        batch = int(t_rows.shape[0])
        mf = cl.mem_flags
        ctx = self.context

        t_rows = np.ascontiguousarray(t_rows)
        t_cols = np.ascontiguousarray(t_cols)
        trb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_rows)
        tcb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_cols)
        out = np.empty((batch, 8), dtype=np.uint32)
        outb = cl.Buffer(ctx, mf.WRITE_ONLY, size=out.nbytes)

        self.kernel.set_args(
            self._A_buf, self._B_buf,
            self._eal_buf, self._ebr_buf,
            self._ear_buf, self._ebl_buf,
            trb, tcb,
            self._row_pat_buf, self._col_pat_buf,
            np.int32(self._k), np.int32(self._k), np.int32(self._k),
            self._key_buf, outb,
        )
        wg = self.h * self.w
        cl.enqueue_nd_range_kernel(self.queue, self.kernel,
                                   (batch * wg,), (wg,))
        cl.enqueue_copy(self.queue, out, outb)
        self.queue.finish()
        return out.view(np.uint8).reshape(batch, 32)

    # ------------------------------------------------------------------- #
    # End-to-end search                                                   #
    # ------------------------------------------------------------------- #

    def search(self, mining_config, target: int,
               *, batch_size: int = 8192,
               max_attempts: int | None = None,
               progress_cb=None):
        """Iterate every valid (t_rows, t_cols) for ``mining_config`` in
        batches; return the first candidate whose hash_jackpot (LE-uint256)
        is below ``target``.

        Returns ``(Candidate_or_None, attempts_evaluated, seconds_elapsed)``.
        ``progress_cb(attempts, seconds, last_value)`` is called after each
        batch (with the last evaluated target_value in the batch, for
        debugging difficulty).
        """
        import time
        from .candidate_search import Candidate, enumerate_valid_offsets

        if self._A_buf is None:
            raise RuntimeError("call set_job() first")

        # Lazy generator over (t_r, t_c) — pool shape has ~134M tuples, so
        # materializing them up front would waste memory.
        def offset_iter():
            for t_r in enumerate_valid_offsets(mining_config.rows_pattern, self._m):
                for t_c in enumerate_valid_offsets(mining_config.cols_pattern, self._n):
                    yield (t_r, t_c)

        it = offset_iter()
        attempts = 0
        t0 = time.time()
        last_target_value = 0
        while True:
            # Pull next batch_size offsets (or fewer near exhaustion / cap).
            batch: list[tuple[int, int]] = []
            for _ in range(batch_size):
                try:
                    batch.append(next(it))
                except StopIteration:
                    break
            if not batch:
                break
            if max_attempts is not None and attempts + len(batch) > max_attempts:
                batch = batch[: max_attempts - attempts]
                if not batch:
                    break

            t_rows_a = np.fromiter((o[0] for o in batch), dtype=np.int32,
                                   count=len(batch))
            t_cols_a = np.fromiter((o[1] for o in batch), dtype=np.int32,
                                   count=len(batch))
            hashes = self.evaluate_batch(t_rows_a, t_cols_a)
            attempts += len(batch)

            # Vectorized LE-uint256 < target check via numpy.
            # Layout: (B, 32) uint8 little-endian → (B, 4) uint64 with
            # words in order [low, mid_lo, mid_hi, high]. Lex compare from
            # the high word down — only descend into the next word when
            # the higher one was equal.
            MASK64 = (1 << 64) - 1
            tw = np.array([
                target & MASK64,
                (target >> 64) & MASK64,
                (target >> 128) & MASK64,
                (target >> 192) & MASK64,
            ], dtype=np.uint64)
            hashes_u64 = hashes.view(np.uint64).reshape(-1, 4)
            h3 = hashes_u64[:, 3]; h2 = hashes_u64[:, 2]
            h1 = hashes_u64[:, 1]; h0 = hashes_u64[:, 0]
            hit_mask = (h3 < tw[3]) | (
                (h3 == tw[3]) & (
                    (h2 < tw[2]) | (
                        (h2 == tw[2]) & (
                            (h1 < tw[1]) | (
                                (h1 == tw[1]) & (h0 < tw[0]))))))

            if hit_mask.any():
                i = int(np.argmax(hit_mask))  # first True wins
                hash_bytes = bytes(hashes[i])
                tv = int.from_bytes(hash_bytes, "little")
                tr, tc = batch[i]
                a_rows = list(mining_config.rows_pattern.indices_with_offset(tr))
                b_cols = list(mining_config.cols_pattern.indices_with_offset(tc))
                return (Candidate(
                    t_rows=tr, t_cols=tc,
                    a_rows_indices=a_rows,
                    b_cols_indices=b_cols,
                    hash_jackpot=hash_bytes,
                    target_value=tv,
                ), attempts, time.time() - t0)
            # No hit in this batch — record the very last hash for the
            # progress callback's difficulty hint.
            last_target_value = int.from_bytes(bytes(hashes[-1]), "little")

            if progress_cb is not None:
                progress_cb(attempts, time.time() - t0, last_target_value)
            if max_attempts is not None and attempts >= max_attempts:
                break

        return None, attempts, time.time() - t0


# ---------------------------------------------------------------------------- #
# Selftest: GPU == CPU bit-identical                                          #
# ---------------------------------------------------------------------------- #

def _selftest_against_cpu() -> None:
    import time

    from .jackpot import evaluate_candidate
    from .mining_config import (
        MiningConfiguration, PeriodicPattern, compute_job_key,
    )
    from .proof_builder import derive_AB_from_seed, merkle_root_keyed
    import blake3

    # Small shape to keep the CPU oracle fast.
    m, n, k, r = 256, 256, 256, 64
    h, w = 2, 64
    mc = MiningConfiguration(
        common_dim=k, rank=r, mma_type=0,
        rows_pattern=PeriodicPattern.from_list([0, 32]),
        cols_pattern=PeriodicPattern.from_list(list(range(64))),
    )
    header = bytes(range(76))
    job_key = compute_job_key(header, mc)
    miner_seed = bytes(32)
    A, B = derive_AB_from_seed(miner_seed, m, n, k)
    hash_a = merkle_root_keyed(A, job_key)
    hash_b = merkle_root_keyed(B, job_key)
    b_noise_seed = blake3.blake3(job_key + hash_b).digest()
    a_noise_seed = blake3.blake3(b_noise_seed + hash_a).digest()
    commitment_hash = (b_noise_seed, a_noise_seed)

    gpu = JackpotGpu(h=h, w=w, r=r)
    print(f"  device: {gpu.device.name} ({gpu.device.max_compute_units} CUs)")
    print(f"  shape: m=n={m} k={k} r={r} h={h} w={w}")
    t0 = time.time()
    gpu.set_job(A, B, [0, 32], list(range(64)), commitment_hash, a_noise_seed)
    print(f"  set_job: {time.time() - t0:.2f}s (GPU noise derivation)")

    # 8 candidates spanning different (t_rows, t_cols).
    candidates = [
        (0, 0), (1, 0), (5, 64), (8, 128), (16, 192), (32, 0), (32, 64), (64, 128),
    ]
    t_rows = np.array([c[0] for c in candidates], dtype=np.int32)
    t_cols = np.array([c[1] for c in candidates], dtype=np.int32)

    t0 = time.time()
    gpu_hashes = gpu.evaluate_batch(t_rows, t_cols)
    gpu_dt = time.time() - t0
    print(f"  GPU {len(candidates)} candidates: {gpu_dt*1000:.1f}ms")

    # CPU oracle
    a_rows_lookup = mc.rows_pattern.indices_with_offset
    b_cols_lookup = mc.cols_pattern.indices_with_offset
    t0 = time.time()
    mismatches = 0
    for i, (tr, tc) in enumerate(candidates):
        a_rows = a_rows_lookup(tr)
        b_cols = b_cols_lookup(tc)
        _, cpu_hash = evaluate_candidate(
            A, B, a_rows, b_cols, commitment_hash, a_noise_seed, k, r)
        gpu_hash = bytes(gpu_hashes[i])
        if cpu_hash != gpu_hash:
            mismatches += 1
            print(f"    MISMATCH cand {i} (t_rows={tr}, t_cols={tc}):")
            print(f"      CPU: {cpu_hash.hex()}")
            print(f"      GPU: {gpu_hash.hex()}")
        else:
            print(f"    cand {i} (t_rows={tr:3d}, t_cols={tc:3d}): {cpu_hash.hex()[:16]}... ✓")
    cpu_dt = time.time() - t0
    print(f"  CPU {len(candidates)} candidates: {cpu_dt*1000:.1f}ms")
    print(f"  speedup: {cpu_dt/gpu_dt:.1f}x")
    if mismatches:
        raise AssertionError(f"{mismatches}/{len(candidates)} GPU hashes mismatch CPU")


def _bench_pool_shape() -> None:
    """Bench at live pool shape with a larger batch to gauge throughput."""
    import time

    from .mining_config import (
        MiningConfiguration, PeriodicPattern, compute_job_key,
    )
    from .proof_builder import derive_AB_from_seed, merkle_root_keyed
    import blake3

    m, n, k, r = 131072, 131072, 4096, 128
    h, w = 2, 64
    mc = MiningConfiguration(
        common_dim=k, rank=r, mma_type=0,
        rows_pattern=PeriodicPattern.from_list([0, 32]),
        cols_pattern=PeriodicPattern.from_list(list(range(64))),
    )
    header = bytes(range(76))
    job_key = compute_job_key(header, mc)
    miner_seed = bytes(32)
    print(f"  deriving A, B at pool shape... ", end="", flush=True)
    t0 = time.time()
    A, B = derive_AB_from_seed(miner_seed, m, n, k)
    print(f"{time.time() - t0:.1f}s")
    print(f"  hashing roots... ", end="", flush=True)
    t0 = time.time()
    hash_a = merkle_root_keyed(A, job_key)
    hash_b = merkle_root_keyed(B, job_key)
    print(f"{time.time() - t0:.1f}s")
    b_noise_seed = blake3.blake3(job_key + hash_b).digest()
    a_noise_seed = blake3.blake3(b_noise_seed + hash_a).digest()

    gpu = JackpotGpu(h=h, w=w, r=r)
    print(f"  set_job (pool shape; GPU noise derivation)... ", end="", flush=True)
    t0 = time.time()
    gpu.set_job(A, B, [0, 32], list(range(64)),
                (b_noise_seed, a_noise_seed), a_noise_seed)
    print(f"{time.time() - t0:.1f}s")

    for batch in [256, 1024, 4096, 16384]:
        # Random valid offsets for a probe
        t_rows = np.array([(i % 65536) * 64 % m for i in range(batch)], dtype=np.int32)
        t_cols = np.array([(i % 2048) * 64 for i in range(batch)], dtype=np.int32)
        # warm-up
        gpu.evaluate_batch(t_rows[:8], t_cols[:8])
        t0 = time.time()
        gpu.evaluate_batch(t_rows, t_cols)
        dt = time.time() - t0
        print(f"  batch={batch:>6}: {dt*1000:7.1f}ms  =>  "
              f"{batch/dt:>10,.0f} cand/s")


if __name__ == "__main__":
    print("== correctness ==")
    _selftest_against_cpu()
    print("\n== pool-shape throughput ==")
    _bench_pool_shape()
