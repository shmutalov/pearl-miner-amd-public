"""Construct A, B^T matrices the miner commits to in PlainProof, and compute
their BLAKE3-keyed Merkle roots (= ``hash_a``, ``hash_b`` in
``PublicProofParams``).

Pearl gives the miner complete freedom over the contents of A and B. We
generate both deterministically from a 32-byte miner-chosen seed so that:
  - Recovering them only needs the seed, no on-disk storage.
  - Each (job_key, miner_seed) pair yields a unique (A, B) pair, so the
    search space across noise seeds is meaningful.

Layout:
  - A is ``m × k`` int8, row-major. Row i = bytes ``[i*k, (i+1)*k)``.
  - B is ``n × k`` int8 (stored as B^T in the proof — same row-major layout
    over the n "rows" each of k bytes; "row j of B^T" = "column j of B").
  - Values are in ``[-64, 63]`` (signed 7-bit), so that
    ``Int7xInt7ToInt32`` accumulation cannot overflow over k ≤ 2^18.

``hash_a`` and ``hash_b`` are standard BLAKE3-keyed hashes of the full
matrix bytes, with ``job_key`` as the 32-byte key. (BLAKE3's tree structure
*is* the Merkle tree; the partial-proof building for selected rows comes
in a later milestone.)
"""
from __future__ import annotations

import struct
import time
from dataclasses import dataclass

import blake3
import numpy as np

from .mining_config import HEADER_SERIALIZED_SIZE, MiningConfiguration, compute_job_key


def derive_matrix(seed: bytes, rows: int, cols: int,
                  domain_tag: bytes = b"PEARL_MATRIX") -> np.ndarray:
    """Expand a 32-byte miner seed into a ``rows × cols`` int8 matrix in
    [-64, 63] using BLAKE3 XOF mode.

    The domain tag is mixed in so that the same seed can produce both A and
    B without collisions: pass different ``domain_tag`` values (default vs.
    a caller-supplied override) for the two matrices.

    Output is **row-major** flat ``np.int8`` array reshaped to ``(rows, cols)``.
    """
    if len(seed) != 32:
        raise ValueError(f"seed must be 32 bytes, got {len(seed)}")
    n_bytes = rows * cols
    raw = blake3.blake3(domain_tag + seed).digest(length=n_bytes)
    # Treat each byte as a 7-bit signed value: mask off the top bit, then
    # if bit 6 was set, subtract 128 — i.e. arithmetic sign-extend from
    # bit 6. Done branchlessly via numpy:
    arr = np.frombuffer(raw, dtype=np.uint8)
    # value = (byte & 0x7F) - ((byte & 0x40) << 1)
    low7 = (arr & 0x7F).astype(np.int16)
    sign = ((arr & 0x40).astype(np.int16) << 1)  # 0x80 if bit 6 set else 0
    signed = (low7 - sign).astype(np.int8)
    return signed.reshape(rows, cols)


def derive_AB_from_seed(miner_seed: bytes, m: int, n: int, k: int
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: derive both A (m×k) and B (n×k) from a single 32-byte
    miner seed using different domain tags."""
    A = derive_matrix(miner_seed, m, k, domain_tag=b"PEARL_MATRIX_A")
    B = derive_matrix(miner_seed, n, k, domain_tag=b"PEARL_MATRIX_B")
    return A, B


def merkle_root_keyed(matrix: np.ndarray, job_key: bytes) -> bytes:
    """BLAKE3-keyed hash of a contiguous int8 matrix's bytes.

    BLAKE3's internal tree IS the Merkle tree pearl uses; the root of
    BLAKE3-keyed(matrix.tobytes(), key=job_key) is the same value that
    ``MerkleTree::root()`` returns in ``reference/pearl-blake3``.
    """
    if matrix.dtype != np.int8:
        raise ValueError(f"matrix must be int8, got {matrix.dtype}")
    if len(job_key) != 32:
        raise ValueError(f"job_key must be 32 bytes, got {len(job_key)}")
    data = matrix.tobytes() if matrix.flags["C_CONTIGUOUS"] else np.ascontiguousarray(matrix).tobytes()
    return blake3.blake3(data, key=job_key).digest()


@dataclass
class JobMatrices:
    """A, B + their merkle roots, ready for noisy-GEMM + share assembly."""

    A: np.ndarray            # m × k int8
    B: np.ndarray            # n × k int8 (stored as B; merkle hashed as B^T = B's rows)
    miner_seed: bytes        # 32 bytes
    hash_a: bytes            # 32 bytes
    hash_b: bytes            # 32 bytes
    job_key: bytes           # 32 bytes (= blake3(header || mining_config))
    m: int
    n: int
    k: int

    def commitment_hash(self) -> tuple[bytes, bytes]:
        """``(b_noise_seed, a_noise_seed)`` per ``commitment_hash`` in
        ``PublicProofParams``."""
        b_noise = blake3.blake3(self.job_key + self.hash_b).digest()
        a_noise = blake3.blake3(b_noise + self.hash_a).digest()
        return b_noise, a_noise


def build_job_matrices(miner_seed: bytes,
                       header_bytes: bytes,
                       mining_config: MiningConfiguration,
                       m: int, n: int,
                       A: np.ndarray | None = None,
                       B: np.ndarray | None = None) -> JobMatrices:
    """End-to-end: from miner seed + pool's (header, mining_config, m, n)
    produce the committed A, B plus their merkle roots and the job_key.

    If ``A`` and ``B`` are passed (e.g. derived once via the GPU path and
    cached across jobs), the expensive ``derive_AB_from_seed`` call is
    skipped — at pool shape that saves ~5 s per job change.
    """
    if len(header_bytes) != HEADER_SERIALIZED_SIZE:
        raise ValueError(f"header must be {HEADER_SERIALIZED_SIZE} bytes")
    k = mining_config.common_dim
    if A is None or B is None:
        A, B = derive_AB_from_seed(miner_seed, m, n, k)
    job_key = compute_job_key(header_bytes, mining_config)
    hash_a = merkle_root_keyed(A, job_key)
    hash_b = merkle_root_keyed(B, job_key)
    return JobMatrices(A=A, B=B, miner_seed=miner_seed,
                       hash_a=hash_a, hash_b=hash_b, job_key=job_key,
                       m=m, n=n, k=k)


# ---------------------------------------------------------------------------- #
# Selftest                                                                     #
# ---------------------------------------------------------------------------- #

def _selftest_small() -> None:
    """Tiny shape — fast unit-style checks."""
    seed = bytes(range(32))
    m, n, k = 64, 128, 256
    A = derive_matrix(seed, m, k, b"PEARL_MATRIX_A")
    B = derive_matrix(seed, n, k, b"PEARL_MATRIX_B")
    assert A.shape == (m, k) and A.dtype == np.int8
    assert B.shape == (n, k) and B.dtype == np.int8
    assert A.min() >= -64 and A.max() <= 63
    assert B.min() >= -64 and B.max() <= 63
    # Deterministic across calls:
    A2 = derive_matrix(seed, m, k, b"PEARL_MATRIX_A")
    assert np.array_equal(A, A2)
    # Distinct under different domain tags:
    assert not np.array_equal(A, derive_matrix(seed, m, k, b"PEARL_MATRIX_B"))
    # Merkle root depends on key:
    k1 = bytes(range(32))
    k2 = bytes(range(31, -1, -1))
    h1 = merkle_root_keyed(A, k1)
    h2 = merkle_root_keyed(A, k2)
    assert h1 != h2
    print(f"  small: A {A.shape}, B {B.shape}, min(A)={A.min()} max(A)={A.max()}, "
          f"keyed-root[:8]={h1[:8].hex()}")


def _selftest_pool_shape() -> None:
    """Live pool shape: m=n=131072, k=4096. Real timings on the user's box."""
    m, n, k = 131072, 131072, 4096
    seed = bytes.fromhex("c0ffee" * 10 + "abcd")  # 64 hex chars = 32 bytes
    assert len(seed) == 32, len(seed)
    print(f"  pool-shape: generating A ({m*k/2**20:.0f} MiB) and B ({n*k/2**20:.0f} MiB)...")
    t0 = time.time()
    A = derive_matrix(seed, m, k, b"PEARL_MATRIX_A")
    t1 = time.time()
    B = derive_matrix(seed, n, k, b"PEARL_MATRIX_B")
    t2 = time.time()
    print(f"  pool-shape: A in {t1-t0:.2f}s, B in {t2-t1:.2f}s "
          f"({(m*k)/(t1-t0)/1e9:.2f} GB/s + {(n*k)/(t2-t1)/1e9:.2f} GB/s)")
    print(f"  pool-shape: A[:3, :8] = {A[:3, :8].tolist()}")

    # Roots
    key = bytes.fromhex("00" * 32)
    t3 = time.time()
    ra = merkle_root_keyed(A, key)
    t4 = time.time()
    rb = merkle_root_keyed(B, key)
    t5 = time.time()
    print(f"  pool-shape: hash_a in {t4-t3:.2f}s, hash_b in {t5-t4:.2f}s "
          f"({(m*k)/(t4-t3)/1e9:.2f} GB/s + {(n*k)/(t5-t4)/1e9:.2f} GB/s)")
    print(f"  pool-shape: hash_a={ra.hex()}")
    print(f"  pool-shape: hash_b={rb.hex()}")


if __name__ == "__main__":
    _selftest_small()
    _selftest_pool_shape()
    print("all selftests OK")
