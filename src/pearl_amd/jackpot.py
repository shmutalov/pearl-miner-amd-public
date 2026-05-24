"""Per-candidate noisy-GEMM + jackpot hash, ported from
``reference/zk-pow/src/circuit/pearl_noise.rs`` and ``.../jackpot/helper.rs``.

This is the inner loop of Pearl mining: given the miner-chosen A, B matrices
(committed via ``hash_a``, ``hash_b``) and a candidate tile position
``(t_rows, t_cols)``, build the noise matrices, run the noisy-GEMM on the
selected strips, hash the result, and compare against the share target.

Pure-Python reference — correctness over speed. OpenCL acceleration is the
next milestone; for now we keep this readable so it can serve as the test
oracle for the GPU kernel.

Wire-protocol bridge:
  - ``compute_noise``-equivalent inputs come from ``JobMatrices.commitment_hash()``
    plus the touched ``a_rows_indices`` (= ``rows_pattern.indices_with_offset(t_rows)``)
    and ``b_cols_indices`` (= ``cols_pattern.indices_with_offset(t_cols)``).
  - ``target`` is the decoded ``share_nbits`` (see :func:`mining_config.nbits_hex_to_target`).
"""
from __future__ import annotations

import struct

import blake3
import numpy as np


# ---------------------------------------------------------------------------- #
# Pearl protocol constants                                                     #
# ---------------------------------------------------------------------------- #

NOISE_RANGE = 128                  # number of distinct *signed* noise values
IDXS_PER_COL = 2                   # +/-1 per col in sparse perm
UNIFORM_NOISE_RANGE = NOISE_RANGE // IDXS_PER_COL                # 64
ZERO_POINT_TRANSLATION = UNIFORM_NOISE_RANGE // 2                # 32
RANGE_MASK = UNIFORM_NOISE_RANGE - 1                             # 0x3F
BLAKE3_DIGEST_SIZE = 32

JACKPOT_SIZE = 16                  # number of u32 words in jackpot message
LROT_PER_TILE = 13                 # left-rotate amount applied per outer-loop iter

SEED_LABEL_A = b"A_tensor".ljust(32, b"\x00")
SEED_LABEL_B = b"B_tensor".ljust(32, b"\x00")


# ---------------------------------------------------------------------------- #
# Helpers                                                                      #
# ---------------------------------------------------------------------------- #

def _get_random_hash(index: int, seed: bytes, key: bytes, prepend_index: int) -> bytes:
    """Mirrors ``get_random_hash`` in ``pearl_noise.rs``.

    Builds a 64-byte message = 32 bytes of (zero-padded i32 slots, with
    ``(index+1)`` placed at offset ``prepend_index*4``) + 32-byte seed, then
    ``blake3-keyed(msg, key=key)`` → 32-byte digest.
    """
    assert len(seed) == 32 and len(key) == 32
    msg = bytearray(64)
    msg[prepend_index * 4: prepend_index * 4 + 4] = struct.pack("<i", index + 1)
    msg[32:64] = seed
    return blake3.blake3(bytes(msg), key=key).digest()


def _mul_hi_u32(a: int, b: int) -> int:
    return ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF) >> 32) & 0xFFFFFFFF


def _rotl32(x: int, n: int) -> int:
    n %= 32
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


# ---------------------------------------------------------------------------- #
# Noise matrix generation                                                      #
# ---------------------------------------------------------------------------- #

def generate_uniform_random_matrix(seed: bytes, key: bytes,
                                   row_indices: list[int],
                                   num_cols: int) -> np.ndarray:
    """For each ``row_idx`` in ``row_indices``, produce ``num_cols`` i8 values
    in ``[-32, 31]``. Bytes are drawn from BLAKE3-keyed hashes; each row is
    contiguous in a flat virtual byte stream indexed by ``row_idx * num_cols``.
    """
    out = np.empty((len(row_indices), num_cols), dtype=np.int8)
    for i, row_idx in enumerate(row_indices):
        start = row_idx * num_cols
        end = start + num_cols
        block_lo = start // BLAKE3_DIGEST_SIZE
        block_hi = (end + BLAKE3_DIGEST_SIZE - 1) // BLAKE3_DIGEST_SIZE
        row_bytes = bytearray()
        for block in range(block_lo, block_hi):
            row_bytes += _get_random_hash(block, seed, key, prepend_index=0)
        offset_in_stream = start - block_lo * BLAKE3_DIGEST_SIZE
        for j in range(num_cols):
            b = row_bytes[offset_in_stream + j]
            out[i, j] = (b & RANGE_MASK) - ZERO_POINT_TRANSLATION
    return out


def generate_permutation_matrix(seed: bytes, key: bytes,
                                k: int, noise_rank: int) -> np.ndarray:
    """``k`` rows of ``[first_idx, second_idx]`` u32 pairs. Each row encodes a
    sparse permutation: ``out[i] = vec[first_idx] - vec[second_idx]``. Result
    dtype is ``np.uint32`` with shape ``(k, 2)``.
    """
    rank_mask = noise_rank - 1
    bytes_per_line = 4
    lines_per_hash = BLAKE3_DIGEST_SIZE // bytes_per_line
    out = np.empty((k, 2), dtype=np.uint32)
    for line_block in range((k + lines_per_hash - 1) // lines_per_hash):
        rh = _get_random_hash(line_block, seed, key, prepend_index=1)
        for j in range(lines_per_hash):
            row = line_block * lines_per_hash + j
            if row >= k:
                break
            v = struct.unpack_from("<I", rh, j * 4)[0]
            first_idx = v & rank_mask
            second_idx = first_idx ^ (1 + _mul_hi_u32(noise_rank - 1, v))
            out[row, 0] = first_idx
            out[row, 1] = second_idx
    return out


def matvec_sparse_perm(perm: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """``result[i] = vec[perm[i, 0]] - vec[perm[i, 1]]`` as int8. ``vec``
    must be int8 of length ``noise_rank``; ``perm`` shape ``(k, 2)`` uint32."""
    pos = vec[perm[:, 0]].astype(np.int32)
    neg = vec[perm[:, 1]].astype(np.int32)
    diff = pos - neg
    return diff.astype(np.int8)


def compute_noise_for_indices(k: int, noise_rank: int,
                              commitment_hash: tuple[bytes, bytes],
                              a_rows_indices: list[int],
                              b_cols_indices: list[int],
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(noise_a, noise_b)`` strips:
      - ``noise_a``: shape ``(len(a_rows_indices), k)`` int8
      - ``noise_b``: shape ``(len(b_cols_indices), k)`` int8

    Mirrors ``compute_noise_for_indices`` in ``pearl_noise.rs`` exactly.
    """
    if noise_rank <= 0 or (noise_rank & (noise_rank - 1)) != 0:
        raise ValueError("noise_rank must be a power of two")
    if noise_rank % BLAKE3_DIGEST_SIZE != 0:
        raise ValueError("noise_rank must be divisible by 32")
    b_noise_seed, a_noise_seed = commitment_hash

    e_al = generate_uniform_random_matrix(SEED_LABEL_A, a_noise_seed,
                                          a_rows_indices, noise_rank)
    e_ar_t = generate_permutation_matrix(SEED_LABEL_A, a_noise_seed, k, noise_rank)
    e_bl = generate_permutation_matrix(SEED_LABEL_B, b_noise_seed, k, noise_rank)
    e_br_t = generate_uniform_random_matrix(SEED_LABEL_B, b_noise_seed,
                                            b_cols_indices, noise_rank)

    noise_a = np.empty((len(a_rows_indices), k), dtype=np.int8)
    for u in range(len(a_rows_indices)):
        noise_a[u] = matvec_sparse_perm(e_ar_t, e_al[u])
    noise_b = np.empty((len(b_cols_indices), k), dtype=np.int8)
    for v in range(len(b_cols_indices)):
        noise_b[v] = matvec_sparse_perm(e_bl, e_br_t[v])
    return noise_a, noise_b


# ---------------------------------------------------------------------------- #
# Jackpot compute + hash                                                       #
# ---------------------------------------------------------------------------- #

def compute_jackpot(secret_a: np.ndarray, secret_b: np.ndarray,
                    noise_a: np.ndarray, noise_b: np.ndarray,
                    k: int, r: int) -> np.ndarray:
    """Port of ``compute_jackpot`` in ``circuit/chip/jackpot/helper.rs``.

    Inputs:
      - secret_a: (h, k) int8 strips of the touched rows of A
      - secret_b: (w, k) int8 strips of the touched rows of B^T (cols of B)
      - noise_a:  (h, k) int8 noise strips for A
      - noise_b:  (w, k) int8 noise strips for B
      - k: contraction dim
      - r: noise_rank

    Returns the jackpot_msg as ``np.ndarray`` shape ``(JACKPOT_SIZE,)``
    ``np.uint32``, ready to be hashed.
    """
    h, w = secret_a.shape[0], secret_b.shape[0]
    if secret_a.shape[1] != k or secret_b.shape[1] != k:
        raise ValueError("strip width mismatch")
    if noise_a.shape != (h, k) or noise_b.shape != (w, k):
        raise ValueError("noise shape mismatch")

    # Accumulate as int32: each cell = sum over l in [ll-r, ll) of
    # (secret_a[u, l] + noise_a[u, l]) * (secret_b[v, l] + noise_b[v, l]).
    sa = secret_a.astype(np.int32) + noise_a.astype(np.int32)
    sb = secret_b.astype(np.int32) + noise_b.astype(np.int32)

    jackpot_msg = np.zeros(JACKPOT_SIZE, dtype=np.uint32)
    for ll in range(r, k + 1, r):
        # int32 GEMM on the [ll-r, ll) slice; result shape (h, w)
        block = sa[:, ll - r:ll] @ sb[:, ll - r:ll].T
        # XOR all values together (uint32 mod 2^32 — same as int32 mod 2^32)
        xored_tile = np.bitwise_xor.reduce(block.flatten().view(np.uint32))
        tid = ((ll // r) - 1) % JACKPOT_SIZE
        jackpot_msg[tid] = _rotl32(int(jackpot_msg[tid]), LROT_PER_TILE) ^ int(xored_tile)
    return jackpot_msg


def compute_jackpot_hash(jackpot_msg: np.ndarray, a_noise_seed: bytes) -> bytes:
    """``blake3-keyed(jackpot_msg_as_64_le_bytes, key=a_noise_seed)``."""
    if jackpot_msg.shape != (JACKPOT_SIZE,) or jackpot_msg.dtype != np.uint32:
        raise ValueError("jackpot_msg must be (16,) uint32")
    if len(a_noise_seed) != 32:
        raise ValueError("a_noise_seed must be 32 bytes")
    msg_bytes = jackpot_msg.tobytes()  # little-endian by default on x86
    return blake3.blake3(msg_bytes, key=a_noise_seed).digest()


def jackpot_hash_as_target(jackpot_hash: bytes) -> int:
    """The verifier compares ``hash_jackpot`` as a **little-endian** 256-bit
    integer. So we deserialize byte-wise LSB-first.
    """
    return int.from_bytes(jackpot_hash, "little")


# ---------------------------------------------------------------------------- #
# Per-candidate evaluation                                                     #
# ---------------------------------------------------------------------------- #

def evaluate_candidate(A: np.ndarray, B: np.ndarray,
                       a_rows_indices: list[int],
                       b_cols_indices: list[int],
                       commitment_hash: tuple[bytes, bytes],
                       a_noise_seed: bytes,
                       k: int, r: int) -> tuple[int, bytes]:
    """Compute one candidate's ``hash_jackpot`` interpreted as LE-uint256.
    Returns ``(target_value, hash_jackpot_bytes)``.
    """
    secret_a = A[a_rows_indices, :k]
    secret_b = B[b_cols_indices, :k]
    noise_a, noise_b = compute_noise_for_indices(
        k, r, commitment_hash, a_rows_indices, b_cols_indices)
    jp_msg = compute_jackpot(secret_a, secret_b, noise_a, noise_b, k, r)
    jp_hash = compute_jackpot_hash(jp_msg, a_noise_seed)
    return jackpot_hash_as_target(jp_hash), jp_hash


# ---------------------------------------------------------------------------- #
# Selftest                                                                     #
# ---------------------------------------------------------------------------- #

def _selftest_helpers() -> None:
    """Sanity checks on the byte-level helpers."""
    assert SEED_LABEL_A == b"A_tensor\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    assert SEED_LABEL_B == b"B_tensor\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    assert _rotl32(0xDEADBEEF, 13) == ((0xDEADBEEF << 13) | (0xDEADBEEF >> 19)) & 0xFFFFFFFF
    assert _mul_hi_u32(0xFFFFFFFF, 0xFFFFFFFF) == 0xFFFFFFFE  # ≈ 2^32 - 2
    print("  helpers OK")


def _selftest_jackpot_pipeline() -> None:
    """Run one candidate end-to-end at pool shape using fixed test inputs.
    No oracle to compare against without the Rust crate, but we check:
    - The pipeline runs without error.
    - Output sizes are correct.
    - Determinism (same inputs → same output).
    - The output is sensitive to each input.
    """
    from .mining_config import mining_config_from_pool_params, compute_job_key
    from .proof_builder import derive_AB_from_seed
    import time

    # Pool capture from 2026-05-24
    m, n, k, r = 131072, 131072, 4096, 128
    pool_params = {"m": m, "n": n, "k": k, "rank": r,
                   "rows_pattern": [0, 32],
                   "cols_pattern": list(range(64)),
                   "mma_type": "Int7xInt7ToInt32"}
    mc = mining_config_from_pool_params(pool_params)
    header_hex = ("00004020f8d70ee30e785506ed2ddeda7a2b19c583f9f7dce4eeb6e4a620c82570f81c"
                  "1a6f4972e98960f0dc29dd251aff1e75b7e8221d06b065b27591f35003efcfe2c"
                  "4c5fd116af7810318")
    header = bytes.fromhex(header_hex)
    job_key = compute_job_key(header, mc)

    # Synthetic miner side
    miner_seed = bytes(32)  # all zeros for repeatability
    print(f"  generating A({m}x{k}) and B({n}x{k}) from seed... ", end="", flush=True)
    t0 = time.time()
    A, B = derive_AB_from_seed(miner_seed, m, n, k)
    print(f"done in {time.time() - t0:.2f}s")

    # Merkle roots → commitment_hash
    from .proof_builder import merkle_root_keyed
    print(f"  hashing roots... ", end="", flush=True)
    t0 = time.time()
    hash_a = merkle_root_keyed(A, job_key)
    hash_b = merkle_root_keyed(B, job_key)
    print(f"done in {time.time() - t0:.2f}s")

    # commitment_hash = (b_noise_seed, a_noise_seed) per PublicProofParams
    b_noise_seed = blake3.blake3(job_key + hash_b).digest()
    a_noise_seed = blake3.blake3(b_noise_seed + hash_a).digest()
    commitment_hash = (b_noise_seed, a_noise_seed)

    # Candidate tile: rows_pattern + t_rows=0, cols_pattern + t_cols=0
    a_rows_indices = mc.rows_pattern.indices_with_offset(0)
    b_cols_indices = mc.cols_pattern.indices_with_offset(0)
    print(f"  candidate tile: rows={list(a_rows_indices)} "
          f"cols_len={len(b_cols_indices)}")

    print(f"  evaluate candidate (k={k}, h={len(a_rows_indices)}, "
          f"w={len(b_cols_indices)})... ", end="", flush=True)
    t0 = time.time()
    target_val, jp_hash = evaluate_candidate(
        A, B, list(a_rows_indices), list(b_cols_indices),
        commitment_hash, a_noise_seed, k, r)
    dt = time.time() - t0
    print(f"done in {dt*1000:.0f}ms")
    print(f"  jackpot_hash = {jp_hash.hex()}")
    lz = 256 - target_val.bit_length() if target_val > 0 else 256
    print(f"  hash as LE-uint256 has {lz} leading zero bits")

    # Determinism: re-run
    t2, h2 = evaluate_candidate(
        A, B, list(a_rows_indices), list(b_cols_indices),
        commitment_hash, a_noise_seed, k, r)
    assert h2 == jp_hash and t2 == target_val, "non-deterministic!"

    # Sensitivity: change tile, hash should differ
    a_rows_indices2 = mc.rows_pattern.indices_with_offset(64)
    t3, h3 = evaluate_candidate(
        A, B, list(a_rows_indices2), list(b_cols_indices),
        commitment_hash, a_noise_seed, k, r)
    assert h3 != jp_hash, "hash not sensitive to t_rows!"
    print(f"  sensitivity check: differing t_rows -> differing hash ✓")


if __name__ == "__main__":
    _selftest_helpers()
    _selftest_jackpot_pipeline()
    print("all selftests OK")
