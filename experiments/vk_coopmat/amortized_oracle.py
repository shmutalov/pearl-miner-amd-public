"""Prove the amortized-GEMM decomposition is bit-identical to the canonical
per-candidate ``evaluate_candidate`` — in pure numpy, before any GPU code.

The claim under test (the whole basis for the coopmat rewrite):

    noise_a[row][l] depends only on `row` and `l`  (not on the candidate's cols)
    noise_b[col][l] depends only on `col` and `l`  (not on the candidate's rows)

    => PA = (A + noise_a)  is a single (m x k) int8 matrix, global per job
       PB = (B + noise_b)  is a single (n x k) int8 matrix, global per job

    and a candidate's hash is obtained by gathering its 2 rows / 64 cols out of
    PA / PB, doing the per-r-slice int32 GEMM, XOR-reducing each tile, folding
    via rotl13 into jackpot_msg, and BLAKE3-keying it.

If every sampled candidate's hash matches evaluate_candidate(), the math is
proven and the GPU phases just have to reproduce these three steps.

Run:  python amortized_oracle.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import blake3  # noqa: E402
from pearl_amd.mining_config import (  # noqa: E402
    MiningConfiguration, PeriodicPattern, compute_job_key)
from pearl_amd.proof_builder import (  # noqa: E402
    derive_AB_from_seed, merkle_root_keyed, JobMatrices)
from pearl_amd.candidate_search import enumerate_valid_offsets  # noqa: E402
from pearl_amd.jackpot import (  # noqa: E402
    SEED_LABEL_A, SEED_LABEL_B, JACKPOT_SIZE, LROT_PER_TILE,
    generate_uniform_random_matrix, generate_permutation_matrix,
    compute_noise_for_indices, evaluate_candidate, _rotl32)


def build_global_PA_PB(A, B, k, r, commitment_hash):
    """Build the two global noised matrices PA (m x k) and PB (n x k) as int8.

    Returns (PA_i8, PB_i8, e_ar_t, e_bl, e_al_full, e_br_full) — the last four
    are the global noise tables the GPU Phase A will consume.
    """
    m, n = A.shape[0], B.shape[0]
    b_noise_seed, a_noise_seed = commitment_hash

    # Global per-row / per-col uniform tables (the kernel's EAL (m,r), EBR (n,r)).
    e_al_full = generate_uniform_random_matrix(SEED_LABEL_A, a_noise_seed,
                                               list(range(m)), r)          # (m, r) i8
    e_br_full = generate_uniform_random_matrix(SEED_LABEL_B, b_noise_seed,
                                               list(range(n)), r)          # (n, r) i8
    # Global permutation tables (the kernel's EAR (k,2), EBL (k,2)).
    e_ar_t = generate_permutation_matrix(SEED_LABEL_A, a_noise_seed, k, r)  # (k, 2) u32
    e_bl = generate_permutation_matrix(SEED_LABEL_B, b_noise_seed, k, r)    # (k, 2) u32

    # noise_a[row][l] = e_al[row][e_ar_t[l,0]] - e_al[row][e_ar_t[l,1]]
    noise_a = (e_al_full[:, e_ar_t[:, 0]].astype(np.int32)
               - e_al_full[:, e_ar_t[:, 1]].astype(np.int32))              # (m, k)
    noise_b = (e_br_full[:, e_bl[:, 0]].astype(np.int32)
               - e_br_full[:, e_bl[:, 1]].astype(np.int32))                # (n, k)

    PA32 = A.astype(np.int32) + noise_a
    PB32 = B.astype(np.int32) + noise_b

    # The int7-input assumption: A,B in [-64,63], noise in [-63,63] => sum fits
    # int8 exactly, so the kernel's `& 0xFF` wrap is a no-op and PA_i8 == PA32.
    a_lo, a_hi = int(PA32.min()), int(PA32.max())
    b_lo, b_hi = int(PB32.min()), int(PB32.max())
    print(f"  PA range [{a_lo}, {a_hi}]  PB range [{b_lo}, {b_hi}] "
          f"(must be within [-128,127])")
    assert -128 <= a_lo and a_hi <= 127, "PA overflows int8 -> int7 assumption broken"
    assert -128 <= b_lo and b_hi <= 127, "PB overflows int8 -> int7 assumption broken"

    PA_i8 = PA32.astype(np.int8)
    PB_i8 = PB32.astype(np.int8)
    # Round-trip: int8 store must equal the int32 value used by compute_jackpot.
    assert np.array_equal(PA_i8.astype(np.int32), PA32)
    assert np.array_equal(PB_i8.astype(np.int32), PB32)
    return PA_i8, PB_i8, e_ar_t, e_bl, e_al_full, e_br_full


def amortized_hash(PA_i8, PB_i8, rows, cols, k, r, a_noise_seed):
    """Reproduce evaluate_candidate from the global PA/PB via gather + per-slice
    int32 GEMM + XOR-reduce + rotl13 fold + BLAKE3-keyed."""
    n_iters = k // r
    pa = PA_i8[rows].astype(np.int32)        # (h, k)
    pb = PB_i8[cols].astype(np.int32)        # (w, k)
    jackpot_msg = np.zeros(JACKPOT_SIZE, dtype=np.uint32)
    for it in range(n_iters):
        sl = slice(it * r, (it + 1) * r)
        G = pa[:, sl] @ pb[:, sl].T          # (h, w) int32
        xored = np.bitwise_xor.reduce(G.flatten().view(np.uint32))
        tid = it % JACKPOT_SIZE
        jackpot_msg[tid] = _rotl32(int(jackpot_msg[tid]), LROT_PER_TILE) ^ int(xored)
    return blake3.blake3(jackpot_msg.tobytes(), key=a_noise_seed).digest()


def main() -> int:
    # Small shape (fast end-to-end), same family as the live pool shape.
    m, n, k, r = 1024, 1024, 256, 64
    if "--pool" in sys.argv:
        m, n, k, r = 131072, 131072, 4096, 128  # live pool shape
    mc = MiningConfiguration(
        common_dim=k, rank=r, mma_type=0,
        rows_pattern=PeriodicPattern.from_list([0, 32]),
        cols_pattern=PeriodicPattern.from_list(list(range(64))),
    )
    header = bytes(76)
    job_key = compute_job_key(header, mc)
    A, B = derive_AB_from_seed(bytes(32), m, n, k)
    hash_a = merkle_root_keyed(A, job_key)
    hash_b = merkle_root_keyed(B, job_key)
    jm = JobMatrices(A=A, B=B, miner_seed=bytes(32), hash_a=hash_a, hash_b=hash_b,
                     job_key=job_key, m=m, n=n, k=k)
    commitment_hash = jm.commitment_hash()
    _, a_noise_seed = commitment_hash

    print(f"shape m={m} n={n} k={k} r={r}  n_iters={k//r}")
    t0 = time.time()
    PA_i8, PB_i8, e_ar_t, e_bl, e_al_full, e_br_full = build_global_PA_PB(
        A, B, k, r, commitment_hash)
    print(f"  built global PA{PA_i8.shape}, PB{PB_i8.shape} in {time.time()-t0:.2f}s")

    # Cross-check the decomposition directly: per-candidate compute_noise must
    # equal the gathered rows/cols of the global noise tables.
    rows0 = list(mc.rows_pattern.indices_with_offset(0))
    cols0 = list(mc.cols_pattern.indices_with_offset(0))
    na0, nb0 = compute_noise_for_indices(k, r, commitment_hash, rows0, cols0)
    noise_a_rows = (e_al_full[rows0][:, e_ar_t[:, 0]].astype(np.int32)
                    - e_al_full[rows0][:, e_ar_t[:, 1]].astype(np.int32)).astype(np.int8)
    noise_b_cols = (e_br_full[cols0][:, e_bl[:, 0]].astype(np.int32)
                    - e_br_full[cols0][:, e_bl[:, 1]].astype(np.int32)).astype(np.int8)
    assert np.array_equal(na0, noise_a_rows), "noise_a decomposition mismatch"
    assert np.array_equal(nb0, noise_b_cols), "noise_b decomposition mismatch"
    print("  decomposition check: per-candidate noise == global-table gather  OK")

    # Sample candidates across the valid offset grid and compare hashes.
    row_offsets = list(enumerate_valid_offsets(mc.rows_pattern, m))
    col_offsets = list(enumerate_valid_offsets(mc.cols_pattern, n))
    print(f"  valid offsets: rows={len(row_offsets)} cols={len(col_offsets)}")

    import random
    random.seed(12345)
    samples = []
    # include the corners + random interior
    for tr in (row_offsets[0], row_offsets[-1]):
        for tc in (col_offsets[0], col_offsets[-1]):
            samples.append((tr, tc))
    for _ in range(40):
        samples.append((random.choice(row_offsets), random.choice(col_offsets)))

    ok = 0
    for (tr, tc) in samples:
        rows = list(mc.rows_pattern.indices_with_offset(tr))
        cols = list(mc.cols_pattern.indices_with_offset(tc))
        ref_val, ref_hash = evaluate_candidate(
            A, B, rows, cols, commitment_hash, a_noise_seed, k, r)
        got_hash = amortized_hash(PA_i8, PB_i8, rows, cols, k, r, a_noise_seed)
        if got_hash != ref_hash:
            print(f"  MISMATCH at (t_r={tr}, t_c={tc}):")
            print(f"    ref={ref_hash.hex()}")
            print(f"    got={got_hash.hex()}")
            return 1
        ok += 1
    print(f"  {ok}/{len(samples)} candidates: amortized hash == evaluate_candidate  OK")
    print("AMORTIZED DECOMPOSITION PROVEN (bit-identical).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
