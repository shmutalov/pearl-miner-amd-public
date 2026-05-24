"""Offline end-to-end miner check.

Runs the full pipeline at SMALL shape so the pure-Python path completes in
seconds. Exercises:
  - JobMatrices derivation (A, B from miner_seed)
  - BLAKE3-keyed Merkle layers for A and B^T
  - Candidate search at a loose target
  - PlainProof assembly via merkle_proof.get_multileaf_proof
  - bincode encoding via plain_proof_codec
  - Round-trip decoding to confirm wire bytes are well-formed

No pool connection. Useful as a smoke test for changes to any pipeline
stage; the byte-level layer remains identical between this small shape and
the pool's 131072² shape (only the *size* of A, B, Merkle layers differ).
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import blake3
import numpy as np

from pearl_amd.candidate_search import search_candidate  # noqa: E402
from pearl_amd.merkle_proof import build_merkle_tree  # noqa: E402
from pearl_amd.miner import build_plain_proof  # noqa: E402
from pearl_amd.mining_config import (  # noqa: E402
    MiningConfiguration, PeriodicPattern, compute_job_key,
)
from pearl_amd.plain_proof_codec import PlainProof  # noqa: E402
from pearl_amd.proof_builder import (  # noqa: E402
    JobMatrices, derive_AB_from_seed, merkle_root_keyed,
)


def main() -> None:
    # Small-shape configuration. Same protocol bytes, just dialed down.
    m, n, k, r = 256, 256, 256, 64
    rows_indices = [0, 32]   # 2 rows per group
    cols_indices = list(range(64))  # 64 cols per group
    mc = MiningConfiguration(
        common_dim=k, rank=r, mma_type=0,
        rows_pattern=PeriodicPattern.from_list(rows_indices),
        cols_pattern=PeriodicPattern.from_list(cols_indices),
    )
    # Synthetic 76-byte header (would normally come from mining.notify)
    header = bytes(range(76))
    job_key = compute_job_key(header, mc)
    print(f"job_key = {job_key.hex()}")

    miner_seed = bytes(32)
    print(f"\n[1/5] deriving A ({m}x{k}) and B ({n}x{k})... ", end="", flush=True)
    t0 = time.time()
    A, B = derive_AB_from_seed(miner_seed, m, n, k)
    print(f"{time.time() - t0:.2f}s, "
          f"A min={A.min()} max={A.max()}, B min={B.min()} max={B.max()}")

    print(f"[2/5] hashing roots... ", end="", flush=True)
    t0 = time.time()
    hash_a = merkle_root_keyed(A, job_key)
    hash_b = merkle_root_keyed(B, job_key)
    print(f"{time.time() - t0:.2f}s")
    print(f"        hash_a = {hash_a.hex()}")
    print(f"        hash_b = {hash_b.hex()}")

    print(f"[3/5] building full Merkle layers for A and B... ", end="", flush=True)
    t0 = time.time()
    A_layers = build_merkle_tree(A.tobytes(), job_key)
    B_layers = build_merkle_tree(B.tobytes(), job_key)
    print(f"{time.time() - t0:.2f}s, A leaves={len(A_layers[0])}, B leaves={len(B_layers[0])}")
    # Sanity: tree root must equal merkle_root_keyed
    assert A_layers[-1][0] == hash_a, "A Merkle layers don't reach hash_a"
    assert B_layers[-1][0] == hash_b, "B Merkle layers don't reach hash_b"

    jm = JobMatrices(A=A, B=B, miner_seed=miner_seed,
                     hash_a=hash_a, hash_b=hash_b, job_key=job_key,
                     m=m, n=n, k=k)

    # Loose target: ~1 leading zero bit → first attempt almost always hits.
    target = 1 << 255
    print(f"\n[4/5] searching for candidate with target ≤ 2^255 (~1 lz bit)...")
    t0 = time.time()
    hit, attempts, dt = search_candidate(jm, mc, target, max_attempts=100)
    if hit is None:
        print(f"  no hit in {attempts} attempts ({dt:.2f}s) — should not happen at this target")
        sys.exit(1)
    print(f"  hit in {attempts} attempts ({dt*1000:.0f}ms): t_rows={hit.t_rows}, "
          f"t_cols={hit.t_cols}")
    print(f"  hash_jackpot = {hit.hash_jackpot.hex()}")

    print(f"\n[5/5] building + encoding PlainProof... ", end="", flush=True)
    t0 = time.time()
    proof = build_plain_proof(jm, mc, A_layers, B_layers, hit)
    proof_bytes = proof.encode()
    dt = time.time() - t0
    print(f"{dt*1000:.0f}ms, {len(proof_bytes)} bytes")

    # Round-trip: decode then re-encode, must match byte-for-byte
    decoded = PlainProof.decode(proof_bytes)
    assert decoded.encode() == proof_bytes, "PlainProof round-trip mismatch"
    print(f"  byte round-trip OK")

    # Cross-check the decoded proof
    assert decoded.m == m and decoded.n == n and decoded.k == k
    assert decoded.noise_rank == r
    assert decoded.a.row_indices == list(hit.a_rows_indices)
    assert decoded.bt.row_indices == list(hit.b_cols_indices)
    assert decoded.a.proof.root == hash_a
    assert decoded.bt.proof.root == hash_b
    print(f"  semantic round-trip OK (matches m/n/k/rank, row_indices, roots)")

    print(f"\nDONE. PlainProof is wire-ready: base64 prefix = "
          f"{__import__('base64').b64encode(proof_bytes).decode()[:60]}...")


if __name__ == "__main__":
    main()
