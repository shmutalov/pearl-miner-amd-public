"""Confirm build_plain_proof is identical whether A/B are serialized inside
(the old per-share jm.A.tobytes()) or passed in precomputed (the per-round
cache the coopmat pipeline now uses). GPU-free; small CPU Merkle build.
"""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from pearl_amd.mining_config import MiningConfiguration, PeriodicPattern, compute_job_key
from pearl_amd.proof_builder import derive_AB_from_seed, merkle_root_keyed, JobMatrices
from pearl_amd.merkle_proof import build_merkle_tree
from pearl_amd.candidate_search import Candidate
from pearl_amd.miner import build_plain_proof


def main() -> int:
    m = n = 256
    k = 256
    mc = MiningConfiguration(common_dim=k, rank=64, mma_type=0,
        rows_pattern=PeriodicPattern.from_list([0, 32]),
        cols_pattern=PeriodicPattern.from_list(list(range(64))))
    job_key = compute_job_key(bytes(76), mc)
    A, B = derive_AB_from_seed(bytes(32), m, n, k)
    jm = JobMatrices(A=A, B=B, miner_seed=bytes(32),
                     hash_a=merkle_root_keyed(A, job_key), hash_b=merkle_root_keyed(B, job_key),
                     job_key=job_key, m=m, n=n, k=k)
    A_layers = build_merkle_tree(A.tobytes(), job_key)
    B_layers = build_merkle_tree(B.tobytes(), job_key)

    cand = Candidate(t_rows=0, t_cols=0,
        a_rows_indices=list(mc.rows_pattern.indices_with_offset(0)),
        b_cols_indices=list(mc.cols_pattern.indices_with_offset(0)),
        hash_jackpot=bytes(32), target_value=0)

    p_compute = build_plain_proof(jm, mc, A_layers, B_layers, cand)
    p_cached = build_plain_proof(jm, mc, A_layers, B_layers, cand,
                                 a_bytes=jm.A.tobytes(), b_bytes=jm.B.tobytes())
    ok = p_compute.encode() == p_cached.encode()
    print(f"cached-bytes proof identical to computed: {ok} "
          f"({len(p_compute.encode())} B)")
    print("PROOF CACHE EQUIVALENT" if ok else "PROOF CACHE **MISMATCH**")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
