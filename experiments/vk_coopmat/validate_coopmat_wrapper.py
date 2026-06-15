"""End-to-end validation of JackpotCoopmat (DLL + ctypes wrapper):

  1. set_job_raw with oracle-derived noise -> search(loose target) -> the
     returned Candidate's hash must match evaluate_candidate AND be < target.
  2. set_job (GPU-derived noise, noise_rank=r) -> search -> same checks. This
     confirms the GPU noise-derivation convention matches the proven oracle.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from pearl_amd.mining_config import MiningConfiguration, PeriodicPattern, compute_job_key
from pearl_amd.proof_builder import derive_AB_from_seed, merkle_root_keyed, JobMatrices
from pearl_amd.jackpot import evaluate_candidate
from pearl_amd.jackpot_coopmat import JackpotCoopmat
from amortized_oracle import build_global_PA_PB


def main() -> int:
    m, n, k, r = 1024, 1024, 256, 64
    mc = MiningConfiguration(common_dim=k, rank=r, mma_type=0,
        rows_pattern=PeriodicPattern.from_list([0, 32]),
        cols_pattern=PeriodicPattern.from_list(list(range(64))))
    job_key = compute_job_key(bytes(76), mc)
    A, B = derive_AB_from_seed(bytes(32), m, n, k)
    jm = JobMatrices(A=A, B=B, miner_seed=bytes(32),
                     hash_a=merkle_root_keyed(A, job_key), hash_b=merkle_root_keyed(B, job_key),
                     job_key=job_key, m=m, n=n, k=k)
    commitment_hash = jm.commitment_hash(); _, a_noise_seed = commitment_hash
    PA, PB, e_ar_t, e_bl, e_al_full, e_br_full = build_global_PA_PB(A, B, k, r, commitment_hash)

    target_lz = 8
    target = 1 << (256 - target_lz)

    jc = JackpotCoopmat(2, 64, r, k)

    def verify(tag):
        cand, attempts, dt = jc.search(mc, target, max_attempts=None)
        if cand is None:
            print(f"  [{tag}] NO HIT in {attempts} attempts ({dt*1000:.0f}ms) -- "
                  f"loosen target?"); return False
        # Cross-check the returned candidate against the CPU oracle.
        ev_val, ev_hash = evaluate_candidate(
            A, B, cand.a_rows_indices, cand.b_cols_indices, commitment_hash, a_noise_seed, k, r)
        ok_hash = (ev_hash == cand.hash_jackpot)
        ok_below = (cand.target_value < target)
        rate = attempts / dt / 1e6 if dt > 0 else 0
        print(f"  [{tag}] hit t_r={cand.t_rows} t_c={cand.t_cols}  "
              f"hash_ok={ok_hash} below_target={ok_below}  "
              f"({attempts} attempts, {rate:.1f}M/s)")
        if not ok_hash:
            print(f"      oracle={ev_hash.hex()}")
            print(f"      gpu   ={cand.hash_jackpot.hex()}")
        return ok_hash and ok_below

    print("1) set_job_raw (oracle noise):")
    jc.set_job_raw(A, B, e_al_full, e_br_full, e_ar_t, e_bl, a_noise_seed)
    ok1 = verify("raw")

    print("2) set_job (GPU noise, noise_rank=r):")
    try:
        jc.set_job(A, B, mc.rows_pattern, mc.cols_pattern, commitment_hash, a_noise_seed)
        ok2 = verify("gpu")
    except Exception as e:
        print(f"  set_job(GPU noise) raised: {e!r}")
        ok2 = False

    jc.close()
    if ok1 and ok2:
        print("COOPMAT WRAPPER CORRECT (both noise paths bit-identical to oracle)")
        return 0
    print("COOPMAT WRAPPER **FAILED**" + ("" if ok1 else " [raw]") + ("" if ok2 else " [gpu]"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
