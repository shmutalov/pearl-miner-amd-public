"""Validate the NJOB double-buffered job slots (jcm_set_job_slot + search_all_stream
job_slot=...). Two DIFFERENT jobs (distinct A/B/noise) are built into slot 0 and
slot 1 simultaneously; searching each slot must return that job's own share set,
proving the slots are independent and don't clobber each other (the guarantee
mine_coopmat_continuous relies on when it prebuilds the next round into the idle
slot while the current round searches the active one).
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from pearl_amd.mining_config import MiningConfiguration, PeriodicPattern, compute_job_key
from pearl_amd.proof_builder import derive_AB_from_seed, merkle_root_keyed, JobMatrices
from pearl_amd.jackpot_coopmat import JackpotCoopmat
from amortized_oracle import build_global_PA_PB

M = N = 1024
K = 256
R = 64
MC = MiningConfiguration(common_dim=K, rank=R, mma_type=0,
    rows_pattern=PeriodicPattern.from_list([0, 32]),
    cols_pattern=PeriodicPattern.from_list(list(range(64))))
TARGET = 1 << (256 - 6)          # loose -> ~128 shares per job
CHUNK = 32                        # 8 tiles -> both ping-pong slots exercised
BIG = 1_000_000


def build_job(seed_byte: int):
    job_key = compute_job_key(bytes(76), MC)
    seed = bytes([seed_byte]) * 32
    A, B = derive_AB_from_seed(seed, M, N, K)
    jm = JobMatrices(A=A, B=B, miner_seed=seed,
                     hash_a=merkle_root_keyed(A, job_key), hash_b=merkle_root_keyed(B, job_key),
                     job_key=job_key, m=M, n=N, k=K)
    ch = jm.commitment_hash(); _, a_noise_seed = ch
    _, _, e_ar_t, e_bl, e_al, e_br = build_global_PA_PB(A, B, K, R, ch)
    return dict(A=A, B=B, e_al=e_al, e_br_t=e_br, e_ar_t=e_ar_t, e_bl=e_bl, ans=a_noise_seed)


def _set(jc, job, slot):
    jc.set_job_raw(job["A"], job["B"], job["e_al"], job["e_br_t"],
                   job["e_ar_t"], job["e_bl"], job["ans"], job_slot=slot)


def _stream(jc, slot):
    return {(c.t_rows, c.t_cols, c.hash_jackpot)
            for c in jc.search_all_stream(MC, TARGET, max_return=BIG, chunk_wg=CHUNK, job_slot=slot)}


def main() -> int:
    jobA, jobB = build_job(0), build_job(7)
    jc = JackpotCoopmat(2, 64, R, K)

    # Ground truth: search_all reads job slot 0, so build each job there alone.
    _set(jc, jobA, 0)
    setA = {(c.t_rows, c.t_cols, c.hash_jackpot)
            for c in jc.search_all(MC, TARGET, max_return=BIG, chunk_wg=CHUNK)[0]}
    _set(jc, jobB, 0)
    setB = {(c.t_rows, c.t_cols, c.hash_jackpot)
            for c in jc.search_all(MC, TARGET, max_return=BIG, chunk_wg=CHUNK)[0]}

    # Both jobs resident at once: slot 0 = A, slot 1 = B.
    _set(jc, jobA, 0)
    _set(jc, jobB, 1)
    a0 = _stream(jc, 0)
    b1 = _stream(jc, 1)
    a0_again = _stream(jc, 0)          # re-read slot 0 after touching slot 1

    ok_distinct = setA != setB and len(setA) > 10 and len(setB) > 10
    ok_a = a0 == setA
    ok_b = b1 == setB
    ok_stable = a0_again == setA       # slot 0 not clobbered by slot 1 search
    print(f"jobs differ        : {ok_distinct} (|A|={len(setA)} |B|={len(setB)})")
    print(f"slot0 == jobA      : {ok_a}")
    print(f"slot1 == jobB      : {ok_b}")
    print(f"slot0 stable       : {ok_stable} (no clobber from slot1)")

    jc.close()
    ok = ok_distinct and ok_a and ok_b and ok_stable
    print("DOUBLE-BUFFER SLOTS INDEPENDENT" if ok else "DOUBLE-BUFFER **FAILED**")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
