"""Validate the double-buffered job slots (jcm_set_job_slot + jcm_search_submit
job-slot arg) and the cross-thread overlap they enable:

  1. isolation : two different jobs live in slots 0 and 1 at once; searching each
                 slot returns exactly that job's share set (no cross-contamination).
  2. overlap   : rebuild the IDLE slot (-> a third job) on a prefetch thread while
                 the main thread searches the ACTIVE slot. The active search must
                 stay correct (not corrupted by the concurrent rebuild), and the
                 rebuilt slot must then search the new job correctly.

Run under the Khronos validation layer with VK_LAYER_VALIDATE_SYNC=1 to also
confirm the concurrent submits raise no SYNC-HAZARD.
"""
from __future__ import annotations
import os, sys, threading
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
JOB_KEY = compute_job_key(bytes(76), MC)
TARGET = 1 << (256 - 6)
BIG = 1_000_000
CHUNK = 32


def build_job(seed_byte: int) -> dict:
    A, B = derive_AB_from_seed(bytes([seed_byte]) * 32, M, N, K)
    jm = JobMatrices(A=A, B=B, miner_seed=bytes([seed_byte]) * 32,
                     hash_a=merkle_root_keyed(A, JOB_KEY), hash_b=merkle_root_keyed(B, JOB_KEY),
                     job_key=JOB_KEY, m=M, n=N, k=K)
    ch = jm.commitment_hash(); _, a_noise_seed = ch
    PA, PB, e_ar_t, e_bl, e_al_full, e_br_full = build_global_PA_PB(A, B, K, R, ch)
    return dict(A=A, B=B, e_al=e_al_full, e_br=e_br_full, e_ar_t=e_ar_t,
                e_bl=e_bl, key=a_noise_seed)


def shares(jc, job_slot) -> set:
    return {(c.t_rows, c.t_cols, c.hash_jackpot)
            for c in jc.search_all_stream(MC, TARGET, max_return=BIG,
                                          chunk_wg=CHUNK, job_slot=job_slot)}


def set_slot(jc, job, slot):
    jc.set_job_raw(job['A'], job['B'], job['e_al'], job['e_br'],
                   job['e_ar_t'], job['e_bl'], job['key'], job_slot=slot)


def main() -> int:
    job0, job1, job2 = build_job(0), build_job(1), build_job(2)
    jc = JackpotCoopmat(2, 64, R, K)

    # Reference share sets (each job built alone into slot 0, swept synchronously).
    set_slot(jc, job0, 0); ref0 = shares(jc, 0)
    set_slot(jc, job1, 0); ref1 = shares(jc, 0)
    set_slot(jc, job2, 0); ref2 = shares(jc, 0)
    print(f"refs: job0={len(ref0)} job1={len(ref1)} job2={len(ref2)} shares "
          f"(distinct={len({len(ref0),len(ref1),len(ref2)})>1 or len(ref0)>0})")

    # 1) Two jobs resident at once, searched independently.
    set_slot(jc, job0, 0)
    set_slot(jc, job1, 1)
    s0, s1 = shares(jc, 0), shares(jc, 1)
    ok_iso = s0 == ref0 and s1 == ref1
    print(f"1 isolation: slot0==job0={s0 == ref0}  slot1==job1={s1 == ref1}  -> {ok_iso}")

    # 2) Rebuild the idle slot (1 -> job2) on a prefetch thread WHILE searching the
    #    active slot 0 (job0). Concurrent vkQueueSubmit from both threads.
    err = []
    def rebuild():
        try:
            set_slot(jc, job2, 1)
        except BaseException as e:
            err.append(e)
    th = threading.Thread(target=rebuild, name="prefetch", daemon=True)
    th.start()
    s0b = shares(jc, 0)          # main-thread search overlaps the slot-1 rebuild
    th.join()
    if err:
        raise err[0]
    s1b = shares(jc, 1)         # slot 1 now holds job2
    ok_overlap = s0b == ref0 and s1b == ref2
    print(f"2 overlap  : active slot0 still job0={s0b == ref0}  "
          f"rebuilt slot1==job2={s1b == ref2}  -> {ok_overlap}")

    jc.close()
    ok = ok_iso and ok_overlap and len(ref0) > 0
    print("DOUBLE-BUFFER CORRECT" if ok else "DOUBLE-BUFFER **FAILED**")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
