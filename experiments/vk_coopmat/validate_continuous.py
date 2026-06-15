"""Validate PearlMiner.mine_coopmat_continuous: the cross-round preflight-overlap
loop. Each round's preflight (here an oracle build into the idle job slot) runs on
a prefetch thread while the previous round searches the active slot; NJOB=2
double-buffers PA/PB/KEY.

_preflight_into is monkeypatched to the oracle path (no OpenCL needed); _submit_hit
is stubbed to (a) assert every submitted share belongs to THAT round's job — so any
slot bleed fails loudly — and (b) record per-round share sets to compare against the
single-job reference. Four rounds over four distinct jobs.
"""
from __future__ import annotations
import os, sys, types, threading
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from pearl_amd.mining_config import MiningConfiguration, PeriodicPattern, compute_job_key
from pearl_amd.proof_builder import derive_AB_from_seed, merkle_root_keyed, JobMatrices
from pearl_amd.jackpot_coopmat import JackpotCoopmat
from pearl_amd.miner import PearlMiner
from amortized_oracle import build_global_PA_PB

M = N = 1024
K, R = 256, 64
MC = MiningConfiguration(common_dim=K, rank=R, mma_type=0,
    rows_pattern=PeriodicPattern.from_list([0, 32]),
    cols_pattern=PeriodicPattern.from_list(list(range(64))))
JOB_KEY = compute_job_key(bytes(76), MC)
TARGET = 1 << (256 - 6)
NROUNDS = 4


def build_job(jc, i):
    A, B = derive_AB_from_seed(bytes([i]) * 32, M, N, K)
    jm = JobMatrices(A=A, B=B, miner_seed=bytes([i]) * 32,
                     hash_a=merkle_root_keyed(A, JOB_KEY), hash_b=merkle_root_keyed(B, JOB_KEY),
                     job_key=JOB_KEY, m=M, n=N, k=K)
    ch = jm.commitment_hash(); _, a_noise_seed = ch
    PA, PB, e_ar_t, e_bl, e_al_full, e_br_full = build_global_PA_PB(A, B, K, R, ch)
    args = dict(A=A, B=B, e_al=e_al_full, e_br=e_br_full, e_ar_t=e_ar_t,
                e_bl=e_bl, key=a_noise_seed)
    # Reference share set (built alone into slot 0, swept synchronously).
    jc.set_job_raw(A, B, e_al_full, e_br_full, e_ar_t, e_bl, a_noise_seed, job_slot=0)
    ref = {(c.t_rows, c.t_cols, c.hash_jackpot)
           for c in jc.search_all_stream(MC, TARGET, max_return=10**9, chunk_wg=32)}
    return args, ref


def main() -> int:
    jc = JackpotCoopmat(2, 64, R, K)
    jobs = {i: build_job(jc, i) for i in range(NROUNDS + 1)}   # +1 prefetched-but-unsearched
    refs = {i: jobs[i][1] for i in jobs}
    print("refs: " + "  ".join(f"job{i}={len(refs[i])}" for i in range(NROUNDS)))

    miner = PearlMiner(
        session=types.SimpleNamespace(submit_share=lambda *a: {"result": True}),
        miner_seed=bytes(32), use_gpu_merkle=False, use_gpu_derive=False,
        use_gpu_jackpot=False, use_coopmat_jackpot=False,
        coopmat_shares_per_round=10**9, on_event=lambda kind, info: None)
    miner._jackpot_coopmat = jc

    # Oracle preflight into the requested slot; the returned state carries the
    # job's ref so the stubbed _submit_hit can verify each share in-round.
    def fake_preflight_into(work, miner_seed, job_slot):
        i = miner_seed[0]
        args = jobs[i][0]
        jc.set_job_raw(args['A'], args['B'], args['e_al'], args['e_br'],
                       args['e_ar_t'], args['e_bl'], args['key'], job_slot=job_slot)
        return types.SimpleNamespace(job_id=i, ref=refs[i])
    miner._preflight_into = fake_preflight_into

    bad = []
    got = {i: set() for i in jobs}
    lock = threading.Lock()
    def fake_submit_hit(work, state, hit, attempts, dt):
        key = (hit.t_rows, hit.t_cols, hit.hash_jackpot)
        with lock:
            if key not in state.ref:          # slot bleed / wrong job -> fail
                bad.append((state.job_id, key))
            got[state.job_id].add(key)
    miner._submit_hit = fake_submit_hit

    work = types.SimpleNamespace(mining_config=MC, target=TARGET, job_id="x")
    seed_i = [0]
    def next_seed():
        s = bytes([seed_i[0]]) * 32
        seed_i[0] += 1
        return s
    done = [0]
    def on_round():
        done[0] += 1
    def should_stop():
        return done[0] >= NROUNDS

    miner.mine_coopmat_continuous(work, should_stop=should_stop,
                                  next_seed=next_seed, on_round=on_round)

    ok_rounds = done[0] == NROUNDS
    per_round_ok = all(got[i] == refs[i] for i in range(NROUNDS))
    for i in range(NROUNDS):
        print(f"round {i}: submitted={len(got[i])} expected={len(refs[i])} "
              f"match={got[i] == refs[i]}")
    print(f"rounds={done[0]} (expect {NROUNDS})  no_slot_bleed={not bad}  "
          f"per_round_correct={per_round_ok}")
    jc.close()
    ok = ok_rounds and per_round_ok and not bad
    print("CONTINUOUS LOOP CORRECT" if ok else "CONTINUOUS LOOP **FAILED**")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
