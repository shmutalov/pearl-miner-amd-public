"""Exercise PearlMiner._search_coopmat_pipelined (producer/consumer threads)
against the real GPU stream, with a stub session and a counting _submit_hit so
no merkle/proof/network stack is needed. Checks:

  1. happy path  : every share the GPU stream finds reaches the consumer exactly
                   once (set matches search_all), with simulated submit latency.
  2. error path  : a consumer exception is re-raised to the caller, no hang.
  3. exhausted   : zero hits emits 'search_exhausted'.
"""
from __future__ import annotations
import os, sys, time, types, threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from pearl_amd.mining_config import MiningConfiguration, PeriodicPattern, compute_job_key
from pearl_amd.proof_builder import derive_AB_from_seed, merkle_root_keyed, JobMatrices
from pearl_amd.jackpot_coopmat import JackpotCoopmat
from pearl_amd.miner import PearlMiner
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

    target = 1 << (256 - 6)
    jc = JackpotCoopmat(2, 64, r, k)
    jc.set_job_raw(A, B, e_al_full, e_br_full, e_ar_t, e_bl, a_noise_seed)

    # Minimal miner: every GPU helper off (no OpenCL needed), coopmat attached by hand.
    miner = PearlMiner(
        session=types.SimpleNamespace(submit_share=lambda *a: {"result": True}),
        miner_seed=bytes(32), use_gpu_merkle=False, use_gpu_derive=False,
        use_gpu_jackpot=False, use_coopmat_jackpot=False,
        coopmat_shares_per_round=1_000_000, on_event=lambda kind, info: None)
    miner._jackpot_coopmat = jc
    work = types.SimpleNamespace(mining_config=mc, target=target, job_id="test")

    # ---- ground truth ----
    base, _, _ = jc.search_all(mc, target, max_return=1_000_000, chunk_wg=32)
    base_set = {(c.t_rows, c.t_cols, c.hash_jackpot) for c in base}

    # ---- 1: happy path ----
    submitted, lock = [], threading.Lock()
    def fake_submit_hit(w, st, hit, attempts, dt):
        time.sleep(0.0005)                      # simulate ~0.5ms network per share
        with lock:
            submitted.append((hit.t_rows, hit.t_cols, hit.hash_jackpot))
    miner._submit_hit = fake_submit_hit
    t0 = time.time(); miner._search_coopmat_pipelined(work, None); dt = time.time() - t0
    got_set = set(submitted)
    ok_happy = got_set == base_set and len(submitted) == len(base_set)
    print(f"1 happy   : submitted={len(submitted)} expected={len(base_set)} "
          f"set_equal={got_set == base_set} no_dupes={len(submitted) == len(got_set)} "
          f"({dt*1000:.0f}ms)")

    # ---- 2: error path (consumer raises) ----
    calls = [0]
    def boom(w, st, hit, attempts, dt):
        calls[0] += 1
        if calls[0] >= 3:               # >= so parallel workers can't skip the trigger
            raise RuntimeError("pool rejected")
    miner._submit_hit = boom
    raised = False
    try:
        miner._search_coopmat_pipelined(work, None)
    except RuntimeError as e:
        raised = str(e) == "pool rejected"
    print(f"2 error   : re-raised={raised} (consumer calls={calls[0]})")

    # ---- 3: exhausted (impossibly tight target -> no hits) ----
    events = []
    miner._on_event = lambda kind, info: events.append(kind)
    miner._submit_hit = fake_submit_hit
    miner._search_coopmat_pipelined(
        types.SimpleNamespace(mining_config=mc, target=1, job_id="t"), None)
    ok_exh = "search_exhausted" in events
    print(f"3 exhaust : search_exhausted emitted={ok_exh}")

    # ---- 4: interruptible (should_continue=False) -> prompt return, 0 submits, no hang ----
    stopped = []
    miner._submit_hit = lambda w, st, h, a, d: stopped.append(1)
    t0 = time.time()
    miner._search_coopmat_pipelined(work, None, should_continue=lambda: False)
    dt_stop = time.time() - t0
    ok_stop = len(stopped) == 0
    print(f"4 interrupt: submitted={len(stopped)} (expect 0), returned in {dt_stop*1000:.0f}ms (no hang)")

    # ---- 5: submit pacing -> spaced starts (no burst), not faster than the gate ----
    miner._coopmat_submit_stagger = 0.02       # 20 ms min gap between submit starts
    miner._coopmat_shares_per_round = 15       # keep the test quick
    stamps, slock = [], threading.Lock()
    def rec_submit(w, st, hit, attempts, dt):
        now = time.monotonic()
        with slock:
            stamps.append(now)
    miner._submit_hit = rec_submit
    miner._search_coopmat_pipelined(work, None)
    s = sorted(stamps)
    gaps = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    min_gap = min(gaps) if gaps else 0.0
    ok_pace = len(s) >= 5 and min_gap >= 0.02 * 0.85   # >= ~17 ms, allowing slop
    print(f"5 pacing  : {len(s)} submits, min start-gap {min_gap*1e3:.1f}ms (>= ~17ms): {ok_pace}")

    jc.close()
    ok = ok_happy and raised and ok_exh and ok_stop and ok_pace
    print("PIPELINE ORCHESTRATION OK" if ok else "PIPELINE **FAILED**")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
