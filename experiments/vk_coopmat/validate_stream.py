"""Validate the async ping-pong path (jcm_search_submit/collect +
JackpotCoopmat.search_all_stream) against the synchronous search_all.

A small chunk_wg forces the (band x block) grid to split into many tiles so
both ping-pong slots are exercised. The streamed share set must be byte-for-byte
identical to the synchronous sweep (same (t_r, t_c, hash), same attempt count),
and the max_return bound must be honored.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from pearl_amd.mining_config import MiningConfiguration, PeriodicPattern, compute_job_key
from pearl_amd.proof_builder import derive_AB_from_seed, merkle_root_keyed, JobMatrices
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

    target_lz = 6                       # ~1/64 pass -> ~128 hits across the grid
    target = 1 << (256 - target_lz)

    jc = JackpotCoopmat(2, 64, r, k)
    jc.set_job_raw(A, B, e_al_full, e_br_full, e_ar_t, e_bl, a_noise_seed)

    CHUNK = 32                          # 16x16=256 WG grid -> 8 tiles, both slots
    BIG = 1_000_000                     # effectively unbounded for this grid

    def key(c):
        return (c.t_rows, c.t_cols, c.hash_jackpot)

    t0 = time.time()
    base, attempts_b, _ = jc.search_all(mc, target, max_return=BIG, chunk_wg=CHUNK)
    dt_list = time.time() - t0

    t0 = time.time()
    stream = list(jc.search_all_stream(mc, target, max_return=BIG, chunk_wg=CHUNK))
    dt_stream = time.time() - t0

    base_keys = [key(c) for c in base]
    stream_keys = [key(c) for c in stream]
    ok_count = len(base) == len(stream)
    ok_set = set(base_keys) == set(stream_keys)
    ok_order = base_keys == stream_keys     # stronger: same tile/record order
    ok_attempts = attempts_b == jc.last_attempts

    print(f"search_all       : {len(base):4d} shares, attempts={attempts_b}, {dt_list*1000:6.0f}ms")
    print(f"search_all_stream: {len(stream):4d} shares, attempts={jc.last_attempts}, {dt_stream*1000:6.0f}ms")
    print(f"same count={ok_count}  same set={ok_set}  same order={ok_order}  same attempts={ok_attempts}")
    if not ok_set:
        only_b = set(base_keys) - set(stream_keys)
        only_s = set(stream_keys) - set(base_keys)
        print(f"  only in search_all={len(only_b)}  only in stream={len(only_s)}")

    capped = list(jc.search_all_stream(mc, target, max_return=7, chunk_wg=CHUNK))
    ok_cap = len(capped) == 7
    print(f"max_return=7 honored: {ok_cap} (got {len(capped)})")

    # Interruptible: stop from the 3rd tile-check onward -> fewer shares, no hang,
    # and the slots stay reusable (a normal stream afterward still returns the
    # full set, proving in-flight tiles were drained and fences left signaled).
    calls = [0]
    def cont():
        calls[0] += 1
        return calls[0] <= 2
    part = list(jc.search_all_stream(mc, target, max_return=BIG, chunk_wg=CHUNK,
                                     should_continue=cont))
    after = list(jc.search_all_stream(mc, target, max_return=BIG, chunk_wg=CHUNK))
    ok_interrupt = len(part) < len(stream) and len(after) == len(stream)
    print(f"interrupt: stopped early ({len(part)} < {len(stream)}) and slots reusable "
          f"(after={len(after)}): {ok_interrupt}")

    jc.close()
    ok = ok_count and ok_set and ok_attempts and ok_cap and ok_interrupt
    print("STREAM EQUIVALENT TO search_all" if ok else "STREAM **MISMATCH**")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
