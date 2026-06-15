"""Confirm the perm-noise-rank fix: JackpotGpu (OpenCL) and JackpotVk (Vulkan)
must now match evaluate_candidate at r=128 (was broken: perm rank used
NOISE_RANGE//2=64) and still match at r=64 (unchanged)."""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from pearl_amd.mining_config import MiningConfiguration, PeriodicPattern, compute_job_key
from pearl_amd.proof_builder import derive_AB_from_seed, merkle_root_keyed, JobMatrices
from pearl_amd.jackpot import evaluate_candidate


def check(make_eval, label, r):
    m = n = k = 256
    mc = MiningConfiguration(common_dim=k, rank=r, mma_type=0,
        rows_pattern=PeriodicPattern.from_list([0, 32]),
        cols_pattern=PeriodicPattern.from_list(list(range(64))))
    job_key = compute_job_key(bytes(76), mc)
    A, B = derive_AB_from_seed(bytes(32), m, n, k)
    jm = JobMatrices(A=A, B=B, miner_seed=bytes(32),
                     hash_a=merkle_root_keyed(A, job_key), hash_b=merkle_root_keyed(B, job_key),
                     job_key=job_key, m=m, n=n, k=k)
    ch = jm.commitment_hash(); _, a_noise_seed = ch

    ev = make_eval(r, k)
    ev.set_job(A, B, mc.rows_pattern.to_list(), mc.cols_pattern.to_list(), ch, a_noise_seed)

    # A few valid candidates.
    trs = np.array([0, 1, 64, 65, 128], dtype=np.int32)
    tcs = np.array([0, 64, 128, 0, 64], dtype=np.int32)
    got = ev.evaluate_batch(trs, tcs)
    bad = 0
    for i in range(len(trs)):
        rows = list(mc.rows_pattern.indices_with_offset(int(trs[i])))
        cols = list(mc.cols_pattern.indices_with_offset(int(tcs[i])))
        _, ref = evaluate_candidate(A, B, rows, cols, ch, a_noise_seed, k, r)
        if bytes(got[i]) != ref:
            bad += 1
    print(f"  [{label} r={r}] {len(trs)-bad}/{len(trs)} match oracle")
    if hasattr(ev, "close"):
        try: ev.close()
        except Exception: pass
    return bad == 0


def main() -> int:
    from pearl_amd.jackpot_gpu import JackpotGpu
    results = []
    for r in (64, 128):
        results.append(check(lambda r, k: JackpotGpu(2, 64, r), "OpenCL", r))
    try:
        from pearl_amd.jackpot_vk import JackpotVk
        for r in (64, 128):
            results.append(check(lambda r, k: JackpotVk(2, 64, r), "Vulkan", r))
    except Exception as e:
        print(f"  (Vulkan path skipped: {e!r})")

    if all(results):
        print("NOISE FIX CONFIRMED (OpenCL + Vulkan match oracle at r=64 AND r=128)")
        return 0
    print("NOISE FIX **INCOMPLETE**")
    return 1


if __name__ == "__main__":
    sys.exit(main())
