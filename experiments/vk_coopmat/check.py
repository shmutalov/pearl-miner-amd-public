"""Compare pearl_host.exe's hashes.bin against the proven numpy reference.

Candidate index i (within the dumped range) -> w = wg_off + i//32, c = i%32;
band_g = w//nblocks, block_b = w%nblocks; t_r = band_g*64 + c, t_c = block_b*64.
rows = [t_r, t_r+32], cols = [t_c .. t_c+63].
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from pearl_amd.mining_config import MiningConfiguration, PeriodicPattern, compute_job_key
from pearl_amd.proof_builder import derive_AB_from_seed, merkle_root_keyed, JobMatrices
from pearl_amd.jackpot import evaluate_candidate
from amortized_oracle import build_global_PA_PB, amortized_hash


def main() -> int:
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "job"
    wg_off = int(sys.argv[sys.argv.index("--wg-off") + 1]) if "--wg-off" in sys.argv else 0
    sample = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else 256

    meta = json.load(open(os.path.join(out, "meta.json")))
    m, n, k, r = meta["m"], meta["n"], meta["k"], meta["r"]
    nblocks = meta["nblocks"]

    # Rebuild the same job (deterministic from seed) and PA/PB.
    mc = MiningConfiguration(common_dim=k, rank=r, mma_type=0,
        rows_pattern=PeriodicPattern.from_list([0, 32]),
        cols_pattern=PeriodicPattern.from_list(list(range(64))))
    job_key = compute_job_key(bytes(76), mc)
    A, B = derive_AB_from_seed(bytes(32), m, n, k)
    jm = JobMatrices(A=A, B=B, miner_seed=bytes(32),
                     hash_a=merkle_root_keyed(A, job_key), hash_b=merkle_root_keyed(B, job_key),
                     job_key=job_key, m=m, n=n, k=k)
    commitment_hash = jm.commitment_hash(); _, a_noise_seed = commitment_hash
    PA_i8, PB_i8, *_ = build_global_PA_PB(A, B, k, r, commitment_hash)

    hashes = np.fromfile(os.path.join(out, "hashes.bin"), dtype="<u4")
    n_cand = hashes.size // 8
    hashes = hashes.reshape(n_cand, 8)
    print(f"loaded {n_cand} GPU hashes; checking {min(sample, n_cand)} (+corners)")

    import random
    random.seed(7)
    idxs = set([0, n_cand - 1])
    idxs |= set(random.sample(range(n_cand), min(sample, n_cand)))

    bad = 0
    checked_eval = 0
    for i in sorted(idxs):
        w = wg_off + i // 32
        c = i % 32
        band_g, block_b = w // nblocks, w % nblocks
        t_r, t_c = band_g * 64 + c, block_b * 64
        rows = [t_r, t_r + 32]
        cols = list(range(t_c, t_c + 64))
        ref = amortized_hash(PA_i8, PB_i8, rows, cols, k, r, a_noise_seed)
        got = hashes[i].tobytes()
        if got != ref:
            if bad < 5:
                print(f"  MISMATCH cand i={i} (t_r={t_r},t_c={t_c}):")
                print(f"    ref={ref.hex()}")
                print(f"    got={got.hex()}")
            bad += 1
            continue
        # extra: confirm a few directly against evaluate_candidate
        if checked_eval < 8:
            ev_val, ev_hash = evaluate_candidate(A, B, rows, cols, commitment_hash, a_noise_seed, k, r)
            assert ev_hash == ref, "oracle drift!"
            checked_eval += 1

    total = len(idxs)
    print(f"  {total - bad}/{total} GPU hashes match numpy reference "
          f"({checked_eval} also confirmed vs evaluate_candidate)")
    if bad:
        print("FUSED COOPMAT KERNEL **WRONG**")
        return 1
    print("FUSED COOPMAT KERNEL CORRECT (bit-identical)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
