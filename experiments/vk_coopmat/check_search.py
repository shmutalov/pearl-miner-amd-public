"""Validate search_host hits against numpy: recompute every small-shape
candidate's hash, determine the true hit set for the given target_lz, and check
the GPU recorded exactly that set (each record's hash correct + below target +
no misses)."""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from pearl_amd.mining_config import MiningConfiguration, PeriodicPattern, compute_job_key
from pearl_amd.proof_builder import derive_AB_from_seed, merkle_root_keyed, JobMatrices
from amortized_oracle import build_global_PA_PB, amortized_hash


def main() -> int:
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "job"
    target_lz = int(sys.argv[sys.argv.index("--lz") + 1])
    meta = json.load(open(os.path.join(out, "meta.json")))
    m, n, k, r, nblocks, nbands = (meta[x] for x in ("m", "n", "k", "r", "nblocks", "nbands"))
    target = 1 << (256 - target_lz)

    mc = MiningConfiguration(common_dim=k, rank=r, mma_type=0,
        rows_pattern=PeriodicPattern.from_list([0, 32]),
        cols_pattern=PeriodicPattern.from_list(list(range(64))))
    job_key = compute_job_key(bytes(76), mc)
    A, B = derive_AB_from_seed(bytes(32), m, n, k)
    jm = JobMatrices(A=A, B=B, miner_seed=bytes(32),
                     hash_a=merkle_root_keyed(A, job_key), hash_b=merkle_root_keyed(B, job_key),
                     job_key=job_key, m=m, n=n, k=k)
    commitment_hash = jm.commitment_hash(); _, a_noise_seed = commitment_hash
    PA, PB, *_ = build_global_PA_PB(A, B, k, r, commitment_hash)

    # True hit set over all candidates.
    true_hits = {}
    for g in range(nbands):
        for b in range(nblocks):
            for c in range(32):
                t_r, t_c = g * 64 + c, b * 64
                h = amortized_hash(PA, PB, [t_r, t_r + 32], list(range(t_c, t_c + 64)),
                                   k, r, a_noise_seed)
                if int.from_bytes(h, "little") < target:
                    true_hits[(t_r, t_c)] = h
    print(f"numpy true hits below target_lz={target_lz}: {len(true_hits)}")

    raw = np.fromfile(os.path.join(out, "hits_out.bin"), dtype="<u4")
    nhit = int(raw[0]); recs = raw[1:].reshape(-1, 10)
    print(f"GPU reported {nhit} hits, saved {recs.shape[0]} records")

    gpu_hits = {}
    bad = 0
    for rec in recs:
        t_r, t_c = int(rec[0]), int(rec[1])
        h = rec[2:10].tobytes()
        gpu_hits[(t_r, t_c)] = h
        ref = true_hits.get((t_r, t_c))
        if ref is None:
            print(f"  FALSE HIT (t_r={t_r},t_c={t_c}) not below target"); bad += 1
        elif ref != h:
            print(f"  WRONG HASH at (t_r={t_r},t_c={t_c})"); bad += 1
    missed = set(true_hits) - set(gpu_hits)
    if nhit == recs.shape[0] and missed:
        print(f"  MISSED {len(missed)} true hits, e.g. {list(missed)[:3]}"); bad += len(missed)

    if bad or nhit != len(true_hits):
        print(f"SEARCH **WRONG** (bad={bad}, gpu_count={nhit} vs true={len(true_hits)})")
        return 1
    print("SEARCH CORRECT (hit set + hashes match numpy exactly)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
