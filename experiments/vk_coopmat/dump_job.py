"""Build a (small or pool) Pearl job, write the global PA/PB/key + meta to
disk for the C++ coopmat host (pearl_host.exe) to consume, and write the
expected per-candidate hashes for the checker.

Outputs (in --out dir, default ./job/):
  PA.bin   int8  m*k
  PB.bin   int8  n*k
  key.bin  8 u32 (a_noise_seed)
  meta.json  {m,n,k,r,nblocks,nbands}
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from pearl_amd.mining_config import MiningConfiguration, PeriodicPattern, compute_job_key
from pearl_amd.proof_builder import derive_AB_from_seed, merkle_root_keyed, JobMatrices

# Reuse the proven decomposition.
sys.path.insert(0, os.path.dirname(__file__))
from amortized_oracle import build_global_PA_PB


def main() -> int:
    pool = "--pool" in sys.argv
    out = "job"
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    os.makedirs(out, exist_ok=True)

    m, n, k, r = (131072, 131072, 4096, 128) if pool else (1024, 1024, 256, 64)
    mc = MiningConfiguration(
        common_dim=k, rank=r, mma_type=0,
        rows_pattern=PeriodicPattern.from_list([0, 32]),
        cols_pattern=PeriodicPattern.from_list(list(range(64))),
    )
    header = bytes(76)
    job_key = compute_job_key(header, mc)
    A, B = derive_AB_from_seed(bytes(32), m, n, k)
    hash_a = merkle_root_keyed(A, job_key)
    hash_b = merkle_root_keyed(B, job_key)
    jm = JobMatrices(A=A, B=B, miner_seed=bytes(32), hash_a=hash_a, hash_b=hash_b,
                     job_key=job_key, m=m, n=n, k=k)
    commitment_hash = jm.commitment_hash()
    _, a_noise_seed = commitment_hash

    PA_i8, PB_i8, e_ar_t, e_bl, e_al_full, e_br_full = build_global_PA_PB(
        A, B, k, r, commitment_hash)

    # Row-major C-contiguous int8.
    np.ascontiguousarray(PA_i8).tofile(os.path.join(out, "PA.bin"))
    np.ascontiguousarray(PB_i8).tofile(os.path.join(out, "PB.bin"))
    # BLAKE3 keyed-hash key = a_noise_seed as 8 LE u32.
    np.frombuffer(a_noise_seed, dtype="<u4").tofile(os.path.join(out, "key.bin"))

    if "--raw" in sys.argv:
        # Raw inputs for the GPU PA/PB-build kernel (Phase A) validation.
        np.ascontiguousarray(A).tofile(os.path.join(out, "A.bin"))                 # m x k i8
        np.ascontiguousarray(B).tofile(os.path.join(out, "B.bin"))                 # n x k i8
        np.ascontiguousarray(e_al_full).tofile(os.path.join(out, "EAL.bin"))       # m x r i8
        np.ascontiguousarray(e_br_full).tofile(os.path.join(out, "EBR.bin"))       # n x r i8
        np.ascontiguousarray(e_ar_t.astype("<u4")).tofile(os.path.join(out, "EAR.bin"))  # k x 2 u32
        np.ascontiguousarray(e_bl.astype("<u4")).tofile(os.path.join(out, "EBL.bin"))    # k x 2 u32
        print("  + raw inputs A,B,EAL,EBR,EAR,EBL")

    meta = dict(m=m, n=n, k=k, r=r, nbands=m // 64, nblocks=n // 64)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"wrote {out}/ PA{PA_i8.shape} PB{PB_i8.shape} key(8u32) meta={meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
