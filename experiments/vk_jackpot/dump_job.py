"""Dump jackpot-kernel inputs + reference hashes to a directory of binary files
that the Vulkan host (host.cpp) consumes. Reference hashes come from the
validated OpenCL kernel fed the identical CPU-generated noise.

Usage:
  python dump_job.py <out_dir> [--shape small|pool] [--batch N] [--seed S]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import blake3

# Allow running from the repo root or the experiment dir.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.pearl_amd.jackpot import (  # noqa: E402
    NOISE_RANGE, SEED_LABEL_A, SEED_LABEL_B,
    generate_uniform_random_matrix, generate_permutation_matrix,
)
from src.pearl_amd.jackpot_gpu import JackpotGpu  # noqa: E402
from src.pearl_amd.mining_config import (  # noqa: E402
    MiningConfiguration, PeriodicPattern, compute_job_key,
)
from src.pearl_amd.proof_builder import derive_AB_from_seed, merkle_root_keyed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--shape", choices=["small", "pool"], default="small")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.shape == "small":
        m = n = k = 256; r = 64
    else:
        m = n = 131072; k = 4096; r = 128
    h, w = 2, 64

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    mc = MiningConfiguration(
        common_dim=k, rank=r, mma_type=0,
        rows_pattern=PeriodicPattern.from_list([0, 32]),
        cols_pattern=PeriodicPattern.from_list(list(range(64))),
    )
    job_key = compute_job_key(bytes(range(76)), mc)
    A, B = derive_AB_from_seed(bytes(32), m, n, k)
    hash_a = merkle_root_keyed(A, job_key); hash_b = merkle_root_keyed(B, job_key)
    b_noise_seed = blake3.blake3(job_key + hash_b).digest()
    a_noise_seed = blake3.blake3(b_noise_seed + hash_a).digest()

    row_pattern = np.array([0, 32], dtype=np.int32)
    col_pattern = np.arange(64, dtype=np.int32)

    # Valid candidate offsets (in-range; both kernels evaluate them identically).
    t_rows = ((rng.integers(0, (m - 32) // 64, size=args.batch)) * 64).astype(np.int32)
    t_cols = ((rng.integers(0, (n - 63) // 64, size=args.batch)) * 64).astype(np.int32)

    # Generate noise on the GPU (fast even at pool shape) and read it back so the
    # dumped noise is byte-identical to what the reference kernel consumes.
    import pyopencl as cl
    gpu = JackpotGpu(h=h, w=w, r=r, variant="rdna3_wtile", use_gpu_noise=True)
    gpu.set_job(A, B, [0, 32], list(range(64)), (b_noise_seed, a_noise_seed), a_noise_seed)
    e_al  = np.empty(m * r, np.int8);  cl.enqueue_copy(gpu.queue, e_al,  gpu._eal_buf)
    e_br_t = np.empty(n * r, np.int8); cl.enqueue_copy(gpu.queue, e_br_t, gpu._ebr_buf)
    e_ar_t = np.empty(k * 2, np.uint32); cl.enqueue_copy(gpu.queue, e_ar_t, gpu._ear_buf)
    e_bl   = np.empty(k * 2, np.uint32); cl.enqueue_copy(gpu.queue, e_bl,   gpu._ebl_buf)
    gpu.queue.finish()
    ref = gpu.evaluate_batch(t_rows, t_cols)  # (batch,32) uint8

    def dump(name, arr): (out / name).write_bytes(np.ascontiguousarray(arr).tobytes())
    dump("A.bin", A.astype(np.int8))
    dump("B.bin", B.astype(np.int8))
    dump("e_al.bin", e_al.astype(np.int8))
    dump("e_br_t.bin", e_br_t.astype(np.int8))
    dump("e_ar_t.bin", e_ar_t.astype(np.uint32))
    dump("e_bl.bin", e_bl.astype(np.uint32))
    dump("t_rows.bin", t_rows); dump("t_cols.bin", t_cols)
    dump("row_pattern.bin", row_pattern); dump("col_pattern.bin", col_pattern)
    dump("key.bin", np.frombuffer(a_noise_seed, dtype=np.uint32))
    dump("ref.bin", ref)

    meta = dict(m=m, n=n, k=k, r=r, h=h, w=w, batch=args.batch,
                n_cols_A=k, n_cols_B=k)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    # Space-separated for the C++ host (avoids a JSON dep).
    (out / "meta.txt").write_text(
        f"{m} {n} {k} {r} {h} {w} {args.batch} {k} {k}\n")
    print(f"dumped {args.shape} shape, batch={args.batch} to {out}")
    print(f"  meta: {meta}")
    print(f"  ref[0]={ref[0].tobytes().hex()}")


if __name__ == "__main__":
    main()
