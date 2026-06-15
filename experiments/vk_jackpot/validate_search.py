"""Validate native jvk_search vs JackpotGpu.search (same first hit) and measure
end-to-end search throughput (native C++ loop vs the Python loop).
Usage: validate_search.py [small|pool]"""
import sys, time
from pathlib import Path
import numpy as np, blake3, pyopencl as cl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from jackpot_vk import JackpotVk
from src.pearl_amd.jackpot_gpu import JackpotGpu
from src.pearl_amd.mining_config import MiningConfiguration, PeriodicPattern, compute_job_key
from src.pearl_amd.proof_builder import derive_AB_from_seed, merkle_root_keyed

shape = sys.argv[1] if len(sys.argv) > 1 else "pool"
if shape == "small": m = n = k = 1024; r = 64
else: m = n = 131072; k = 4096; r = 128
h, w = 2, 64
mc = MiningConfiguration(common_dim=k, rank=r, mma_type=0,
    rows_pattern=PeriodicPattern.from_list([0, 32]), cols_pattern=PeriodicPattern.from_list(list(range(64))))
jk = compute_job_key(bytes(range(76)), mc)
A, B = derive_AB_from_seed(bytes(32), m, n, k)
ha = merkle_root_keyed(A, jk); hb = merkle_root_keyed(B, jk)
bs = blake3.blake3(jk + hb).digest(); as_ = blake3.blake3(bs + ha).digest()

ocl = JackpotGpu(h=h, w=w, r=r, variant="rdna3_wtile", use_gpu_noise=True)
ocl.set_job(A, B, [0, 32], list(range(64)), (bs, as_), as_)
e_al = np.empty(m * r, np.int8);  cl.enqueue_copy(ocl.queue, e_al,  ocl._eal_buf)
e_br_t = np.empty(n * r, np.int8); cl.enqueue_copy(ocl.queue, e_br_t, ocl._ebr_buf)
e_ar_t = np.empty(k * 2, np.uint32); cl.enqueue_copy(ocl.queue, e_ar_t, ocl._ear_buf)
e_bl = np.empty(k * 2, np.uint32); cl.enqueue_copy(ocl.queue, e_bl, ocl._ebl_buf); ocl.queue.finish()
vk = JackpotVk(h=h, w=w, r=r)
vk.set_job_raw(A, B, e_al.reshape(m, r), e_br_t.reshape(n, r), e_ar_t.reshape(k, 2), e_bl.reshape(k, 2),
               [0, 32], list(range(64)), as_)

print("== correctness: same first hit (compare candidate identity) ==")
# Use batch_size=1 for the OpenCL reference so its whole-batch attempts counter
# matches jvk_search's count-to-the-hit convention exactly.
for bits in (255, 252, 250, 247):
    target = 1 << bits
    co, ao, _ = ocl.search(mc, target, batch_size=1, max_attempts=300000)
    cv, av, _ = vk.search(mc, target, batch_size=16384, max_attempts=300000)
    if co is None and cv is None:
        print(f"  target 2^{bits}: both no-hit  ({ao} vs {av} attempts)")
    elif co and cv:
        same = (co.t_rows == cv.t_rows and co.t_cols == cv.t_cols
                and co.hash_jackpot == cv.hash_jackpot)
        print(f"  target 2^{bits}: hit tr={cv.t_rows} tc={cv.t_cols} attempts={av} "
              f"(ocl@{ao})  candidate {'OK' if same else 'MISMATCH'}")
    else:
        print(f"  target 2^{bits}: MISMATCH (ocl found={co is not None} vk found={cv is not None})")

print("\n== throughput: scan N candidates (impossible target) ==")
N = 200000 if shape == "pool" else 50000
co, ao, dto = ocl.search(mc, 0, batch_size=16384, max_attempts=N)
cv, av, dtv = vk.search(mc, 0, batch_size=16384, max_attempts=N)
print(f"  JackpotGpu.search (Python loop) : {ao/dto:>10,.0f} cand/s ({dto*1000:.0f} ms / {ao})")
print(f"  JackpotVk.search  (native loop) : {av/dtv:>10,.0f} cand/s ({dtv*1000:.0f} ms / {av})")
print(f"  speedup: {(av/dtv)/(ao/dto):.2f}x")
