"""Validate JackpotVk (Vulkan, via DLL) bit-identical to JackpotGpu (OpenCL) on
identical noise, and benchmark both. Usage: validate_vk.py [small|pool] [batch]"""
import sys, time
from pathlib import Path
import numpy as np, blake3, pyopencl as cl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.pearl_amd.jackpot_vk import JackpotVk  # noqa: E402
from src.pearl_amd.jackpot_gpu import JackpotGpu  # noqa: E402
from src.pearl_amd.mining_config import MiningConfiguration, PeriodicPattern, compute_job_key  # noqa
from src.pearl_amd.proof_builder import derive_AB_from_seed, merkle_root_keyed  # noqa

shape = sys.argv[1] if len(sys.argv) > 1 else "small"
batch = int(sys.argv[2]) if len(sys.argv) > 2 else (256 if shape == "small" else 16384)
if shape == "small":
    m = n = k = 256; r = 64
else:
    m = n = 131072; k = 4096; r = 128
h, w = 2, 64

mc = MiningConfiguration(common_dim=k, rank=r, mma_type=0,
    rows_pattern=PeriodicPattern.from_list([0, 32]),
    cols_pattern=PeriodicPattern.from_list(list(range(64))))
jk = compute_job_key(bytes(range(76)), mc)
A, B = derive_AB_from_seed(bytes(32), m, n, k)
ha = merkle_root_keyed(A, jk); hb = merkle_root_keyed(B, jk)
bs = blake3.blake3(jk + hb).digest(); as_ = blake3.blake3(bs + ha).digest()

# Reference (OpenCL) + read back its GPU noise so Vulkan consumes identical bytes.
ocl = JackpotGpu(h=h, w=w, r=r, variant="rdna3_wtile", use_gpu_noise=True)
ocl.set_job(A, B, [0, 32], list(range(64)), (bs, as_), as_)
e_al = np.empty(m * r, np.int8);  cl.enqueue_copy(ocl.queue, e_al,  ocl._eal_buf)
e_br_t = np.empty(n * r, np.int8); cl.enqueue_copy(ocl.queue, e_br_t, ocl._ebr_buf)
e_ar_t = np.empty(k * 2, np.uint32); cl.enqueue_copy(ocl.queue, e_ar_t, ocl._ear_buf)
e_bl = np.empty(k * 2, np.uint32); cl.enqueue_copy(ocl.queue, e_bl,   ocl._ebl_buf)
ocl.queue.finish()

vk = JackpotVk(h=h, w=w, r=r)  # packed, NTILES=4, V0
vk.set_job_raw(A, B, e_al.reshape(m, r), e_br_t.reshape(n, r),
               e_ar_t.reshape(k, 2), e_bl.reshape(k, 2), [0, 32], list(range(64)), as_)

# Valid in-range offsets for both shapes (b_col = t_c + col_pattern[<64] < n).
t_rows = ((np.arange(batch) % ((m - 32) // 64)) * 64).astype(np.int32)
t_cols = ((np.arange(batch) % ((n - 63) // 64)) * 64).astype(np.int32)

ref = ocl.evaluate_batch(t_rows, t_cols)
got = vk.evaluate_batch(t_rows, t_cols)
ident = np.array_equal(ref, got)
print(f"shape={shape} batch={batch}  bit-identical(VK==OpenCL): {ident}")
if not ident:
    i = int(np.argmax(np.any(ref != got, axis=1)))
    print(f"  first mismatch cand {i}:\n   ocl {ref[i].tobytes().hex()}\n   vk  {got[i].tobytes().hex()}")

# Throughput (pure GPU time both sides)
vk.evaluate_batch(t_rows[:8], t_cols[:8])
best_vk = 1e30
for _ in range(15):
    vk.evaluate_batch(t_rows, t_cols); best_vk = min(best_vk, vk.last_gpu_ms())
print(f"  Vulkan : {best_vk:7.2f} ms  =>  {batch/(best_vk/1000):,.0f} cand/s")
