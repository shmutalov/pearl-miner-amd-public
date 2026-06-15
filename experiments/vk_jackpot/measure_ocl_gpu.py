"""Measure the OpenCL jackpot kernel's PURE GPU time (CL profiling events), for
an apples-to-apples comparison with the Vulkan timestamp number (host.cpp). The
JackpotGpu.evaluate_batch wall-time includes per-batch buffer alloc + readback +
Python overhead; this strips all of that."""
import sys
from pathlib import Path
import numpy as np, blake3, pyopencl as cl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.pearl_amd.jackpot_gpu import JackpotGpu  # noqa: E402
from src.pearl_amd.mining_config import MiningConfiguration, PeriodicPattern, compute_job_key  # noqa
from src.pearl_amd.proof_builder import derive_AB_from_seed, merkle_root_keyed  # noqa

m, n, k, r = 131072, 131072, 4096, 128; h, w = 2, 64
mc = MiningConfiguration(common_dim=k, rank=r, mma_type=0,
    rows_pattern=PeriodicPattern.from_list([0, 32]),
    cols_pattern=PeriodicPattern.from_list(list(range(64))))
jk = compute_job_key(bytes(range(76)), mc)
A, B = derive_AB_from_seed(bytes(32), m, n, k)
ha = merkle_root_keyed(A, jk); hb = merkle_root_keyed(B, jk)
bs = blake3.blake3(jk + hb).digest(); as_ = blake3.blake3(bs + ha).digest()

dev = JackpotGpu(h=h, w=w, r=r).device
ctx = cl.Context([dev])
pq = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
gpu = JackpotGpu(h=h, w=w, r=r, context=ctx, queue=pq, variant="rdna3_wtile")
gpu.set_job(A, B, [0, 32], list(range(64)), (bs, as_), as_)

batch = 16384
t_rows = np.array([(i % 65536) * 64 % m for i in range(batch)], dtype=np.int32)
t_cols = np.array([(i % 2048) * 64 for i in range(batch)], dtype=np.int32)
mf = cl.mem_flags
trb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_rows)
tcb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_cols)
out = np.empty((batch, 8), np.uint32)
outb = cl.Buffer(ctx, mf.WRITE_ONLY, size=out.nbytes)
wg = h * w
gpu.kernel.set_args(gpu._A_buf, gpu._B_buf, gpu._eal_buf, gpu._ebr_buf, gpu._ear_buf, gpu._ebl_buf,
    trb, tcb, gpu._row_pat_buf, gpu._col_pat_buf, np.int32(k), np.int32(k), np.int32(k), gpu._key_buf, outb)
best = 1e30
for _ in range(20):
    e = cl.enqueue_nd_range_kernel(pq, gpu.kernel, (batch * wg,), (wg,))
    e.wait()
    best = min(best, (e.profile.end - e.profile.start) / 1e6)
print(f"OpenCL kernel-only GPU time: {best:.3f} ms  =>  {batch/(best/1000):,.0f} cand/s  "
      f"({gpu.kernel_path.name})")
