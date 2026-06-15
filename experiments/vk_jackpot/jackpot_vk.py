"""ctypes wrapper around libjackpot_vk (the Vulkan jackpot evaluator), exposing
a JackpotGpu-compatible surface so the search loop / miner can use the ~2.6x
faster Vulkan kernel.

set_job_raw() takes the kernel inputs directly; set_job() matches
JackpotGpu.set_job (derives noise via PearlNoiseGpu, then uploads).
"""
from __future__ import annotations
import ctypes as C
from ctypes import c_void_p, c_int, c_double, POINTER, c_int8, c_int32, c_uint32
from pathlib import Path
import numpy as np

_DIR = Path(__file__).resolve().parent
_DLL = _DIR / "jackpot_vk.dll"


def _ptr(arr, ctype):
    return arr.ctypes.data_as(POINTER(ctype))


class JackpotVk:
    def __init__(self, h: int, w: int, r: int, *, ntiles: int = 4,
                 reduce_mode: int = 0, subgroup_size: int = 0,
                 noise_context=None, noise_queue=None, noise_device=None):
        self.h, self.w, self.r = h, w, r
        spv = _DIR / f"jackpot_r{r}_n{ntiles}_red{reduce_mode}.spv"
        if not spv.exists():
            raise FileNotFoundError(f"shader {spv} not built (run build.sh)")
        self.lib = C.CDLL(str(_DLL))
        self.lib.jvk_create.restype = c_void_p
        self.lib.jvk_create.argtypes = [c_int, c_int, c_int, C.c_char_p, c_int]
        self.lib.jvk_set_job.restype = c_int
        self.lib.jvk_set_job.argtypes = [c_void_p, POINTER(c_int8), POINTER(c_int8),
            POINTER(c_int8), POINTER(c_int8), POINTER(c_uint32), POINTER(c_uint32),
            POINTER(c_int32), POINTER(c_int32), POINTER(c_uint32), c_int, c_int, c_int]
        self.lib.jvk_evaluate.restype = c_int
        self.lib.jvk_evaluate.argtypes = [c_void_p, POINTER(c_int32), POINTER(c_int32), c_int, POINTER(c_uint32)]
        self.lib.jvk_last_gpu_ms.restype = c_double
        self.lib.jvk_last_gpu_ms.argtypes = [c_void_p]
        self.lib.jvk_destroy.argtypes = [c_void_p]
        self.ctx = self.lib.jvk_create(h, w, r, str(spv).encode(), subgroup_size)
        if not self.ctx:
            raise RuntimeError("jvk_create failed")
        self._m = self._n = self._k = 0
        self._noise_ctx = (noise_context, noise_queue, noise_device)

    def set_job_raw(self, A, B, e_al, e_br_t, e_ar_t, e_bl,
                    row_pattern, col_pattern, a_noise_seed):
        A = np.ascontiguousarray(A, dtype=np.int8)
        B = np.ascontiguousarray(B, dtype=np.int8)
        m, k = A.shape; n, k2 = B.shape
        assert k == k2
        e_al = np.ascontiguousarray(e_al, dtype=np.int8).reshape(-1)
        e_br_t = np.ascontiguousarray(e_br_t, dtype=np.int8).reshape(-1)
        e_ar_t = np.ascontiguousarray(e_ar_t, dtype=np.uint32).reshape(-1)
        e_bl = np.ascontiguousarray(e_bl, dtype=np.uint32).reshape(-1)
        rp = np.ascontiguousarray(row_pattern, dtype=np.int32)
        cp = np.ascontiguousarray(col_pattern, dtype=np.int32)
        key = np.frombuffer(a_noise_seed, dtype=np.uint32).copy()
        self._hold = (A, B, e_al, e_br_t, e_ar_t, e_bl, rp, cp, key)  # keep refs alive
        rc = self.lib.jvk_set_job(self.ctx, _ptr(A, c_int8), _ptr(B, c_int8),
            _ptr(e_al, c_int8), _ptr(e_br_t, c_int8), _ptr(e_ar_t, c_uint32), _ptr(e_bl, c_uint32),
            _ptr(rp, c_int32), _ptr(cp, c_int32), _ptr(key, c_uint32), m, n, k)
        if rc != 0:
            raise RuntimeError(f"jvk_set_job failed ({rc})")
        self._m, self._n, self._k = m, n, k

    def set_job(self, A, B, row_pattern, col_pattern, commitment_hash, a_noise_seed):
        """JackpotGpu-compatible: derive noise on the GPU, then upload to Vulkan."""
        from src.pearl_amd.pearl_noise_gpu import PearlNoiseGpu
        from src.pearl_amd.jackpot import NOISE_RANGE, SEED_LABEL_A, SEED_LABEL_B
        m, k = A.shape; n, _ = B.shape
        b_noise_seed, _ = commitment_hash
        ng = PearlNoiseGpu(*self._noise_ctx) if any(self._noise_ctx) else PearlNoiseGpu()
        e_al, _ = ng.uniform(SEED_LABEL_A, a_noise_seed, m, self.r, read_back=True)
        e_ar_t, _ = ng.perm(SEED_LABEL_A, a_noise_seed, k, NOISE_RANGE // 2, read_back=True)
        e_bl, _ = ng.perm(SEED_LABEL_B, b_noise_seed, k, NOISE_RANGE // 2, read_back=True)
        e_br_t, _ = ng.uniform(SEED_LABEL_B, b_noise_seed, n, self.r, read_back=True)
        self.set_job_raw(A, B, e_al, e_br_t, e_ar_t, e_bl, row_pattern, col_pattern, a_noise_seed)

    def evaluate_batch(self, t_rows, t_cols) -> np.ndarray:
        t_rows = np.ascontiguousarray(t_rows, dtype=np.int32)
        t_cols = np.ascontiguousarray(t_cols, dtype=np.int32)
        batch = int(t_rows.shape[0])
        out = np.empty((batch, 8), dtype=np.uint32)
        rc = self.lib.jvk_evaluate(self.ctx, _ptr(t_rows, c_int32), _ptr(t_cols, c_int32),
                                   batch, _ptr(out, c_uint32))
        if rc != 0:
            raise RuntimeError(f"jvk_evaluate failed ({rc})")
        return out.view(np.uint8).reshape(batch, 32)

    def last_gpu_ms(self) -> float:
        return float(self.lib.jvk_last_gpu_ms(self.ctx))

    def close(self):
        if getattr(self, "ctx", None):
            self.lib.jvk_destroy(self.ctx); self.ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
