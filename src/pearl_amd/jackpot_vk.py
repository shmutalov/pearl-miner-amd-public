"""ctypes wrapper around libjackpot_vk (the Vulkan jackpot evaluator), exposing
a JackpotGpu-compatible surface so the search loop / miner can use the ~2.6x
faster Vulkan kernel.

set_job_raw() takes the kernel inputs directly; set_job() matches
JackpotGpu.set_job (derives noise via PearlNoiseGpu, then uploads).
"""
from __future__ import annotations
import ctypes as C
from ctypes import (c_void_p, c_int, c_double, POINTER, byref,
                    c_int8, c_int32, c_uint32, c_int64, c_uint8)
from pathlib import Path
import numpy as np

# Native artifacts (built by src/pearl_amd/vk/build.sh) live in the vk/ subdir.
_VK = Path(__file__).resolve().parent / "vk"
_DLL = _VK / "jackpot_vk.dll"


class JvkHit(C.Structure):
    _fields_ = [("found", c_int32), ("t_rows", c_int32), ("t_cols", c_int32),
                ("hash", c_uint8 * 32), ("attempts", c_int64)]


def _ptr(arr, ctype):
    return arr.ctypes.data_as(POINTER(ctype))


class JackpotVk:
    def __init__(self, h: int, w: int, r: int, *, ntiles: int = 4,
                 reduce_mode: int = 0, subgroup_size: int = 0,
                 noise_context=None, noise_queue=None, noise_device=None):
        self.h, self.w, self.r = h, w, r
        spv = _VK / f"jackpot_r{r}_n{ntiles}_red{reduce_mode}.spv"
        if not spv.exists():
            raise FileNotFoundError(f"shader {spv} not built (run src/pearl_amd/vk/build.sh)")
        self.lib = C.CDLL(str(_DLL))
        self.lib.jvk_create.restype = c_void_p
        self.lib.jvk_create.argtypes = [c_int, c_int, c_int, C.c_char_p, c_int]
        self.lib.jvk_set_job.restype = c_int
        self.lib.jvk_set_job.argtypes = [c_void_p, POINTER(c_int8), POINTER(c_int8),
            POINTER(c_int8), POINTER(c_int8), POINTER(c_uint32), POINTER(c_uint32),
            POINTER(c_int32), POINTER(c_int32), POINTER(c_uint32), c_int, c_int, c_int]
        self.lib.jvk_evaluate.restype = c_int
        self.lib.jvk_evaluate.argtypes = [c_void_p, POINTER(c_int32), POINTER(c_int32), c_int, POINTER(c_uint32)]
        self.lib.jvk_search.restype = c_int
        self.lib.jvk_search.argtypes = [c_void_p, c_int, c_int, c_int, c_int, c_int, c_int,
                                        POINTER(c_uint8), c_int, c_int64, POINTER(JvkHit)]
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
        from .pearl_noise_gpu import PearlNoiseGpu
        from .jackpot import SEED_LABEL_A, SEED_LABEL_B
        m, k = A.shape; n, _ = B.shape
        b_noise_seed, _ = commitment_hash
        ng = PearlNoiseGpu(*self._noise_ctx) if any(self._noise_ctx) else PearlNoiseGpu()
        e_al, _ = ng.uniform(SEED_LABEL_A, a_noise_seed, m, self.r, read_back=True)
        e_ar_t, _ = ng.perm(SEED_LABEL_A, a_noise_seed, k, self.r, read_back=True)
        e_bl, _ = ng.perm(SEED_LABEL_B, b_noise_seed, k, self.r, read_back=True)
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

    def search(self, mining_config, target: int, *, batch_size: int = 16384,
               max_attempts: int | None = None):
        """Native search: enumerate valid offsets, evaluate, return the first
        candidate whose hash (LE uint256) < target. Returns
        ``(Candidate_or_None, attempts, seconds)`` — matches JackpotGpu.search."""
        import time
        from .candidate_search import _axis_constraint, Candidate
        rp, cp = mining_config.rows_pattern, mining_config.cols_pattern
        r_mod, r_win = _axis_constraint(rp.shape)
        c_mod, c_win = _axis_constraint(cp.shape)
        r_upper = self._m - max(rp.to_list())
        c_upper = self._n - max(cp.to_list())
        tgt = np.frombuffer(int(target).to_bytes(32, "little"), dtype=np.uint8).copy()
        hit = JvkHit()
        t0 = time.time()
        rc = self.lib.jvk_search(self.ctx, r_mod, r_win, r_upper, c_mod, c_win, c_upper,
                                 _ptr(tgt, c_uint8), batch_size, int(max_attempts or 0), byref(hit))
        dt = time.time() - t0
        if rc != 0:
            raise RuntimeError(f"jvk_search failed ({rc})")
        if hit.found:
            hb = bytes(hit.hash)
            cand = Candidate(t_rows=hit.t_rows, t_cols=hit.t_cols,
                a_rows_indices=list(rp.indices_with_offset(hit.t_rows)),
                b_cols_indices=list(cp.indices_with_offset(hit.t_cols)),
                hash_jackpot=hb, target_value=int.from_bytes(hb, "little"))
            return cand, int(hit.attempts), dt
        return None, int(hit.attempts), dt

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
