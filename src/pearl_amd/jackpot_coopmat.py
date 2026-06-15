"""ctypes wrapper around libjackpot_coopmat_vk: the amortized-GEMM +
cooperative_matrix Pearl miner (RDNA3 tensor cores, ~44x the per-candidate
Vulkan kernel).

Specialized to the live pool pattern: rows_pattern=[0,32] (h=2), cols_pattern=
[0..63] (w=64). set_job builds the global noised matrices PA/PB device-local;
search() sweeps the (band x block) grid on tensor cores in TDR-safe tiles and
returns the first candidate whose hash (LE-uint256) is below target. Surface is
JackpotGpu.search-compatible so the miner can route through it.

Noise convention matches jackpot.evaluate_candidate (proven bit-identical):
``noise_rank = r`` for both the uniform width and the perm index range.
"""
from __future__ import annotations
import ctypes as C
from ctypes import (c_void_p, c_int, c_int64, c_double, POINTER, byref,
                    c_int8, c_uint32, c_uint8)
from pathlib import Path
import numpy as np

_VK = Path(__file__).resolve().parent / "vk"
_DLL = _VK / "jackpot_coopmat_vk.dll"

# The kernel hardcodes this geometry.
_POOL_ROWS = [0, 32]
_POOL_COLS = list(range(64))


def _ptr(arr, ctype):
    return arr.ctypes.data_as(POINTER(ctype))


class JackpotCoopmat:
    def __init__(self, h: int, w: int, r: int, k: int, *, wg: int = 256,
                 subgroup_size: int = 32, max_hits: int = 4096,
                 noise_context=None, noise_queue=None, noise_device=None):
        if h != 2 or w != 64:
            raise ValueError(f"coopmat path is specialized to h=2,w=64 (got h={h},w={w})")
        self.h, self.w, self.r, self.k = h, w, r, k
        pmat = _VK / f"pmat_r{r}.spv"
        search = _VK / f"jackpot_coopmat_r{r}_wg{wg}.spv"
        for p in (pmat, search):
            if not p.exists():
                raise FileNotFoundError(f"shader {p} not built (run src/pearl_amd/vk/build.sh)")
        self.lib = C.CDLL(str(_DLL))
        self.lib.jcm_create.restype = c_void_p
        self.lib.jcm_create.argtypes = [C.c_char_p, C.c_char_p, c_int, c_int, c_int, c_int]
        self.lib.jcm_set_job.restype = c_int
        self.lib.jcm_set_job.argtypes = [c_void_p, POINTER(c_int8), POINTER(c_int8),
            POINTER(c_int8), POINTER(c_int8), POINTER(c_uint32), POINTER(c_uint32),
            POINTER(c_uint32), c_int, c_int]
        self.lib.jcm_search_tile.restype = c_int
        self.lib.jcm_search_tile.argtypes = [c_void_p, POINTER(c_uint8), c_int64, c_int,
                                             POINTER(c_uint32), POINTER(c_int)]
        self.lib.jcm_last_gpu_ms.restype = c_double
        self.lib.jcm_last_gpu_ms.argtypes = [c_void_p]
        self.lib.jcm_destroy.argtypes = [c_void_p]
        self.max_hits = max_hits
        self.ctx = self.lib.jcm_create(str(pmat).encode(), str(search).encode(),
                                       k, r, subgroup_size, max_hits)
        if not self.ctx:
            raise RuntimeError("jcm_create failed")
        self._m = self._n = 0
        self._noise_ctx = (noise_context, noise_queue, noise_device)
        self._records = np.empty((max_hits, 10), dtype=np.uint32)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_pattern(row_pattern, col_pattern):
        rl = [int(x) for x in row_pattern.to_list()]
        cl = [int(x) for x in col_pattern.to_list()]
        if rl != _POOL_ROWS or cl != _POOL_COLS:
            raise ValueError(
                f"coopmat path requires rows_pattern={_POOL_ROWS}, cols_pattern=[0..63]; "
                f"got rows={rl}, cols={cl[:4]}...")

    def set_job_raw(self, A, B, e_al, e_br_t, e_ar_t, e_bl, a_noise_seed):
        A = np.ascontiguousarray(A, dtype=np.int8)
        B = np.ascontiguousarray(B, dtype=np.int8)
        m, k = A.shape; n, k2 = B.shape
        assert k == k2 == self.k
        e_al = np.ascontiguousarray(e_al, dtype=np.int8).reshape(-1)
        e_br_t = np.ascontiguousarray(e_br_t, dtype=np.int8).reshape(-1)
        e_ar_t = np.ascontiguousarray(e_ar_t, dtype=np.uint32).reshape(-1)
        e_bl = np.ascontiguousarray(e_bl, dtype=np.uint32).reshape(-1)
        key = np.frombuffer(a_noise_seed, dtype=np.uint32).copy()
        self._hold = (A, B, e_al, e_br_t, e_ar_t, e_bl, key)
        rc = self.lib.jcm_set_job(self.ctx, _ptr(A, c_int8), _ptr(B, c_int8),
            _ptr(e_al, c_int8), _ptr(e_br_t, c_int8), _ptr(e_ar_t, c_uint32), _ptr(e_bl, c_uint32),
            _ptr(key, c_uint32), m, n)
        if rc != 0:
            raise RuntimeError(f"jcm_set_job failed ({rc})")
        self._m, self._n = m, n

    def set_job(self, A, B, row_pattern, col_pattern, commitment_hash, a_noise_seed):
        """JackpotGpu-compatible. Derives noise (noise_rank=r) then builds PA/PB."""
        self._check_pattern(row_pattern, col_pattern)
        from .pearl_noise_gpu import PearlNoiseGpu
        from .jackpot import SEED_LABEL_A, SEED_LABEL_B
        m, k = A.shape; n, _ = B.shape
        b_noise_seed, _ = commitment_hash
        ng = PearlNoiseGpu(*self._noise_ctx) if any(self._noise_ctx) else PearlNoiseGpu()
        e_al, _ = ng.uniform(SEED_LABEL_A, a_noise_seed, m, self.r, read_back=True)
        e_ar_t, _ = ng.perm(SEED_LABEL_A, a_noise_seed, k, self.r, read_back=True)
        e_bl, _ = ng.perm(SEED_LABEL_B, b_noise_seed, k, self.r, read_back=True)
        e_br_t, _ = ng.uniform(SEED_LABEL_B, b_noise_seed, n, self.r, read_back=True)
        self.set_job_raw(A, B, e_al, e_br_t, e_ar_t, e_bl, a_noise_seed)

    # ------------------------------------------------------------------ #
    def search(self, mining_config, target: int, *, batch_size: int = 16384,
               max_attempts: int | None = None, chunk_wg: int = 300_000):
        """Sweep the (band x block) grid in TDR-safe tiles; return the first
        candidate below target as ``(Candidate_or_None, attempts, seconds)``.
        ``batch_size`` is accepted for JackpotGpu.search compatibility (the tile
        size is governed by ``chunk_wg``)."""
        import time
        from .candidate_search import Candidate
        rp, cp = mining_config.rows_pattern, mining_config.cols_pattern
        self._check_pattern(rp, cp)
        nbands, nblocks = self._m // 64, self._n // 64
        total_wg = nbands * nblocks
        wg_cap = total_wg if not max_attempts else min(total_wg, (int(max_attempts) + 31) // 32)
        tgt = np.frombuffer(int(target).to_bytes(32, "little"), dtype=np.uint8).copy()
        cnt = c_int(0)
        attempts = 0
        t0 = time.time()
        wg_off = 0
        while wg_off < wg_cap:
            this = min(chunk_wg, wg_cap - wg_off)
            rc = self.lib.jcm_search_tile(self.ctx, _ptr(tgt, c_uint8), wg_off, this,
                                          _ptr(self._records, c_uint32), byref(cnt))
            if rc != 0:
                raise RuntimeError(f"jcm_search_tile failed ({rc})")
            attempts += this * 32
            if cnt.value > 0:
                rec = self._records[0]              # any hit below target suffices
                t_r, t_c = int(rec[0]), int(rec[1])
                hb = rec[2:10].astype("<u4").tobytes()
                cand = Candidate(t_rows=t_r, t_cols=t_c,
                    a_rows_indices=list(rp.indices_with_offset(t_r)),
                    b_cols_indices=list(cp.indices_with_offset(t_c)),
                    hash_jackpot=hb, target_value=int.from_bytes(hb, "little"))
                return cand, attempts, time.time() - t0
            wg_off += this
        return None, attempts, time.time() - t0

    def last_gpu_ms(self) -> float:
        return float(self.lib.jcm_last_gpu_ms(self.ctx))

    def close(self):
        if getattr(self, "ctx", None):
            self.lib.jcm_destroy(self.ctx); self.ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
