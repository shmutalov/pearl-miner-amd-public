"""End-to-end Pearl pool miner orchestrator.

Wires together every layer landed in this iteration:

  StratumSession  ─►  Work {job_id, header_bytes, mining_config, job_key, target}
        │
        ▼
  build_job_matrices  ─►  A, B (int8) + hash_a, hash_b
        │
        ▼
  build_merkle_tree   ─►  A_layers, B_layers (full BLAKE3 Merkle CVs)
        │
        ▼
  search_candidate    ─►  Candidate(t_rows, t_cols) where hash_jackpot < target
        │
        ▼
  build_plain_proof   ─►  PlainProof { m, n, k, noise_rank, a, bt }
        │
        ▼
  PlainProof.encode   ─►  bincode bytes  ─►  session.submit_share

For the pool's m=n=131072 / k=4096 shape, the pure-Python Merkle build
(~10 min/matrix) and search (~12 ms/candidate) are too slow to compete.
This module's purpose is to lock in the *wiring* and the byte-level
contract; the OpenCL acceleration of derive_matrix / build_merkle_tree /
search_candidate plugs into the same orchestrator without changing it.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .candidate_search import Candidate, search_candidate
from .merkle_proof import build_merkle_tree, compute_leaf_indices_from_rows, get_multileaf_proof
from .mining_config import MiningConfiguration, compute_job_key
from .plain_proof_codec import MatrixMerkleProof, MerkleProof, PlainProof
from .proof_builder import JobMatrices
from .stratum_session import StratumSession, Work

try:
    from .merkle_gpu import MerkleGpu
    _GPU_MERKLE_AVAILABLE = True
except Exception:  # pragma: no cover — pyopencl missing or no GPU
    MerkleGpu = None  # type: ignore[assignment]
    _GPU_MERKLE_AVAILABLE = False

try:
    from .derive_matrix_gpu import DeriveMatrixGpu
    _GPU_DERIVE_AVAILABLE = True
except Exception:  # pragma: no cover — pyopencl missing or no GPU
    DeriveMatrixGpu = None  # type: ignore[assignment]
    _GPU_DERIVE_AVAILABLE = False

try:
    from .jackpot_gpu import JackpotGpu
    _GPU_JACKPOT_AVAILABLE = True
except Exception:  # pragma: no cover — pyopencl missing or no GPU
    JackpotGpu = None  # type: ignore[assignment]
    _GPU_JACKPOT_AVAILABLE = False

try:
    from .jackpot_vk import JackpotVk  # native Vulkan evaluator (vk/ artifacts)
    _GPU_VK_AVAILABLE = True
except Exception:  # pragma: no cover — wrapper import only; DLL checked at build time
    JackpotVk = None  # type: ignore[assignment]
    _GPU_VK_AVAILABLE = False


log = logging.getLogger("pearl_amd.miner")


@dataclass
class MinerState:
    """Per-job preflight state. A and B themselves outlive jobs (they depend
    only on miner_seed); the Merkle layers must be recomputed on every job_key
    change because the BLAKE3 tree is keyed."""

    miner_seed: bytes
    A: object              # np.ndarray int8 (m, k); kept as object to avoid numpy import here
    B: object              # np.ndarray int8 (n, k)
    job_key: bytes
    job_matrices: JobMatrices
    A_layers: list[list[bytes]]
    B_layers: list[list[bytes]]


def build_plain_proof(jm: JobMatrices, mc: MiningConfiguration,
                      A_layers: list[list[bytes]],
                      B_layers: list[list[bytes]],
                      candidate: Candidate) -> PlainProof:
    """Assemble a wire-ready ``PlainProof`` from a winning candidate.

    Extracts the touched leaves from both matrices, walks the Merkle layers
    to collect siblings, then packages into the bincode-compatible
    ``PlainProof`` dataclass from ``plain_proof_codec``.
    """
    A_bytes = jm.A.tobytes()
    B_bytes = jm.B.tobytes()

    a_leaf_indices = compute_leaf_indices_from_rows(candidate.a_rows_indices, jm.k)
    b_leaf_indices = compute_leaf_indices_from_rows(candidate.b_cols_indices, jm.k)

    a_proof = get_multileaf_proof(A_layers, A_bytes, a_leaf_indices)
    b_proof = get_multileaf_proof(B_layers, B_bytes, b_leaf_indices)

    return PlainProof(
        m=jm.m, n=jm.n, k=jm.k, noise_rank=mc.rank,
        a=MatrixMerkleProof(
            proof=MerkleProof(
                leaf_data=a_proof.leaf_data,
                leaf_indices=a_proof.leaf_indices,
                total_leaves=a_proof.total_leaves,
                root=a_proof.root,
                siblings=a_proof.siblings,
            ),
            row_indices=list(candidate.a_rows_indices),
        ),
        bt=MatrixMerkleProof(
            proof=MerkleProof(
                leaf_data=b_proof.leaf_data,
                leaf_indices=b_proof.leaf_indices,
                total_leaves=b_proof.total_leaves,
                root=b_proof.root,
                siblings=b_proof.siblings,
            ),
            row_indices=list(candidate.b_cols_indices),
        ),
    )


class PearlMiner:
    """Runs the search loop against a live ``StratumSession``.

    Caller pattern::

        with StratumSession(cfg, ...) as sess:
            sess.wait_for_work()
            miner = PearlMiner(sess, miner_seed=os.urandom(32))
            miner.run()      # blocks until session closes
    """

    def __init__(self, session: StratumSession, miner_seed: bytes,
                 max_attempts_per_job: int = 1_000_000,
                 on_event: Callable[[str, dict], None] | None = None,
                 use_gpu_merkle: bool = True,
                 use_gpu_derive: bool = True,
                 use_gpu_jackpot: bool = True,
                 use_vulkan_jackpot: bool = False,
                 jackpot_batch_size: int = 8192) -> None:
        if len(miner_seed) != 32:
            raise ValueError("miner_seed must be 32 bytes")
        self.session = session
        self.miner_seed = miner_seed
        self.max_attempts_per_job = max_attempts_per_job
        self.jackpot_batch_size = jackpot_batch_size
        self._on_event = on_event or (lambda kind, info: log.info("%s %s", kind, info))
        self._A = None
        self._B = None
        self._A_buf = None       # cl.Buffer for GPU-resident A (None on CPU path)
        self._B_buf = None       # cl.Buffer for GPU-resident B
        self._state: MinerState | None = None
        self._merkle_gpu: MerkleGpu | None = None
        self._derive_gpu: DeriveMatrixGpu | None = None
        self._jackpot_gpu: JackpotGpu | None = None
        self._jackpot_shape: tuple[int, int, int] | None = None
        # Vulkan jackpot (experiments/vk_jackpot): ~2.6x the OpenCL kernel.
        # When enabled it takes over set_job + search; A/B are uploaded host->
        # Vulkan once per job (noise derived via the shared OpenCL context).
        self._use_vulkan_jackpot = use_vulkan_jackpot and _GPU_VK_AVAILABLE
        self._jackpot_vk = None
        self._jackpot_vk_shape: tuple[int, int, int] | None = None
        # OpenCL jackpot stays available as a fallback if Vulkan fails to build.
        self._use_gpu_jackpot = use_gpu_jackpot and _GPU_JACKPOT_AVAILABLE
        # Share one OpenCL context across the GPU helpers so device buffers
        # produced by derive can be consumed by merkle / jackpot without
        # round-tripping 512 MiB matrices through host memory.
        if use_gpu_derive and _GPU_DERIVE_AVAILABLE:
            self._derive_gpu = DeriveMatrixGpu()
            if use_gpu_merkle and _GPU_MERKLE_AVAILABLE:
                self._merkle_gpu = MerkleGpu(
                    context=self._derive_gpu.context,
                    queue=self._derive_gpu.queue)
        elif use_gpu_merkle and _GPU_MERKLE_AVAILABLE:
            self._merkle_gpu = MerkleGpu()

    def _ensure_jackpot_gpu(self, h: int, w: int, r: int) -> "JackpotGpu | None":
        """Lazily build (or rebuild) the JackpotGpu for this (h, w, r).
        The kernel is JIT-compiled with PEARL_H / PEARL_W / PEARL_R as
        compile-time constants, so we need a fresh program if the shape
        ever changes mid-session (pool shape is constant in practice)."""
        if not self._use_gpu_jackpot:
            return None
        if self._jackpot_gpu is not None and self._jackpot_shape == (h, w, r):
            return self._jackpot_gpu
        # Share context with derive_gpu (and therefore merkle_gpu) so we
        # can pass the same A / B device buffers without re-uploading.
        if self._derive_gpu is not None:
            self._jackpot_gpu = JackpotGpu(h, w, r,
                                           context=self._derive_gpu.context,
                                           queue=self._derive_gpu.queue)
        else:
            self._jackpot_gpu = JackpotGpu(h, w, r)
        self._jackpot_shape = (h, w, r)
        return self._jackpot_gpu

    def _ensure_jackpot_vk(self, h: int, w: int, r: int):
        """Lazily build the Vulkan jackpot evaluator. Noise is derived on the
        shared OpenCL context; A/B are uploaded host->Vulkan in set_job. If the
        native DLL/shaders aren't built, disable Vulkan and fall back to OpenCL."""
        if not self._use_vulkan_jackpot:
            return None
        if self._jackpot_vk is not None and self._jackpot_vk_shape == (h, w, r):
            return self._jackpot_vk
        nc = self._derive_gpu.context if self._derive_gpu is not None else None
        nq = self._derive_gpu.queue if self._derive_gpu is not None else None
        try:
            self._jackpot_vk = JackpotVk(h, w, r, noise_context=nc, noise_queue=nq)
            self._jackpot_vk_shape = (h, w, r)
        except Exception as e:  # DLL/shaders not built, or device init failed
            self._emit("jackpot_vk_unavailable", error=repr(e))
            self._use_vulkan_jackpot = False
            self._jackpot_vk = None
        return self._jackpot_vk

    def _emit(self, kind: str, **info: object) -> None:
        self._on_event(kind, info)

    def _preflight(self, work: Work) -> MinerState:
        """Build (or refresh) A, B, Merkle layers for the given Work.

        A and B are derived from ``miner_seed`` once and cached. Merkle
        layers depend on ``work.job_key``; recomputed whenever the pool
        ships a new job.
        """
        from .proof_builder import derive_AB_from_seed
        m, n, k = work.m, work.n, work.mining_config.common_dim
        if self._A is None or self._A.shape != (m, k) or self._B.shape != (n, k):
            t0 = time.time()
            if self._derive_gpu is not None:
                self._A, self._B, self._A_buf, self._B_buf = \
                    self._derive_gpu.derive_AB(self.miner_seed, m, n, k)
                backend = "gpu"
            else:
                self._A, self._B = derive_AB_from_seed(self.miner_seed, m, n, k)
                self._A_buf = self._B_buf = None
                backend = "cpu"
            self._emit("derive_AB_done", m=m, n=n, k=k,
                       seconds=time.time() - t0, backend=backend)

        job_key = compute_job_key(work.incomplete_header_bytes, work.mining_config)

        t0 = time.time()
        if self._merkle_gpu is not None:
            # Pool shape m*k = 512 MiB is always a multiple of CHUNK_LEN=1024,
            # so the device-buffer path applies. When buffers are present
            # (= GPU derive ran), reuse them directly to skip 2x512 MiB upload.
            if self._A_buf is not None and (m * k) % 1024 == 0:
                A_layers = self._merkle_gpu.build_layers(
                    None, job_key, data_buf=self._A_buf, n_bytes=m * k)
                t1 = time.time()
                B_layers = self._merkle_gpu.build_layers(
                    None, job_key, data_buf=self._B_buf, n_bytes=n * k)
            else:
                A_layers = self._merkle_gpu.build_layers(self._A.tobytes(), job_key)
                t1 = time.time()
                B_layers = self._merkle_gpu.build_layers(self._B.tobytes(), job_key)
            backend = "gpu"
        else:
            A_layers = build_merkle_tree(self._A.tobytes(), job_key)
            t1 = time.time()
            B_layers = build_merkle_tree(self._B.tobytes(), job_key)
            backend = "cpu"
        t2 = time.time()
        self._emit("merkle_done", a_seconds=t1 - t0, b_seconds=t2 - t1,
                   a_leaves=len(A_layers[0]), b_leaves=len(B_layers[0]),
                   backend=backend)

        # Merkle root == BLAKE3-keyed root of the matrix → use it for
        # commitment_hash instead of a separate CPU BLAKE3 pass (~0.5 s
        # saved at pool shape). The last layer always contains exactly
        # one element (the root) by construction.
        hash_a = A_layers[-1][0]
        hash_b = B_layers[-1][0]
        jm = JobMatrices(A=self._A, B=self._B, miner_seed=self.miner_seed,
                         hash_a=hash_a, hash_b=hash_b, job_key=job_key,
                         m=m, n=n, k=k)

        # Wire up the GPU jackpot evaluator for this job. The kernel
        # constants depend on the candidate-tile shape (h, w, r), so we
        # may JIT-compile on the first preflight; subsequent jobs with
        # the same shape reuse the cached program. set_job re-uploads
        # the noise matrices + patterns + key (cheap with shared ctx).
        mc = work.mining_config
        h = len(mc.rows_pattern.to_list())
        w = len(mc.cols_pattern.to_list())
        r = mc.rank
        # Prefer the Vulkan evaluator when enabled; else the OpenCL one (which
        # can reuse the on-device A/B buffers from derive without re-uploading).
        jvk = self._ensure_jackpot_vk(h, w, r)
        if jvk is not None:
            t0 = time.time()
            jvk.set_job(self._A, self._B,
                        mc.rows_pattern.to_list(), mc.cols_pattern.to_list(),
                        jm.commitment_hash(), jm.commitment_hash()[1])
            self._emit("jackpot_vk_set_job_done", seconds=time.time() - t0,
                       h=h, w=w, r=r)
        else:
            jpg = self._ensure_jackpot_gpu(h, w, r)
            if jpg is not None:
                t0 = time.time()
                jpg.set_job(self._A, self._B,
                            mc.rows_pattern.to_list(), mc.cols_pattern.to_list(),
                            jm.commitment_hash(), jm.commitment_hash()[1],
                            A_buf=self._A_buf, B_buf=self._B_buf)
                self._emit("jackpot_set_job_done", seconds=time.time() - t0,
                           h=h, w=w, r=r)

        return MinerState(
            miner_seed=self.miner_seed,
            A=self._A, B=self._B,
            job_key=jm.job_key,
            job_matrices=jm,
            A_layers=A_layers, B_layers=B_layers,
        )

    def _search_one_job(self, work: Work, state: MinerState) -> None:
        """Run candidate search until a hit, until the job changes, or until
        max_attempts is reached. Uses the GPU JackpotGpu evaluator when
        available (batched ~27k cand/s on RX 570), falls back to the pure-
        Python CPU loop (~80 cand/s) otherwise."""
        def progress(attempts: int, dt: float, last_target: int = 0) -> None:
            self._emit("search_progress", attempts=attempts, seconds=dt,
                       rate=attempts / dt if dt > 0 else 0.0,
                       last_target_lz=(256 - last_target.bit_length()
                                       if last_target > 0 else 256))

        t0 = time.time()
        if self._jackpot_vk is not None:
            # Native search loop runs to a hit / max_attempts in one call (no
            # mid-run progress_cb); the miner bounds it via max_attempts_per_job.
            hit, attempts, dt = self._jackpot_vk.search(
                work.mining_config, work.target,
                batch_size=self.jackpot_batch_size,
                max_attempts=self.max_attempts_per_job)
        elif self._jackpot_gpu is not None:
            hit, attempts, dt = self._jackpot_gpu.search(
                work.mining_config, work.target,
                batch_size=self.jackpot_batch_size,
                max_attempts=self.max_attempts_per_job,
                progress_cb=progress)
        else:
            hit, attempts, dt = search_candidate(
                state.job_matrices, work.mining_config, work.target,
                max_attempts=self.max_attempts_per_job,
                report_every=1000, progress_cb=progress)

        if hit is None:
            self._emit("search_exhausted", job_id=work.job_id,
                       attempts=attempts, seconds=dt)
            return

        self._emit("hit_found", job_id=work.job_id,
                   attempts=attempts, seconds=dt,
                   t_rows=hit.t_rows, t_cols=hit.t_cols,
                   hash=hit.hash_jackpot.hex())

        t1 = time.time()
        proof = build_plain_proof(state.job_matrices, work.mining_config,
                                  state.A_layers, state.B_layers, hit)
        proof_bytes = proof.encode()
        t2 = time.time()
        self._emit("proof_built", proof_bytes=len(proof_bytes),
                   build_seconds=t2 - t1)

        t3 = time.time()
        resp = self.session.submit_share(work.job_id, proof_bytes)
        self._emit("submit_done", response=resp, seconds=time.time() - t3)

    def run(self) -> None:
        """Main loop. Returns when the session disconnects."""
        last_job_key: bytes | None = None
        while not self.session.is_disconnected():
            try:
                work = self.session.wait_for_work(timeout=30.0)
            except (TimeoutError, ConnectionError) as e:
                self._emit("wait_failed", error=repr(e))
                return

            if work.job_key != last_job_key:
                self._emit("job_change", job_id=work.job_id,
                           height=work.block_height,
                           target_lz=256 - work.target.bit_length() if work.target > 0 else 256)
                self._state = self._preflight(work)
                last_job_key = work.job_key

            assert self._state is not None
            self._search_one_job(work, self._state)
