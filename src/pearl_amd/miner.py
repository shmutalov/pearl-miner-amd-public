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
import queue
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

try:
    from .jackpot_coopmat import JackpotCoopmat  # amortized-GEMM + tensor cores
    _GPU_COOPMAT_AVAILABLE = True
except Exception:  # pragma: no cover — wrapper import only; DLL checked at build time
    JackpotCoopmat = None  # type: ignore[assignment]
    _GPU_COOPMAT_AVAILABLE = False


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
                 use_coopmat_jackpot: bool = False,
                 coopmat_shares_per_round: int = 64,
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
        # Coopmat jackpot (amortized GEMM on RDNA3 tensor cores, ~44x the
        # per-candidate Vulkan kernel). Specialized to the live pool pattern
        # (h=2, w=64); falls back to Vulkan/OpenCL if the pattern differs or the
        # DLL/shaders aren't built. Takes priority over Vulkan when enabled.
        self._use_coopmat_jackpot = use_coopmat_jackpot and _GPU_COOPMAT_AVAILABLE
        self._jackpot_coopmat = None
        self._jackpot_coopmat_shape: tuple[int, int, int, int] | None = None
        self._coopmat_shares_per_round = coopmat_shares_per_round
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

    def _ensure_jackpot_coopmat(self, h: int, w: int, r: int, k: int):
        """Lazily build the coopmat evaluator (amortized GEMM + tensor cores).
        Specialized to h=2,w=64; on any failure (unsupported pattern, DLL not
        built, device init) disable it and fall back to Vulkan/OpenCL."""
        if not self._use_coopmat_jackpot:
            return None
        if self._jackpot_coopmat is not None and self._jackpot_coopmat_shape == (h, w, r, k):
            return self._jackpot_coopmat
        nc = self._derive_gpu.context if self._derive_gpu is not None else None
        nq = self._derive_gpu.queue if self._derive_gpu is not None else None
        try:
            self._jackpot_coopmat = JackpotCoopmat(h, w, r, k, noise_context=nc, noise_queue=nq)
            self._jackpot_coopmat_shape = (h, w, r, k)
        except Exception as e:
            self._emit("jackpot_coopmat_unavailable", error=repr(e))
            self._use_coopmat_jackpot = False
            self._jackpot_coopmat = None
        return self._jackpot_coopmat

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
        # Priority: coopmat (tensor cores) > Vulkan > OpenCL. The coopmat path
        # is specialized to the pool pattern; on any failure it disables itself
        # here and we fall through to Vulkan/OpenCL.
        jcoop = self._ensure_jackpot_coopmat(h, w, r, k)
        if jcoop is not None:
            try:
                t0 = time.time()
                jcoop.set_job(self._A, self._B, mc.rows_pattern, mc.cols_pattern,
                              jm.commitment_hash(), jm.commitment_hash()[1])
                self._emit("jackpot_coopmat_set_job_done", seconds=time.time() - t0,
                           h=h, w=w, r=r)
            except Exception as e:
                self._emit("jackpot_coopmat_unavailable", error=repr(e))
                self._use_coopmat_jackpot = False
                self._jackpot_coopmat = None
                jcoop = None

        jvk = None if jcoop is not None else self._ensure_jackpot_vk(h, w, r)
        if jcoop is not None:
            pass
        elif jvk is not None:
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

    def _search_one_job(self, work: Work, state: MinerState,
                        should_continue=None) -> None:
        """Run candidate search until a hit, until the job changes, or until
        max_attempts is reached. Uses the GPU JackpotGpu evaluator when
        available (batched ~27k cand/s on RX 570), falls back to the pure-
        Python CPU loop (~80 cand/s) otherwise. ``should_continue`` is an
        optional ``() -> bool`` polled by the coopmat path so a Ctrl+C / stop
        request is honored within ~one tile rather than after the whole round."""
        def progress(attempts: int, dt: float, last_target: int = 0) -> None:
            self._emit("search_progress", attempts=attempts, seconds=dt,
                       rate=attempts / dt if dt > 0 else 0.0,
                       last_target_lz=(256 - last_target.bit_length()
                                       if last_target > 0 else 256))

        # Coopmat: one job-space yields many distinct shares (≈ share_difficulty
        # candidates per share). Stream them so host proof-build + pool submit
        # overlap the GPU search of the next tile (producer/consumer split).
        if self._jackpot_coopmat is not None:
            self._search_coopmat_pipelined(work, state, should_continue)
            return

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
        self._submit_hit(work, state, hit, attempts, dt)

    def _search_coopmat_pipelined(self, work: Work, state: MinerState,
                                  should_continue=None, *, job_slot: int = 0) -> None:
        """One coopmat round with host/GPU overlap. The GPU search (this thread,
        the producer) streams distinct shares tile-by-tile via
        ``JackpotCoopmat.search_all_stream`` reading double-buffered ``job_slot``;
        a single consumer thread builds the PlainProof and submits each share to
        the pool. Because the producer spends its time blocked in the GPU fence
        wait (GIL released) and submit I/O also releases the GIL, proof-build +
        network for one tile overlap the GPU search of the next, instead of
        serializing after the whole sweep. One round = one (A,B) job-space;
        bounded by coopmat_shares_per_round.

        ``should_continue`` (() -> bool) makes the round interruptible: the
        producer stops the sweep within ~one tile, and the consumer drops any
        still-queued shares without submitting, so Ctrl+C is honored promptly."""
        jcoop = self._jackpot_coopmat
        t0 = time.time()
        q: queue.Queue = queue.Queue(maxsize=64)
        err: list[BaseException] = []
        n_submitted = [0]

        def _consumer() -> None:
            while True:
                item = q.get()
                try:
                    if item is None:
                        return
                    if should_continue is not None and not should_continue():
                        continue        # stop requested: drop the rest, don't submit
                    self._submit_hit(work, state, item,
                                     jcoop.last_attempts, time.time() - t0)
                    n_submitted[0] += 1
                except BaseException as e:  # keep producer alive; surface later
                    if not err:
                        err.append(e)
                finally:
                    q.task_done()

        worker = threading.Thread(target=_consumer, name="coopmat-submit", daemon=True)
        worker.start()
        try:
            for cand in jcoop.search_all_stream(
                    work.mining_config, work.target,
                    max_return=self._coopmat_shares_per_round,
                    should_continue=should_continue, job_slot=job_slot):
                q.put(cand)
                if err:                # consumer died (e.g. pool/socket error)
                    break
        finally:
            q.put(None)                # sentinel: drain + stop the consumer
            worker.join()

        if err:
            raise err[0]
        if n_submitted[0] == 0:
            self._emit("search_exhausted", job_id=work.job_id,
                       attempts=jcoop.last_attempts, seconds=time.time() - t0)

    def _preflight_into(self, work: Work, miner_seed: bytes,
                        job_slot: int) -> MinerState:
        """Coopmat-only, side-effect-isolated preflight for the double-buffered
        continuous loop. Derives A/B from ``miner_seed``, builds Merkle layers,
        and builds PA/PB into ``job_slot`` — all into locals, never touching the
        shared ``self._A``/``self._B`` of the round currently searching the other
        slot. Safe to run on a prefetch thread (OpenCL derive/merkle + Vulkan
        set_job into the idle slot) concurrently with that search. Returns a
        self-contained MinerState the consumer can build proofs from."""
        from .proof_builder import derive_AB_from_seed
        m, n, k = work.m, work.n, work.mining_config.common_dim
        if self._derive_gpu is not None:
            A, B, A_buf, B_buf = self._derive_gpu.derive_AB(miner_seed, m, n, k)
        else:
            A, B = derive_AB_from_seed(miner_seed, m, n, k)
            A_buf = B_buf = None

        job_key = compute_job_key(work.incomplete_header_bytes, work.mining_config)
        if self._merkle_gpu is not None:
            if A_buf is not None and (m * k) % 1024 == 0:
                A_layers = self._merkle_gpu.build_layers(None, job_key, data_buf=A_buf, n_bytes=m * k)
                B_layers = self._merkle_gpu.build_layers(None, job_key, data_buf=B_buf, n_bytes=n * k)
            else:
                A_layers = self._merkle_gpu.build_layers(A.tobytes(), job_key)
                B_layers = self._merkle_gpu.build_layers(B.tobytes(), job_key)
        else:
            A_layers = build_merkle_tree(A.tobytes(), job_key)
            B_layers = build_merkle_tree(B.tobytes(), job_key)

        jm = JobMatrices(A=A, B=B, miner_seed=miner_seed,
                         hash_a=A_layers[-1][0], hash_b=B_layers[-1][0],
                         job_key=job_key, m=m, n=n, k=k)
        mc = work.mining_config
        self._jackpot_coopmat.set_job(A, B, mc.rows_pattern, mc.cols_pattern,
                                      jm.commitment_hash(), jm.commitment_hash()[1],
                                      job_slot=job_slot)
        return MinerState(miner_seed=miner_seed, A=A, B=B, job_key=jm.job_key,
                          job_matrices=jm, A_layers=A_layers, B_layers=B_layers)

    def mine_coopmat_continuous(self, work: Work, *, should_stop,
                                next_seed, on_round=None) -> None:
        """Continuously mine one coopmat pool job with cross-round preflight
        overlap. While round N searches + submits (active job slot), a prefetch
        thread runs round N+1's full preflight (derive + merkle + set_job) into
        the IDLE job slot. NJOB=2 double-buffers PA/PB/KEY so the rebuild overlaps
        the search instead of stalling the GPU between rounds.

        - ``should_stop() -> bool``: polled between and within rounds; the active
          search bails within ~one tile when it flips True.
        - ``next_seed() -> bytes``: 32-byte miner seed for the next round (coopmat
          refreshes the seed each round so the same pool job keeps yielding fresh
          distinct shares). Called once per round.
        - ``on_round()``: optional, invoked after each round's search completes.

        Job changes are handled by the caller: have ``should_stop`` flip True when
        a new pool job arrives, then re-enter with the new ``work``."""
        mc0 = work.mining_config
        jcoop = self._jackpot_coopmat or self._ensure_jackpot_coopmat(
            len(mc0.rows_pattern.to_list()), len(mc0.cols_pattern.to_list()),
            mc0.rank, mc0.common_dim)
        if jcoop is None:
            raise RuntimeError("mine_coopmat_continuous requires the coopmat evaluator")
        njob = jcoop.NJOB
        cont = lambda: not should_stop()

        # Round 0 preflight into slot 0 (blocking — nothing to overlap yet).
        state = self._preflight_into(work, next_seed(), job_slot=0)
        cur = 0
        while not should_stop():
            nxt = (cur + 1) % njob
            nxt_seed = next_seed()
            holder: dict = {}

            def _prefetch() -> None:
                # Build the NEXT round into the idle slot while this round searches.
                try:
                    holder["state"] = self._preflight_into(work, nxt_seed, job_slot=nxt)
                except BaseException as e:  # surface after join; don't kill the GPU thread
                    holder["err"] = e

            pf = threading.Thread(target=_prefetch, name="coopmat-preflight", daemon=True)
            pf.start()
            try:
                self._search_coopmat_pipelined(work, state, cont, job_slot=cur)
            finally:
                pf.join()                       # never leave the prefetch thread running
            if "err" in holder:
                raise holder["err"]
            state = holder["state"]             # promote the prebuilt slot to current
            cur = nxt
            if on_round is not None:
                on_round()

    def _submit_hit(self, work: Work, state: MinerState, hit,
                    attempts: int, dt: float) -> None:
        """Assemble the PlainProof for one hit and submit it."""
        self._emit("hit_found", job_id=work.job_id,
                   attempts=attempts, seconds=dt,
                   t_rows=hit.t_rows, t_cols=hit.t_cols,
                   hash=hit.hash_jackpot.hex())
        t1 = time.time()
        proof = build_plain_proof(state.job_matrices, work.mining_config,
                                  state.A_layers, state.B_layers, hit)
        proof_bytes = proof.encode()
        self._emit("proof_built", proof_bytes=len(proof_bytes),
                   build_seconds=time.time() - t1)
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
