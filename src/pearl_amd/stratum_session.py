"""High-level session wrapper around :class:`StratumClient`.

Lives one layer above the raw wire client and owns the *current view* of pool
state: latest mining job, latest matrix-shape params, latest share difficulty.
A downstream miner thread reads ``session.current_work()`` and submits via
``session.submit_share()``; both are thread-safe.

Job notifications from the pool arrive on the stratum reader thread; this
class shields callers from that thread and exposes a clean blocking/non-
blocking API.

Note on submit format: confirmed by live probe on 2026-05-24 against
``eu1.alphapool.tech:5566`` — pool wants exactly
``mining.submit params = [worker_username, job_id, base64(plain_proof_bytes)]``
and returns ``{"result": true}`` once params are parsed (this is NOT a
share-accepted ack; real proof validation is async).
"""
from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from .mining_config import (
    MiningConfiguration,
    compute_job_key,
    mining_config_from_pool_params,
    nbits_hex_to_target,
)
from .stratum_client import Job, MiningParams, StratumClient, StratumConfig


@dataclass
class Work:
    """A snapshot of everything a miner needs to process one job."""

    job_id: str
    incomplete_header_bytes: bytes
    prev_hash: bytes
    block_height: int
    ntime_hex: str
    share_nbits: str
    clean_jobs: bool
    mining_params: dict[str, Any]      # m, n, k, rank, rows_pattern, cols_pattern, mma_type
    share_difficulty: float
    received_at: float                 # time.monotonic()

    # Derived (computed in _maybe_emit_work) — convenient for downstream miners:
    m: int = 0                         # = mining_params["m"]
    n: int = 0                         # = mining_params["n"]
    mining_config: MiningConfiguration | None = None
    job_key: bytes = b""               # 32 bytes, blake3(header || mining_config.to_bytes())
    target: int = 0                    # 256-bit int decoded from share_nbits


def _parse_notify(job: Job) -> dict[str, Any]:
    """Decode the positional ``mining.notify`` params into named fields.

    Field order confirmed by live capture; see ``docs/pearl-stratum-protocol.md``.
    """
    if not isinstance(job.raw, list) or len(job.raw) < 7:
        raise ValueError(f"unexpected mining.notify payload: {job.raw!r}")
    job_id, prev_hex, header_hex, height, ntime_hex, nbits_hex, clean = job.raw[:7]
    return {
        "job_id": str(job_id),
        "prev_hash": bytes.fromhex(prev_hex),
        "incomplete_header_bytes": bytes.fromhex(header_hex),
        "block_height": int(height),
        "ntime_hex": str(ntime_hex),
        "share_nbits": str(nbits_hex),
        "clean_jobs": bool(clean),
    }


def _parse_mining_params(mp: MiningParams) -> dict[str, Any]:
    """Pool sends ``pearl.set_mining_params`` as a single-element array; the
    wire-level dispatcher already unwraps it for our local schema. Pull out
    the inner object if it's still wrapped."""
    raw = mp.raw
    if isinstance(raw, dict) and set(raw.keys()) == {"params"} and isinstance(raw["params"], list):
        if len(raw["params"]) == 1 and isinstance(raw["params"][0], dict):
            return dict(raw["params"][0])
        return {"params": raw["params"]}
    return dict(raw)


class StratumSession:
    """Stateful wrapper around :class:`StratumClient`.

    Usage::

        sess = StratumSession(cfg)
        sess.start()                 # connects + handshakes (blocks)
        work = sess.wait_for_work()  # blocks until first complete work unit
        # ... compute a proof ...
        sess.submit_share(work.job_id, proof_bytes)
        sess.stop()
    """

    def __init__(self, cfg: StratumConfig,
                 on_work_update: Callable[[Work], None] | None = None,
                 on_log: Callable[[str], None] | None = None) -> None:
        self.cfg = cfg
        self._on_work_update = on_work_update
        self._on_log = on_log or (lambda s: print(s))

        self._client: StratumClient | None = None
        self._lock = threading.Lock()
        self._latest_params: dict[str, Any] | None = None
        self._latest_diff: float | None = None
        self._latest_notify: dict[str, Any] | None = None
        self._latest_work: Work | None = None
        self._work_ready = threading.Event()
        self._notify_count = 0
        self._param_change_count = 0

    # -- lifecycle -------------------------------------------------------- #

    def start(self) -> None:
        client = StratumClient(
            self.cfg,
            on_set_difficulty=self._on_set_difficulty,
            on_mining_params=self._on_mining_params,
            on_notify=self._on_notify,
            on_log=self._on_log,
        )
        client.connect()
        client.handshake()
        self._client = client

    def stop(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "StratumSession":
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    # -- inbound notification handlers (run on the reader thread) -------- #

    def _on_set_difficulty(self, params: Any) -> None:
        # mining.set_difficulty params is a single-element array
        try:
            val = float(params[0]) if isinstance(params, list) else float(params)
        except (TypeError, ValueError, IndexError):
            val = float("nan")
        with self._lock:
            self._latest_diff = val
        self._on_log(f"  [session] set_difficulty={val}")
        self._maybe_emit_work()

    def _on_mining_params(self, mp: MiningParams) -> None:
        params = _parse_mining_params(mp)
        with self._lock:
            self._latest_params = params
            self._param_change_count += 1
        self._on_log(
            f"  [session] mining_params m={params.get('m')} n={params.get('n')} "
            f"k={params.get('k')} rank={params.get('rank')} mma={params.get('mma_type')!r}"
        )
        self._maybe_emit_work()

    def _on_notify(self, job: Job) -> None:
        try:
            parsed = _parse_notify(job)
        except Exception as e:
            self._on_log(f"  [session] malformed notify: {e}")
            return
        with self._lock:
            self._latest_notify = parsed
            self._notify_count += 1
        self._on_log(
            f"  [session] notify job_id={parsed['job_id']} height={parsed['block_height']} "
            f"clean={parsed['clean_jobs']}"
        )
        self._maybe_emit_work()

    def _maybe_emit_work(self) -> None:
        with self._lock:
            if (self._latest_notify is None
                    or self._latest_params is None
                    or self._latest_diff is None):
                return
            params = dict(self._latest_params)
            header = self._latest_notify["incomplete_header_bytes"]
            try:
                mining_config = mining_config_from_pool_params(params)
                job_key = compute_job_key(header, mining_config)
            except Exception as e:
                self._on_log(f"  [session] failed to derive mining_config/job_key: {e!r}")
                return
            try:
                # Per-tile share target (Akoya/ARC):  target = diff_target * DAF.
                #
                #   diff_target — the pdiff target for the current difficulty:
                #     * object-notify pools (HeroMiners) carry it directly in the
                #       notify ("explicit_target");
                #     * pearl/v1 array-notify pools (AlphaPool) carry NO target, so
                #       synthesize it from the last set_difficulty:
                #           diff_target = Diff1Target / D,  Diff1Target = 0xFFFF<<208.
                #
                #   DAF = rows.size * cols.size * dot_product_length  (= 2^19 for the
                #     live shape: 2*64*4096). Each found tile is one "attempt"
                #     representing that many MACs, so the protocol scales the target
                #     UP by it. Akoya GpuWorker.InstallSigmaHalf: adjusted =
                #     NbitsToTarget(nbits) * DifficultyAdjustmentFactor().
                #
                # History: the old `nbits<<32` (factor 2^32) was 2^13 too LOOSE vs
                # the DAF and submitted sub-target garbage -> "too many bad proofs"
                # ban; a brief pdiff-only fix dropped the DAF (2^19 too STRICT).
                # diff_target * DAF is the correct middle (lz~28 at D=50000, which
                # matches the lz~29 share we saw accepted).
                DIFF1_TARGET = 0xFFFF << 208
                explicit = self._latest_notify.get("explicit_target")
                if explicit is not None:
                    diff_target = int(explicit)
                elif self._latest_diff and float(self._latest_diff) > 0:
                    diff_target = max(1, DIFF1_TARGET // int(float(self._latest_diff)))
                else:
                    diff_target = DIFF1_TARGET          # diff=1 fallback
                daf = mining_config.difficulty_adjustment_factor()
                target = min(diff_target * daf, (1 << 256) - 1)
            except Exception as e:
                self._on_log(f"  [session] failed to derive share target: {e!r}")
                return
            work = Work(
                job_id=self._latest_notify["job_id"],
                incomplete_header_bytes=header,
                prev_hash=self._latest_notify["prev_hash"],
                block_height=self._latest_notify["block_height"],
                ntime_hex=self._latest_notify["ntime_hex"],
                share_nbits=self._latest_notify["share_nbits"],
                clean_jobs=self._latest_notify["clean_jobs"],
                mining_params=params,
                share_difficulty=self._latest_diff,
                received_at=time.monotonic(),
                m=int(params["m"]),
                n=int(params["n"]),
                mining_config=mining_config,
                job_key=job_key,
                target=target,
            )
            self._latest_work = work
        self._work_ready.set()
        if self._on_work_update:
            try:
                self._on_work_update(work)
            except Exception as e:
                self._on_log(f"  [session] on_work_update raised: {e!r}")

    # -- consumer API ----------------------------------------------------- #

    def wait_for_work(self, timeout: float | None = None) -> Work:
        """Block until at least one (notify + set_mining_params + set_difficulty)
        triple has been received and return the latest snapshot."""
        if not self._work_ready.wait(timeout):
            if self.is_disconnected():
                raise ConnectionError("connection lost before any work arrived")
            raise TimeoutError("no complete work seen within timeout")
        if self.is_disconnected():
            raise ConnectionError("connection lost before any work arrived")
        with self._lock:
            assert self._latest_work is not None
            return self._latest_work

    def current_work(self) -> Work | None:
        with self._lock:
            return self._latest_work

    def is_disconnected(self) -> bool:
        return self._client is None or self._client.is_disconnected()

    def wait_until_disconnected(self, timeout: float | None = None) -> bool:
        """Block until the underlying TCP socket reports closed, or timeout."""
        assert self._client is not None, "call start() first"
        return self._client.wait_until_disconnected(timeout)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"notify_count": self._notify_count,
                    "param_changes": self._param_change_count}

    # -- submit ----------------------------------------------------------- #

    def submit_share(self, job_id: str, plain_proof: bytes) -> dict[str, Any]:
        """Wire-level share submission. ``plain_proof`` is the raw Pearl
        PlainProof bytes (the same payload the solo gateway expects in
        ``submitPlainProof``). We base64-encode before sending.

        Returns the JSON-RPC response object. ``result: true`` only means
        the pool parsed the params — full proof verification is async.
        """
        assert self._client is not None, "call start() first"
        return self._client.submit_share(job_id, plain_proof)


# --------------------------------------------------------------------------- #
# Reconnect-aware session runner                                              #
# --------------------------------------------------------------------------- #

def run_session_with_reconnect(
    cfg: StratumConfig,
    on_work_update: Callable[[Work], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    on_session_start: Callable[[StratumSession], None] | None = None,
    backoff_initial_sec: float = 2.0,
    backoff_max_sec: float = 60.0,
    stop: threading.Event | None = None,
) -> None:
    """Keep a StratumSession alive indefinitely, transparently handling
    pool-side idle disconnects and other network failures with exponential
    backoff.

    Pool's anti-leech drops idle workers after ~60s. A real miner submits
    shares fast enough to stay alive on its own; for monitoring or
    long-observation use cases, this wrapper just reconnects each time.

    Loops forever unless ``stop`` is set; on each successful connect the
    optional ``on_session_start(sess)`` callback is invoked so callers can
    register handlers or kick off a miner thread bound to that session.
    """
    log = on_log or (lambda s: print(s))
    stop = stop or threading.Event()
    backoff = backoff_initial_sec
    while not stop.is_set():
        try:
            with StratumSession(cfg, on_work_update=on_work_update, on_log=log) as sess:
                if on_session_start is not None:
                    on_session_start(sess)
                backoff = backoff_initial_sec  # reset on successful connect
                sess.wait_until_disconnected()
                log("  [run] session disconnected, reconnecting...")
        except Exception as e:
            log(f"  [run] session failed: {e!r}; backoff {backoff:.0f}s")
            if stop.wait(backoff):
                break
            backoff = min(backoff * 2, backoff_max_sec)
