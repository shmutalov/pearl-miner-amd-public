"""Pearl stratum client (the ``pearl/v1`` dialect spoken by AlphaPool).

Reconstructed from strings recovered out of the official ``alpha-miner`` binary
(see ``alpha-miner-strings.txt`` at the repo root) + a live probe against
``eu1.alphapool.tech:5566``. The wire is newline-delimited JSON-RPC 2.0 over
plain TCP.

Handshake (client point of view, in order):

  -> mining.configure   params: [["pearl/v1"], {}]
  -> mining.subscribe   params: ["alpha-miner/0.1"]   (user-agent string)
  <- pearl.challenge    params: {"seed": <hex32>, "difficulty": <int bits>}
        (repeated until a valid pearl.challenge_response is sent)
  -> pearl.challenge_response  params: {"seed": <same hex>, "nonce": <hex>}
  -> mining.authorize   params: ["<bech32_address>[.<worker>]", "x[;d=N]"]
  <- mining.set_difficulty
  <- pearl.set_mining_params  -> matrix shape m/n/k/rank + row/col patterns
  <- mining.notify           -> job_id + header bytes + share target
  ... loop ...
  -> mining.submit (when a share is found)

The challenge gate is a BLAKE3 proof-of-work: find a 64-bit nonce such that
``BLAKE3(seed_bytes_32 || u64_le(nonce))`` has at least ``difficulty`` leading
zero bits. The official miner solves this on GPU; here we parallelise across
CPU cores with multiprocessing.

This module deliberately does NOT compute shares — it only speaks the wire
protocol and surfaces the work units to the caller via callbacks.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import blake3  # type: ignore
except ImportError as e:  # pragma: no cover
    raise SystemExit("This client needs the 'blake3' package: pip install blake3") from e


# --------------------------------------------------------------------------- #
# BLAKE3 challenge solver                                                     #
# --------------------------------------------------------------------------- #

def _leading_zero_bits(digest: bytes) -> int:
    n = 0
    for b in digest:
        if b == 0:
            n += 8
            continue
        # count leading zeros in a byte
        x = b
        while not (x & 0x80):
            n += 1
            x <<= 1
        break
    return n


def _solver_worker(seed_hex: str, difficulty: int, start: int, step: int,
                   stop_flag, result_q) -> None:
    """One worker process. Walks nonces [start, start+step, start+2*step, ...]."""
    seed = bytes.fromhex(seed_hex)
    nonce = start
    pack = struct.Struct("<Q").pack
    blake = blake3.blake3
    # Tight inner loop. Check the shared stop flag every 4096 hashes.
    inner = 4096
    while not stop_flag.value:
        for _ in range(inner):
            h = blake(seed + pack(nonce)).digest()
            if _leading_zero_bits(h) >= difficulty:
                stop_flag.value = 1
                result_q.put((nonce, h.hex()))
                return
            nonce += step
        # cooperative yield


def solve_challenge(seed_hex: str, difficulty: int,
                    workers: int | None = None,
                    progress_cb: Callable[[float, int], None] | None = None,
                    ) -> tuple[int, str]:
    """Find a nonce satisfying the BLAKE3 PoW. Returns (nonce, digest_hex).

    Layout assumption: ``BLAKE3(seed_bytes || u64_le(nonce))`` with the digest
    interpreted big-endian and the top ``difficulty`` bits required to be zero.
    """
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    ctx = mp.get_context("spawn")
    stop_flag = ctx.Value("i", 0)
    result_q: mp.Queue = ctx.Queue()

    procs = [
        ctx.Process(target=_solver_worker,
                    args=(seed_hex, difficulty, i, workers, stop_flag, result_q),
                    daemon=True)
        for i in range(workers)
    ]
    t0 = time.time()
    for p in procs:
        p.start()

    # Optional progress ticker (we can't reach into the workers, so we just
    # report wall-clock).
    last_tick = t0
    while result_q.empty():
        time.sleep(0.5)
        now = time.time()
        if progress_cb and now - last_tick >= 5.0:
            progress_cb(now - t0, workers)
            last_tick = now

    nonce, digest_hex = result_q.get()
    stop_flag.value = 1
    for p in procs:
        p.join(timeout=2.0)
        if p.is_alive():
            p.terminate()
    return nonce, digest_hex


# --------------------------------------------------------------------------- #
# Stratum client                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class MiningParams:
    """Whatever ``pearl.set_mining_params`` delivers. Stored verbatim."""
    raw: dict[str, Any]


@dataclass
class Job:
    """Whatever ``mining.notify`` delivers. Stored verbatim until we know the
    exact field order from a live capture."""
    job_id: str | None
    raw: Any  # array or object as the pool sends it


@dataclass
class StratumConfig:
    host: str = "eu1.alphapool.tech"
    port: int = 5566
    address: str = ""                # bech32 wallet, required
    worker: str = "rig01"            # appended as ADDRESS.WORKER
    password: str = "x"              # "x" or "x;d=<N>" for static diff
    user_agent: str = "alpha-miner/0.1"
    solver: str = "gpu"              # "gpu" (OpenCL) or "cpu" (multiprocessing)
    solver_workers: int | None = None  # CPU solver only
    connect_timeout_sec: float = 15.0


class StratumClient:
    """Newline-delimited JSON-RPC client for the pearl/v1 stratum dialect."""

    def __init__(self, cfg: StratumConfig,
                 on_set_difficulty: Callable[[Any], None] | None = None,
                 on_mining_params: Callable[[MiningParams], None] | None = None,
                 on_notify: Callable[[Job], None] | None = None,
                 on_challenge: Callable[[str, int], None] | None = None,
                 on_log: Callable[[str], None] | None = None,
                 ) -> None:
        self.cfg = cfg
        self._on_set_difficulty = on_set_difficulty
        self._on_mining_params = on_mining_params
        self._on_notify = on_notify
        self._on_challenge = on_challenge
        self._on_log = on_log or (lambda s: print(s))

        self._sock: socket.socket | None = None
        self._buf = b""
        self._next_id = 100
        self._pending: dict[int, threading.Event] = {}
        self._responses: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._disconnected = threading.Event()  # set when reader_loop exits (clean or error)
        self._authorized = False

    # -- transport -------------------------------------------------------- #

    def connect(self) -> None:
        s = socket.create_connection((self.cfg.host, self.cfg.port),
                                     timeout=self.cfg.connect_timeout_sec)
        s.settimeout(None)
        self._sock = s
        self._on_log(f"connected host={self.cfg.host} port={self.cfg.port}")
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
            self._sock = None

    def _send(self, msg: dict[str, Any]) -> None:
        assert self._sock is not None
        line = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        self._sock.sendall(line)

    def _next_request_id(self) -> int:
        self._next_id += 1
        return self._next_id

    # -- reader / dispatcher --------------------------------------------- #

    def _reader_loop(self) -> None:
        assert self._sock is not None
        try:
            while not self._stop.is_set():
                try:
                    chunk = self._sock.recv(4096)
                except OSError:
                    break
                if not chunk:
                    self._on_log("stratum connection closed")
                    break
                self._buf += chunk
                while b"\n" in self._buf:
                    line, self._buf = self._buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        self._on_log(f"junk from server: {line[:200]!r}")
                        continue
                    self._dispatch(msg)
        finally:
            self._disconnected.set()
            # Wake up anything waiting on a JSON-RPC response — that reply
            # will never come now.
            with self._lock:
                for ev in self._pending.values():
                    ev.set()

    def is_disconnected(self) -> bool:
        return self._disconnected.is_set()

    def wait_until_disconnected(self, timeout: float | None = None) -> bool:
        """Block until the reader thread observes a connection close (or
        ``timeout`` elapses). Returns True if disconnected."""
        return self._disconnected.wait(timeout)

    def _dispatch(self, msg: dict[str, Any]) -> None:
        method = msg.get("method")
        if method is None:
            # response to a call we made
            rid = msg.get("id")
            if rid is not None:
                with self._lock:
                    self._responses[int(rid)] = msg
                    ev = self._pending.pop(int(rid), None)
                if ev is not None:
                    ev.set()
            return

        params = msg.get("params")
        if method == "pearl.challenge":
            seed = params["seed"]
            diff = int(params["difficulty"])
            if self._on_challenge:
                self._on_challenge(seed, diff)
        elif method == "mining.set_difficulty":
            if self._on_set_difficulty:
                self._on_set_difficulty(params)
        elif method == "pearl.set_mining_params":
            if self._on_mining_params:
                self._on_mining_params(MiningParams(raw=params if isinstance(params, dict) else {"params": params}))
        elif method == "mining.notify":
            job_id = None
            if isinstance(params, list) and params:
                job_id = str(params[0])
            elif isinstance(params, dict):
                job_id = str(params.get("job_id") or params.get("id") or "")
            if self._on_notify:
                self._on_notify(Job(job_id=job_id, raw=params))
        else:
            self._on_log(f"unhandled server notification: {method} params={params!r}")

    # -- request helpers -------------------------------------------------- #

    def _call(self, method: str, params: Any, timeout: float = 15.0) -> dict[str, Any]:
        rid = self._next_request_id()
        ev = threading.Event()
        with self._lock:
            self._pending[rid] = ev
        self._send({"id": rid, "method": method, "params": params})
        if not ev.wait(timeout):
            with self._lock:
                self._pending.pop(rid, None)
            raise TimeoutError(f"timeout waiting for response to {method} id={rid}")
        with self._lock:
            resp = self._responses.pop(rid, None)
        if resp is None:
            # Reader thread woke us via disconnect; no real response arrived.
            raise ConnectionError(f"connection lost before response to {method} id={rid}")
        return resp

    # -- high level handshake -------------------------------------------- #

    def handshake(self) -> None:
        """Run mining.configure + mining.subscribe + solve challenge + authorize.

        Blocks until ``mining.authorize`` returns a result.
        """
        cfg = self.cfg

        # configure / subscribe go out first; both responses are usually
        # innocuous (often {"result":{"pearl/v1":true}} / true), but we still
        # send them because the binary does.
        self._send({"id": self._next_request_id(),
                    "method": "mining.configure",
                    "params": [["pearl/v1"], {}]})
        self._send({"id": self._next_request_id(),
                    "method": "mining.subscribe",
                    "params": [cfg.user_agent]})

        # Wait for the first pearl.challenge to land. We use a lightweight
        # spin since the dispatcher pushes the seed/difficulty via callback.
        challenge_seen = threading.Event()
        challenge_state: dict[str, Any] = {}

        original_cb = self._on_challenge
        def _capture(seed: str, diff: int) -> None:
            if not challenge_seen.is_set():
                challenge_state["seed"] = seed
                challenge_state["difficulty"] = diff
                challenge_seen.set()
            if original_cb:
                original_cb(seed, diff)
        self._on_challenge = _capture

        if not challenge_seen.wait(timeout=15.0):
            raise TimeoutError("no pearl.challenge received within 15s")

        seed = challenge_state["seed"]
        diff = challenge_state["difficulty"]
        self._on_log(f"challenge_received difficulty={diff} seed={seed}")
        t0 = time.time()

        if cfg.solver == "gpu":
            from .blake3_challenge_gpu import GpuChallengeSolver
            gpu = GpuChallengeSolver()
            self._on_log(f"  ...GPU solver: {gpu.device.name} ({gpu.device.max_compute_units} CUs)")

            def _gpu_progress(dt: float, tried: int, batch: int) -> None:
                self._on_log(f"  ...solving BLAKE3 challenge: {dt:.1f}s, "
                             f"tried={tried:,}, {tried/dt/1e6:.0f} MH/s")

            nonce, dt, tried = gpu.solve(seed, diff, progress_cb=_gpu_progress)
            # Recompute digest on CPU once for the log line — cheap.
            digest_hex = blake3.blake3(bytes.fromhex(seed) + struct.pack("<Q", nonce)).digest().hex()
        else:
            def _cpu_progress(dt: float, w: int) -> None:
                self._on_log(f"  ...solving BLAKE3 challenge: {dt:.0f}s elapsed, {w} workers")

            nonce, digest_hex = solve_challenge(
                seed, diff, workers=cfg.solver_workers, progress_cb=_cpu_progress)
            dt = time.time() - t0

        nonce_hex = struct.pack(">Q", nonce).hex()  # big-endian hex display
        self._on_log(f"challenge_solved difficulty={diff} nonce={nonce_hex} seconds={dt:.2f} digest={digest_hex}")

        # Send response. The reference miner sends it as a notification (no id)
        # per the strings -- the JSON-RPC error path is never used here, the
        # server simply drops the connection if the response is wrong.
        self._send({"method": "pearl.challenge_response",
                    "params": {"seed": seed, "nonce": nonce_hex}})

        # Now authorize. Username is "<addr>.<worker>" per the help text.
        if not cfg.address:
            raise ValueError("StratumConfig.address is required")
        user = f"{cfg.address}.{cfg.worker}" if cfg.worker else cfg.address
        auth_resp = self._call("mining.authorize", [user, cfg.password], timeout=20.0)
        if auth_resp.get("error"):
            raise RuntimeError(f"authorize rejected: {auth_resp['error']}")
        if auth_resp.get("result") is False:
            raise RuntimeError(f"authorize returned false: {auth_resp}")
        self._authorized = True
        self._on_log(f"authorized user={user}")

    # -- share submission ------------------------------------------------- #

    def submit_share(self, job_id: str, plain_proof: bytes) -> dict[str, Any]:
        """Submit a share. Wire format (confirmed by live probe 2026-05-24):

            params = [worker_username, job_id, base64(plain_proof_bytes)]

        Returns the JSON-RPC response object (with ``result``/``error`` keys).
        Note: ``result: true`` from the pool only means "params parsed";
        proof validation happens asynchronously and surfaces in pool stats,
        not in this response.
        """
        import base64
        user = f"{self.cfg.address}.{self.cfg.worker}" if self.cfg.worker else self.cfg.address
        proof_b64 = base64.b64encode(plain_proof).decode("ascii")
        return self._call("mining.submit", [user, job_id, proof_b64], timeout=30.0)


# --------------------------------------------------------------------------- #
# CLI demo                                                                    #
# --------------------------------------------------------------------------- #

def _demo() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="eu1.alphapool.tech")
    ap.add_argument("--port", type=int, default=5566)
    ap.add_argument("--address", required=True, help="bech32 pearl wallet (prl1...)")
    ap.add_argument("--worker", default="probe")
    ap.add_argument("--password", default="x")
    ap.add_argument("--solver", choices=("gpu", "cpu"), default="gpu",
                    help="BLAKE3 challenge solver backend (default: gpu)")
    ap.add_argument("--workers", type=int, default=None,
                    help="CPU solver process count (default: cpu_count-1)")
    ap.add_argument("--observe-seconds", type=float, default=20.0,
                    help="Stay connected this long after handshake to capture jobs")
    args = ap.parse_args()

    cfg = StratumConfig(
        host=args.host, port=args.port,
        address=args.address, worker=args.worker, password=args.password,
        solver=args.solver, solver_workers=args.workers,
    )

    def on_diff(p: Any) -> None:
        print(f"  << mining.set_difficulty {p}")

    def on_params(mp: MiningParams) -> None:
        print(f"  << pearl.set_mining_params {json.dumps(mp.raw)}")

    def on_notify(j: Job) -> None:
        print(f"  << mining.notify job_id={j.job_id} raw={json.dumps(j.raw)[:400]}")

    def on_challenge(seed: str, diff: int) -> None:
        # only logged the first time; reader thread fires this on every repeat
        pass

    client = StratumClient(cfg,
                           on_set_difficulty=on_diff,
                           on_mining_params=on_params,
                           on_notify=on_notify,
                           on_challenge=on_challenge,
                           on_log=lambda s: print(f"  [client] {s}"))
    client.connect()
    try:
        client.handshake()
        print(f"  [client] handshake complete; observing for {args.observe_seconds:.0f}s...")
        time.sleep(args.observe_seconds)
    finally:
        client.close()


if __name__ == "__main__":
    _demo()
