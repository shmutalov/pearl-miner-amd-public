"""Live miner smoke test against a real Pearl stratum pool.

End-to-end exercise of the full pipeline:

  StratumSession  ─►  Work
        ▼
  PearlMiner._preflight   (GPU derive + GPU merkle + GPU set_job)
        ▼
  PearlMiner._search_one_job   (GPU JackpotGpu.search → Candidate)
        ▼
  build_plain_proof + encode + session.submit_share

Defaults are intentionally conservative for a one-off run:
  - 60 s wall-clock cap (pool's anti-leech drops idle workers at ~60 s
    anyway, so this matches the natural session lifetime)
  - 200 000 attempts per job (~7 s of GPU search at pool throughput;
    enough to see the pipeline run end-to-end but not so much that a
    misbehaving build burns hours of GPU)
  - --dry-run by default: assemble + log every PlainProof but do NOT
    call submit_share; pass --submit to actually post shares

All PearlMiner events are streamed to stdout with timestamps; whatever
the pool responds is logged verbatim.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from pearl_amd.miner import PearlMiner, build_plain_proof  # noqa: E402
from pearl_amd.stratum_client import StratumConfig  # noqa: E402
from pearl_amd.stratum_session import StratumSession, Work  # noqa: E402


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="eu1.alphapool.tech")
    ap.add_argument("--port", type=int, default=5566)
    ap.add_argument("--address", required=True,
                    help="Pearl wallet (prl1…)")
    ap.add_argument("--worker", default="rx570")
    ap.add_argument("--password", default="x",
                    help="stratum password; 'x;d=1' requests min difficulty")
    ap.add_argument("--observe-seconds", type=float, default=60.0)
    ap.add_argument("--max-attempts-per-job", type=int, default=200_000)
    ap.add_argument("--jackpot-batch-size", type=int, default=8192)
    ap.add_argument("--seed-hex", default=None,
                    help="32-byte miner seed; default = random")
    ap.add_argument("--submit", action="store_true",
                    help="actually call session.submit_share; default is dry-run")
    ap.add_argument("--max-hits", type=int, default=8,
                    help="stop after submitting this many hits (real or dry)")
    args = ap.parse_args()

    if args.seed_hex:
        miner_seed = bytes.fromhex(args.seed_hex)
        if len(miner_seed) != 32:
            ap.error("--seed-hex must decode to 32 bytes")
    else:
        miner_seed = os.urandom(32)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    cfg = StratumConfig(
        host=args.host, port=args.port,
        address=args.address, worker=args.worker, password=args.password,
    )

    stop_evt = threading.Event()
    hits_submitted = [0]
    print(f"[{_ts()}] live miner starting — pool={args.host}:{args.port}, "
          f"worker={args.worker}, submit={'ON' if args.submit else 'DRY-RUN'}")
    print(f"[{_ts()}] miner_seed = {miner_seed.hex()}")

    def _on_event(kind: str, info: dict) -> None:
        # Pretty-print events; truncate long values.
        parts = []
        for key, val in info.items():
            if isinstance(val, float):
                parts.append(f"{key}={val:.3f}")
            elif isinstance(val, (bytes, bytearray)):
                parts.append(f"{key}={val.hex()[:32]}...")
            elif isinstance(val, str) and len(val) > 40:
                parts.append(f"{key}={val[:40]}...")
            else:
                parts.append(f"{key}={val}")
        print(f"[{_ts()}] {kind}: {', '.join(parts)}")

        if kind == "hit_found":
            hits_submitted[0] += 1
            if hits_submitted[0] >= args.max_hits:
                print(f"[{_ts()}] reached --max-hits={args.max_hits}, stopping")
                stop_evt.set()

    # Signal handlers.
    def _stop_handler(sig, frame):
        print(f"[{_ts()}] signal {sig}; stopping...")
        stop_evt.set()
    signal.signal(signal.SIGINT, _stop_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop_handler)

    # Wall-clock cap.
    def _cap_timer():
        if stop_evt.wait(args.observe_seconds):
            return
        print(f"[{_ts()}] observe-seconds={args.observe_seconds:.0f} reached, "
              f"stopping")
        stop_evt.set()
    threading.Thread(target=_cap_timer, daemon=True).start()

    with StratumSession(cfg, on_log=lambda s: print(f"[{_ts()}] stratum: {s}")) as sess:
        print(f"[{_ts()}] session ready, waiting for first job...")
        try:
            initial = sess.wait_for_work(timeout=30.0)
        except (TimeoutError, ConnectionError) as e:
            print(f"[{_ts()}] failed to receive initial work: {e!r}")
            return 1
        print(f"[{_ts()}] initial work: job_id={initial.job_id}, m={initial.m} "
              f"n={initial.n} k={initial.mining_config.common_dim} "
              f"rank={initial.mining_config.rank}, "
              f"target_lz={256 - initial.target.bit_length() if initial.target > 0 else 256}, "
              f"difficulty={initial.share_difficulty}")

        miner = PearlMiner(
            sess, miner_seed=miner_seed,
            max_attempts_per_job=args.max_attempts_per_job,
            on_event=_on_event,
            jackpot_batch_size=args.jackpot_batch_size,
        )
        print(f"[{_ts()}] PearlMiner ready: derive_gpu={miner._derive_gpu is not None}, "
              f"merkle_gpu={miner._merkle_gpu is not None}, "
              f"jackpot_gpu={miner._use_gpu_jackpot}")

        # Patch submit_share for dry-run mode: log the proof bytes' length
        # and the first hash bytes; don't actually send to the pool.
        if not args.submit:
            real_submit = sess.submit_share
            def fake_submit(job_id, proof_bytes):
                print(f"[{_ts()}] [DRY-RUN] would submit job_id={job_id}, "
                      f"proof_bytes={len(proof_bytes)} B")
                return {"result": True, "dry_run": True}
            sess.submit_share = fake_submit  # type: ignore[assignment]

        # Run loop until stop_evt or session disconnects.
        try:
            while not stop_evt.is_set() and not sess.is_disconnected():
                try:
                    work = sess.wait_for_work(timeout=5.0)
                except TimeoutError:
                    continue
                except ConnectionError as e:
                    print(f"[{_ts()}] session lost: {e!r}")
                    break

                if stop_evt.is_set():
                    break

                # Manually drive _preflight + _search_one_job per job, so
                # we can break out cleanly when stop_evt fires.
                state = miner._preflight(work)
                miner._search_one_job(work, state)
        finally:
            print(f"[{_ts()}] done. hits_submitted={hits_submitted[0]}, "
                  f"session_stats={sess.stats()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
