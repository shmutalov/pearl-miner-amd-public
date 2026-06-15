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
    ap.add_argument("--vulkan", action="store_true",
                    help="use the Vulkan jackpot evaluator (~2.6x; experiments/vk_jackpot)")
    ap.add_argument("--coopmat", action="store_true",
                    help="use the amortized-GEMM + tensor-core evaluator (~44x; pool pattern only)")
    ap.add_argument("--coopmat-batch", type=int, default=64,
                    help="coopmat: distinct shares to find+submit per round")
    ap.add_argument("--coopmat-submit-threads", type=int, default=4,
                    help="coopmat: parallel submit/proof-build workers (overlaps the "
                         "~1s/share pool submit; raise if host-bound, lower toward 1 "
                         "if the pool rejects concurrent submits)")
    ap.add_argument("--coopmat-submit-stagger-ms", type=float, default=25.0,
                    help="coopmat: minimum gap (ms) between submit starts across "
                         "workers, so a round's shares don't hit the pool as one "
                         "burst. 0 disables. Acts as a max submit rate of "
                         "1000/this per second; keep below the GPU share rate so it "
                         "doesn't throttle (e.g. 25ms = 40/s cap)")
    ap.add_argument("--seed-hex", default=None,
                    help="32-byte miner seed; default = random")
    ap.add_argument("--submit", action="store_true",
                    help="actually call session.submit_share; default is dry-run")
    ap.add_argument("--max-hits", type=int, default=8,
                    help="stop after submitting this many hits (real or dry)")
    ap.add_argument("--summary-interval-sec", type=float, default=300.0,
                    help="print the D-tuning summary every N seconds while mining "
                         "(each covers the window since the last); 0 = only at exit")
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
    # D-tuning telemetry. The pipeline is balanced when the host can submit shares
    # at least as fast as the GPU finds them: GPU offers R/D shares/s (R=attempts/s),
    # one consumer drains 1/(t_proof+t_submit) shares/s -> balance at D_min =
    # R*(t_proof+t_submit). Below D_min the queue fills and the GPU stalls; above it
    # you're GPU-bound (good) but variance rises with D. Measured from miner events.
    tune = {"build": [], "submit": [], "best_attempts": 0, "best_seconds": 0.0,
            "D": None, "win_start": time.monotonic()}
    tune_lock = threading.Lock()      # tune is written by N submit threads + read by timer
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

        if kind in ("proof_built", "submit_done", "hit_found"):
            with tune_lock:
                if kind == "proof_built":
                    tune["build"].append(float(info.get("build_seconds", 0.0)))
                elif kind == "submit_done":
                    tune["submit"].append(float(info.get("seconds", 0.0)))
                else:  # hit_found: best (longest-integration) sample = steadiest rate
                    s = float(info.get("seconds", 0.0))
                    if s > tune["best_seconds"]:
                        tune["best_seconds"] = s
                        tune["best_attempts"] = int(info.get("attempts", 0))

        if kind == "hit_found":
            hits_submitted[0] += 1
            if hits_submitted[0] >= args.max_hits:
                print(f"[{_ts()}] reached --max-hits={args.max_hits}, stopping")
                stop_evt.set()

    def _print_tuning() -> None:
        import statistics
        # Snapshot + reset the window atomically so each summary covers the time
        # since the last one (lets you watch submit latency / R drift live).
        with tune_lock:
            build, submit = tune["build"], tune["submit"]
            best_a, best_s = tune["best_attempts"], tune["best_seconds"]
            now = time.monotonic()
            win = now - tune["win_start"]
            D = tune["D"]
            tune["build"], tune["submit"] = [], []
            tune["best_attempts"], tune["best_seconds"] = 0, 0.0
            tune["win_start"] = now
        nb, ns = len(build), len(submit)
        if nb == 0 or best_s <= 0:
            print(f"[{_ts()}] D-tuning: no shares in the last {win:.0f}s window")
            return
        t_proof = statistics.fmean(build)
        t_submit = statistics.fmean(submit) if ns else 0.0
        host = t_proof + t_submit
        W = max(1, args.coopmat_submit_threads)         # parallel submit workers
        drain = (W / host) if host > 0 else float("inf")  # shares/s the host sustains
        R = best_a / best_s                              # effective attempts/s
        d_min = R / drain if drain > 0 else 0.0
        print(f"[{_ts()}] ---- D-tuning summary (last {win:.0f}s, {ns} shares) ----")
        print(f"    GPU rate R (effective) : {R/1e6:.1f} M attempts/s")
        print(f"    proof build  avg       : {t_proof*1e3:.2f} ms  (n={nb})")
        print(f"    pool submit  avg       : {t_submit*1e3:.2f} ms  (n={ns})"
              + ("" if args.submit else "  [DRY-RUN ~0; rerun --submit for real submit cost]"))
        print(f"    submit workers         : {W}  (parallel; --coopmat-submit-threads)")
        print(f"    host cost / share      : {host*1e3:.2f} ms  -> host drains ~{drain:.1f} shares/s ({W}x parallel)")
        if D:
            print(f"    current D              : {D:g}  -> GPU offers ~{R/D:.0f} shares/s at this D")
        print(f"    balance point D_min    : ~{d_min:,.0f}  (host keeps up when D >= this)")
        print(f"    suggested D            : ~{d_min*2:,.0f}-{d_min*4:,.0f} (2-4x margin); lower D only")
        print(f"                             smooths payout variance, it does not raise earnings")
        print(f"    (R is throttled while host-bound; re-measure as you raise D / workers)")

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

    # Periodic D-tuning summary (each covers the window since the last print).
    def _summary_timer():
        while not stop_evt.wait(args.summary_interval_sec):
            _print_tuning()
    if args.summary_interval_sec > 0:
        threading.Thread(target=_summary_timer, daemon=True).start()

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
        tune["D"] = initial.share_difficulty

        miner = PearlMiner(
            sess, miner_seed=miner_seed,
            max_attempts_per_job=args.max_attempts_per_job,
            on_event=_on_event,
            jackpot_batch_size=args.jackpot_batch_size,
            use_vulkan_jackpot=args.vulkan,
            use_coopmat_jackpot=args.coopmat,
            coopmat_shares_per_round=args.coopmat_batch,
            coopmat_submit_threads=args.coopmat_submit_threads,
            coopmat_submit_stagger=args.coopmat_submit_stagger_ms / 1000.0,
        )
        print(f"[{_ts()}] PearlMiner ready: derive_gpu={miner._derive_gpu is not None}, "
              f"merkle_gpu={miner._merkle_gpu is not None}, "
              f"jackpot_gpu={miner._use_gpu_jackpot}, "
              f"jackpot_vulkan={miner._use_vulkan_jackpot}, "
              f"jackpot_coopmat={miner._use_coopmat_jackpot}")

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
                tune["D"] = work.share_difficulty

                if args.coopmat:
                    # Continuous coopmat mining: round N+1's preflight (derive +
                    # merkle + set_job into the idle double-buffered job slot) runs
                    # on a prefetch thread while round N searches the active slot,
                    # so the per-round rebuild overlaps the search. Each round
                    # refreshes the seed, so the same pool job keeps yielding fresh
                    # distinct shares. The loop exits (and the outer loop re-enters
                    # with the new work) when stop_evt fires or the pool ships a new
                    # job; the round itself is interruptible within ~one tile.
                    job_key = work.job_key
                    def _should_stop() -> bool:
                        if stop_evt.is_set() or sess.is_disconnected():
                            return True
                        cw = sess.current_work()
                        return cw is not None and cw.job_key != job_key
                    miner.mine_coopmat_continuous(
                        work, should_stop=_should_stop,
                        next_seed=lambda: os.urandom(32))
                    continue

                # Non-coopmat: one preflight + one search per job, so we can break
                # out cleanly when stop_evt fires. Passing the stop predicate makes
                # the round interruptible (~one tile) instead of running to
                # completion before Ctrl+C is seen.
                state = miner._preflight(work)
                miner._search_one_job(work, state,
                                      should_continue=lambda: not stop_evt.is_set())
        finally:
            _print_tuning()
            print(f"[{_ts()}] done. hits_submitted={hits_submitted[0]}, "
                  f"session_stats={sess.stats()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
