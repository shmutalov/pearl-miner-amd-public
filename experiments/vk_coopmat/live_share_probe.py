"""Decisive live probe: how good a hash can our miner produce, and does the pool
accept it?

For each scan we derive a fresh (A,B) (new miner seed -> fresh 134M candidate
space over the same pool job), find the single BEST (lowest LE-uint256) jackpot
hash via the coopmat kernel, and keep the global best. Then we SUBMIT that best
proof and print the pool's verdict verbatim.

  - If accepted -> the real share threshold is <= our best_lz (feasible!).
  - If rejected -> the pool's message names the required difficulty/threshold,
    which (with H_per_attempt=2^32) tells us exactly where the bar is.

Dry by default; pass --submit to actually post the best proof.
"""
from __future__ import annotations
import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from pearl_amd.miner import PearlMiner, build_plain_proof
from pearl_amd.stratum_client import StratumConfig
from pearl_amd.stratum_session import StratumSession


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="eu1.alphapool.tech")
    ap.add_argument("--port", type=int, default=5566)
    ap.add_argument("--address", required=True)
    ap.add_argument("--worker", default="rx7900xt")
    ap.add_argument("--password", default="x")  # default -> VarDiff
    ap.add_argument("--scans", type=int, default=3, help="fresh (A,B) spaces to scan")
    ap.add_argument("--loose-lz", type=int, default=24, help="record candidates above this lz")
    ap.add_argument("--submit", action="store_true", help="actually submit the best proof")
    args = ap.parse_args()

    def log(s): print(f"[{time.strftime('%H:%M:%S')}] {s}")

    cfg = StratumConfig(host=args.host, port=args.port, address=args.address,
                        worker=args.worker, password=args.password)
    with StratumSession(cfg, on_log=lambda s: log(f"stratum: {s}")) as sess:
        work = sess.wait_for_work(timeout=30.0)
        required_lz = 256 - work.target.bit_length() if work.target > 0 else 256
        log(f"job {work.job_id}: m={work.m} n={work.n} k={work.mining_config.common_dim} "
            f"rank={work.mining_config.rank}  REQUIRED target_lz={required_lz} "
            f"(difficulty={work.share_difficulty})")

        miner = PearlMiner(sess, miner_seed=os.urandom(32), use_coopmat_jackpot=True,
                           on_event=lambda k, i: None)

        # work.target is now the H_per_attempt-scaled acceptance target (lz~16).
        accepted = rejected = 0
        for it in range(args.scans):
            if it > 0:
                miner.miner_seed = os.urandom(32)
                miner._A = None  # fresh derive -> distinct (A,B) -> distinct share
            state = miner._preflight(work)
            jc = miner._jackpot_coopmat
            if jc is None:
                log("coopmat did not activate (pattern/DLL); aborting"); return 1
            # First hit below the (scaled) share target — this is the production path.
            cand, attempts, dt = jc.search(work.mining_config, work.target)
            if cand is None:
                log(f"scan {it+1}/{args.scans}: no hit in {attempts:,} candidates "
                    f"(target_lz={required_lz}?)"); continue
            cand_lz = 256 - cand.target_value.bit_length()
            log(f"scan {it+1}/{args.scans}: hit target_lz={cand_lz} "
                f"after {attempts:,} candidates ({attempts/dt/1e6:.0f}M/s)")
            if not args.submit:
                continue
            proof = build_plain_proof(state.job_matrices, work.mining_config,
                                      state.A_layers, state.B_layers, cand)
            try:
                resp = sess.submit_share(work.job_id, proof.encode())
                ok = isinstance(resp, dict) and resp.get("result") is True
                accepted += ok; rejected += (not ok)
                log(f"  submit -> {'ACCEPTED' if ok else 'REJECTED'}  {resp}")
            except Exception as e:
                log(f"  submit raised: {e!r}"); rejected += 1
        if args.submit:
            log(f"=== shares: {accepted} accepted, {rejected} rejected "
                f"(target_lz required={required_lz}, scaled-search) ===")
        return 0


if __name__ == "__main__":
    sys.exit(main())
