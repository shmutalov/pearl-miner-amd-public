"""Continuous earning demo: one job yields thousands of distinct shares (expected
~D=share_difficulty candidates per share). Derive (A,B) once, search_all for many
distinct hits below the (scaled) share target, submit them, tally acceptances.

Dry by default; --submit to actually post.
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
    ap.add_argument("--password", default="x")           # VarDiff
    ap.add_argument("--max-submit", type=int, default=30)
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    def log(s): print(f"[{time.strftime('%H:%M:%S')}] {s}")
    cfg = StratumConfig(host=args.host, port=args.port, address=args.address,
                        worker=args.worker, password=args.password)
    with StratumSession(cfg, on_log=lambda s: log(f"stratum: {s}")) as sess:
        work = sess.wait_for_work(timeout=30.0)
        req_lz = 256 - work.target.bit_length() if work.target > 0 else 256
        log(f"job {work.job_id}: rank={work.mining_config.rank} D={work.share_difficulty} "
            f"acceptance target_lz={req_lz}  (expected ~{int(work.share_difficulty):,} "
            f"candidates/share)")

        miner = PearlMiner(sess, miner_seed=os.urandom(32), use_coopmat_jackpot=True,
                           on_event=lambda k, i: None)
        t_pf = time.time()
        state = miner._preflight(work)
        jc = miner._jackpot_coopmat
        if jc is None:
            log("coopmat did not activate"); return 1
        log(f"preflight (derive+merkle+PA/PB build): {time.time()-t_pf:.2f}s")

        hits, attempts, dt = jc.search_all(work.mining_config, work.target,
                                           max_return=args.max_submit)
        log(f"search_all: {len(hits)} distinct shares from {attempts:,} candidates "
            f"({attempts/dt/1e6:.0f}M cand/s); GPU produced 1 share per "
            f"~{attempts//max(len(hits),1):,} candidates")
        if not args.submit:
            log(f"[DRY] {len(hits)} valid shares ready (lz "
                f"{min(256-h.target_value.bit_length() for h in hits)}.."
                f"{max(256-h.target_value.bit_length() for h in hits)}); --submit to post")
            return 0

        acc = rej = 0
        t0 = time.time()
        for h in hits:
            proof = build_plain_proof(state.job_matrices, work.mining_config,
                                      state.A_layers, state.B_layers, h)
            try:
                resp = sess.submit_share(work.job_id, proof.encode())
                ok = isinstance(resp, dict) and resp.get("result") is True
                acc += ok; rej += (not ok)
            except Exception as e:
                rej += 1; log(f"submit raised: {e!r}")
        wall = time.time() - t0
        log(f"=== submitted {len(hits)} shares: {acc} ACCEPTED, {rej} rejected in "
            f"{wall:.1f}s ({acc/wall:.1f} accepted shares/s, network-bound) ===")
        return 0


if __name__ == "__main__":
    sys.exit(main())
