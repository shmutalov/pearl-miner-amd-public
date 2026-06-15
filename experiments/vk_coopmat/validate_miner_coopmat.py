"""Miner-level integration test for the coopmat path: drive
PearlMiner._preflight + _search_one_job with use_coopmat_jackpot=True at small
shape, with a loose target so a hit fires, and confirm the whole chain
(coopmat set_job -> tensor-core search -> hit_found -> proof built -> submit)
runs and produces a proof whose candidate hash matches the CPU oracle.
"""
from __future__ import annotations
import os, sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from pearl_amd.mining_config import MiningConfiguration, PeriodicPattern
from pearl_amd.miner import PearlMiner
from pearl_amd.jackpot import evaluate_candidate
from pearl_amd.plain_proof_codec import PlainProof
from pearl_amd.proof_builder import JobMatrices


class FakeSession:
    def __init__(self):
        self.submitted = []
    def submit_share(self, job_id, proof_bytes):
        self.submitted.append((job_id, proof_bytes))
        return {"result": True, "fake": True}


def main() -> int:
    m, n, k, r = 256, 256, 256, 64
    mc = MiningConfiguration(common_dim=k, rank=r, mma_type=0,
        rows_pattern=PeriodicPattern.from_list([0, 32]),
        cols_pattern=PeriodicPattern.from_list(list(range(64))))
    header = bytes(range(76))
    work = SimpleNamespace(
        m=m, n=n, mining_config=mc,
        incomplete_header_bytes=header,
        target=(1 << 255),          # ~1 lz bit -> hit almost immediately
        job_id="test-job-1",
    )

    events = []
    sess = FakeSession()
    miner = PearlMiner(sess, miner_seed=bytes(32),
                       max_attempts_per_job=1_000_000,
                       on_event=lambda kind, info: events.append((kind, info)),
                       use_coopmat_jackpot=True)

    state = miner._preflight(work)
    used_coopmat = miner._jackpot_coopmat is not None
    print(f"coopmat active after preflight: {used_coopmat}")
    if not used_coopmat:
        ev = [e for e in events if "coopmat" in e[0]]
        print(f"  coopmat did not activate; events: {ev}")
        return 1

    miner._search_one_job(work, state)

    kinds = [k_ for (k_, _) in events]
    print("events:", kinds)
    hit_ev = next((info for (k_, info) in events if k_ == "hit_found"), None)
    submit_ev = next((info for (k_, info) in events if k_ == "submit_done"), None)
    if hit_ev is None:
        print("NO hit_found event"); return 1
    if submit_ev is None or not sess.submitted:
        print("submit_share was not called"); return 1

    # Cross-check the hit hash against the CPU oracle.
    t_r, t_c = hit_ev["t_rows"], hit_ev["t_cols"]
    jm: JobMatrices = state.job_matrices
    ch = jm.commitment_hash(); _, a_noise_seed = ch
    rows = list(mc.rows_pattern.indices_with_offset(t_r))
    cols = list(mc.cols_pattern.indices_with_offset(t_c))
    _, ev_hash = evaluate_candidate(jm.A, jm.B, rows, cols, ch, a_noise_seed, k, r)
    ok_hash = (ev_hash.hex() == hit_ev["hash"])
    print(f"hit t_r={t_r} t_c={t_c}  hash_matches_oracle={ok_hash}")

    # Proof must decode + carry this candidate.
    job_id, proof_bytes = sess.submitted[0]
    dec = PlainProof.decode(proof_bytes)
    ok_proof = (dec.encode() == proof_bytes and dec.a.row_indices == rows
                and dec.bt.row_indices == cols)
    print(f"proof: {len(proof_bytes)} bytes, decode+roundtrip+indices ok={ok_proof}")

    miner._jackpot_coopmat.close()
    if ok_hash and ok_proof:
        print("MINER COOPMAT INTEGRATION CORRECT")
        return 0
    print("MINER COOPMAT INTEGRATION **FAILED**")
    return 1


if __name__ == "__main__":
    sys.exit(main())
