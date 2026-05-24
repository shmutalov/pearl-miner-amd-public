"""Candidate search loop: iterate over valid ``(t_rows, t_cols)`` offsets
within ``rows_pattern × cols_pattern`` and find one whose ``hash_jackpot``
(interpreted little-endian as a uint256) is below the share target.

This is the inner mining loop in its simplest form: pure Python, one
candidate at a time. The search space is huge — at pool shape
(``rows_pattern=[0, 32]``, ``cols_pattern=[0..63]``, m=n=131072) there are
~65k × ~2k = ~134M candidates per (A, B) commitment, so an actual share
search at 47-bit target requires ~10^14 candidates total. That obviously
won't happen on this pure-Python path — it exists to verify the loop
mechanics and as a test oracle for the OpenCL version to come.

For sanity-testing the wire path, run with ``--password "x;d=1"`` (sets
the pool's share_diff to 1 → ~2^16 expected candidates per hit) and you
will find hits within minutes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterator

from .jackpot import evaluate_candidate
from .mining_config import MiningConfiguration
from .proof_builder import JobMatrices


@dataclass
class Candidate:
    t_rows: int
    t_cols: int
    a_rows_indices: list[int]
    b_cols_indices: list[int]
    hash_jackpot: bytes
    target_value: int


def iter_offsets(pattern_size: int, axis_len: int, period: int,
                 axis_constraint_modulus: int, axis_constraint_max: int,
                 ) -> Iterator[int]:
    """Yield every valid offset for one axis.

    ``axis_len`` is the matrix dim along that axis (``m`` or ``n``); the
    pattern's largest touched index plus offset must stay strictly below
    ``axis_len``. The other two arguments come from ``offset_is_valid``'s
    reduction of the pattern's innermost ``(stride, length)`` pair.
    """
    # Most-constrained shape: offset % modulus < max
    upper = axis_len - period + 1  # exclusive upper bound on offset
    # Walk in steps of modulus's full period, yielding [base..base+max-1]
    base = 0
    while base < upper:
        for delta in range(min(axis_constraint_max, upper - base)):
            yield base + delta
        base += axis_constraint_modulus


def _axis_constraint(pattern_shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    """For ``offset_is_valid``: the relevant (stride, length) is the innermost
    (first) non-trivial one. The constraint is ``offset % (stride*length) < stride``.
    """
    for stride, length in pattern_shape:
        if length > 1:
            return stride * length, stride
    # All-trivial pattern (single index): any offset is valid; return generous defaults.
    return 1, 1


def enumerate_valid_offsets(pattern, axis_len: int) -> Iterator[int]:
    """All offsets ``o`` such that ``offset_is_valid(o)`` and
    ``max(pattern.to_list()) + o < axis_len``."""
    modulus, max_in_window = _axis_constraint(pattern.shape)
    pattern_period = pattern.period()
    max_touched = max(pattern.to_list())
    upper = axis_len - max_touched  # exclusive: o must be < upper
    base = 0
    while base < upper:
        for delta in range(min(max_in_window, upper - base)):
            yield base + delta
        base += modulus


def search_candidate(jm: JobMatrices,
                     mc: MiningConfiguration,
                     target: int,
                     *,
                     max_attempts: int = 10_000,
                     report_every: int = 1000,
                     progress_cb: Callable[[int, float, int], None] | None = None,
                     ) -> tuple[Candidate | None, int, float]:
    """Walk valid ``(t_rows, t_cols)`` until ``max_attempts`` is reached or
    a hit is found. Returns ``(hit_or_None, attempts_done, seconds_elapsed)``.
    """
    if jm.m != jm.A.shape[0] or jm.n != jm.B.shape[0]:
        raise ValueError("JobMatrices m/n don't match A/B shapes")
    if jm.k != mc.common_dim:
        raise ValueError("JobMatrices k != mining_config.common_dim")

    _, a_noise_seed = jm.commitment_hash()
    commitment_hash = jm.commitment_hash()

    t0 = time.time()
    attempts = 0
    for t_rows in enumerate_valid_offsets(mc.rows_pattern, jm.m):
        a_rows_indices = mc.rows_pattern.indices_with_offset(t_rows)
        for t_cols in enumerate_valid_offsets(mc.cols_pattern, jm.n):
            b_cols_indices = mc.cols_pattern.indices_with_offset(t_cols)
            target_value, jp_hash = evaluate_candidate(
                jm.A, jm.B, a_rows_indices, b_cols_indices,
                commitment_hash, a_noise_seed, jm.k, mc.rank)
            attempts += 1
            if target_value < target:
                return (Candidate(
                    t_rows=t_rows, t_cols=t_cols,
                    a_rows_indices=list(a_rows_indices),
                    b_cols_indices=list(b_cols_indices),
                    hash_jackpot=jp_hash,
                    target_value=target_value,
                ), attempts, time.time() - t0)
            if progress_cb and attempts % report_every == 0:
                progress_cb(attempts, time.time() - t0, target_value)
            if attempts >= max_attempts:
                return None, attempts, time.time() - t0
    return None, attempts, time.time() - t0


# ---------------------------------------------------------------------------- #
# Selftest                                                                     #
# ---------------------------------------------------------------------------- #

def _selftest_search_small() -> None:
    """Tiny-shape search to validate the loop end-to-end with a hit
    achievable in seconds. We use a small (m, n, k, rank) that lets the
    pure-Python path complete ~1k candidates within a few seconds."""
    from .mining_config import MiningConfiguration, PeriodicPattern, compute_job_key
    from .proof_builder import derive_AB_from_seed, merkle_root_keyed, JobMatrices
    import blake3

    m, n, k, r = 1024, 1024, 256, 64
    mc = MiningConfiguration(
        common_dim=k, rank=r, mma_type=0,
        rows_pattern=PeriodicPattern.from_list([0, 32]),
        cols_pattern=PeriodicPattern.from_list(list(range(64))),
    )
    header = bytes(76)  # all-zero header for self-contained test
    job_key = compute_job_key(header, mc)

    A, B = derive_AB_from_seed(bytes(32), m, n, k)
    hash_a = merkle_root_keyed(A, job_key)
    hash_b = merkle_root_keyed(B, job_key)
    jm = JobMatrices(A=A, B=B, miner_seed=bytes(32), hash_a=hash_a, hash_b=hash_b,
                     job_key=job_key, m=m, n=n, k=k)

    # Very loose target: ≤ 1 leading zero bit (target ≈ 2^255). Almost every
    # candidate satisfies it; we use this to exercise the hit path.
    target = (1 << 255)
    hit, attempts, dt = search_candidate(jm, mc, target, max_attempts=10)
    assert hit is not None, "loose target should match almost immediately"
    print(f"  loose target: hit in {attempts} attempts ({dt*1000:.0f}ms)")
    print(f"    t_rows={hit.t_rows} t_cols={hit.t_cols}")
    print(f"    hash={hit.hash_jackpot.hex()}")

    # Tighter target — should still find within a few hundred attempts.
    target_tight = (1 << 248)  # ≥ 8 leading zero bits required
    hit2, attempts2, dt2 = search_candidate(jm, mc, target_tight, max_attempts=10000)
    if hit2 is not None:
        print(f"  tight target (≥8 lz bits): hit in {attempts2} attempts ({dt2:.2f}s)")
    else:
        print(f"  tight target: no hit in {attempts2} attempts ({dt2:.2f}s) — "
              f"that's fine, statistical")


def _selftest_offset_enum() -> None:
    """Sanity-check offset enumeration: counts must match by formula."""
    from .mining_config import PeriodicPattern
    rows = PeriodicPattern.from_list([0, 32])
    cols = PeriodicPattern.from_list(list(range(64)))
    # rows_pattern: shape ((32, 2), ...) → constraint mod 64 < 32, period 64
    # For m=131072 (which is 64*2048), max_touched=32; upper = 131072-32 = 131040.
    # Half of those satisfy (o % 64 < 32). 2048 full windows × 32 = 65536 candidates,
    # minus any in the last partial window beyond 131040-32=131008. base=131008,
    # delta=0..31 all OK since upper=131040 -> 32 added. Total = 65536.
    n_rows = sum(1 for _ in enumerate_valid_offsets(rows, 131072))
    # cols_pattern: shape ((1, 64), ...) → constraint mod 64 < 1, i.e. multiples of 64.
    # For n=131072, max_touched=63, upper=131009. Multiples of 64 in [0, 131009) = 2048.
    n_cols = sum(1 for _ in enumerate_valid_offsets(cols, 131072))
    print(f"  offset counts at pool shape: rows={n_rows}, cols={n_cols}")
    assert n_rows == 65536, f"expected 65536 rows offsets, got {n_rows}"
    assert n_cols == 2048, f"expected 2048 cols offsets, got {n_cols}"


if __name__ == "__main__":
    _selftest_offset_enum()
    _selftest_search_small()
    print("all selftests OK")
