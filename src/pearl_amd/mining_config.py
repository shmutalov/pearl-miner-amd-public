"""Pearl ``MiningConfiguration`` + ``IncompleteBlockHeader`` Python types,
canonical byte serialization, and ``job_key`` derivation.

Mirrors ``reference/zk-pow/src/api/proof_utils.rs``. Pure Python; no Rust
toolchain needed. The byte layouts here are what hashes into ``job_key`` —
get them wrong and the verifier rejects the share.

Wire-protocol bridge from the pool's ``pearl.set_mining_params`` payload to
this internal representation is in :func:`mining_config_from_pool_params`.
The ``share_nbits`` target decoder is :func:`nbits_to_target` (Bitcoin-style
compact representation, byte-identical to ``nbits_to_difficulty`` in Rust).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Iterable

import blake3


# ---------------------------------------------------------------------------- #
# Constants                                                                    #
# ---------------------------------------------------------------------------- #

HEADER_SERIALIZED_SIZE = 76        # IncompleteBlockHeader::SERIALIZED_SIZE
MINING_CONFIG_SERIALIZED_SIZE = 52  # MiningConfiguration::SERIALIZED_SIZE
MINING_CONFIG_RESERVED_SIZE = 32
PERIODIC_PATTERN_NUM_DIMS = 3
PERIODIC_PATTERN_BYTES = 2 * PERIODIC_PATTERN_NUM_DIMS  # 6

MMA_TYPE_INT7XINT7_TO_INT32 = 0  # only valid value in protocol v1


# ---------------------------------------------------------------------------- #
# Target (Bitcoin-style compact nbits)                                         #
# ---------------------------------------------------------------------------- #

def nbits_to_target(nbits: int) -> int:
    """Convert Bitcoin's compact 32-bit nbits to a 256-bit target integer.

    Returns an int in [0, 2^256). Matches ``nbits_to_difficulty`` in
    ``zk-pow/src/api/proof_utils.rs`` (verified by the six test vectors in
    its test module — see ``selftest_nbits`` below).
    """
    if nbits < 0 or nbits > 0xFFFFFFFF:
        raise ValueError(f"nbits out of u32 range: {nbits:#x}")
    exponent = (nbits >> 24) & 0xFF
    mantissa = nbits & 0x00FFFFFF
    if mantissa == 0 or exponent == 0:
        return 0
    if mantissa & 0x00800000:
        return 0  # negative-bit set -> invalid
    if exponent <= 3:
        return mantissa >> (8 * (3 - exponent))
    return mantissa << (8 * (exponent - 3))


def nbits_hex_to_target(nbits_hex: str) -> int:
    """Pool sends ``share_nbits`` as an 8-char hex string (u32 big-endian
    display, e.g. ``"1b014f8a"``). Convert to a 256-bit target int."""
    return nbits_to_target(int(nbits_hex, 16))


# ---------------------------------------------------------------------------- #
# PeriodicPattern                                                              #
# ---------------------------------------------------------------------------- #

@dataclass
class PeriodicPattern:
    """3-dimensional arithmetic progression of indices.

    Internally stored as exactly 3 ``(stride, length)`` pairs. Wire form is
    6 bytes: 3 × ``(factor-1: u8, length-1: u8)`` where ``factor`` is the
    multiplier relative to the running min_stride. See
    ``zk-pow/src/api/proof_utils.rs::PeriodicPattern``.
    """

    shape: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

    @classmethod
    def from_list(cls, pattern: list[int]) -> "PeriodicPattern":
        """Factor a sorted index list into a 3-dim arithmetic progression.

        Mirrors ``PeriodicPattern::from_list`` in the Rust reference. The
        list must be strictly increasing, start at 0, and be expressible as
        a (possibly nested) periodic structure.
        """
        if not pattern:
            raise ValueError("pattern cannot be empty")
        for a, b in zip(pattern, pattern[1:]):
            if a >= b:
                raise ValueError("pattern must be strictly increasing")
        if pattern[0] != 0:
            raise ValueError("pattern must start at 0")

        p = list(pattern)
        shape_vec: list[tuple[int, int]] = []
        while len(p) > 1:
            found = False
            for period in range(1, len(p)):
                if len(p) % period != 0:
                    continue
                s = p[period]
                if all(p[i] + s == p[i + period] for i in range(len(p) - period)):
                    shape_vec.append((s, len(p) // period))
                    p = p[:period]
                    found = True
                    break
            if not found:
                raise ValueError(f"pattern is not periodic: {pattern}")

        # Reverse and pad to 3 dimensions with (period, 1) entries.
        shape_vec.reverse()
        period = shape_vec[-1][0] * shape_vec[-1][1] if shape_vec else 1
        while len(shape_vec) < PERIODIC_PATTERN_NUM_DIMS:
            shape_vec.append((period, 1))
        return cls(shape=tuple(shape_vec))  # type: ignore[arg-type]

    def to_list(self) -> list[int]:
        """Expand back into the explicit index list."""
        result = [0]
        for stride, length in self.shape:
            new = []
            for i in range(length):
                for r in result:
                    new.append(r + i * stride)
            result = new
        return result

    def to_bytes(self) -> bytes:
        out = bytearray(PERIODIC_PATTERN_BYTES)
        min_stride = 1
        for i, (stride, length) in enumerate(self.shape):
            factor = stride // min_stride
            if factor == 0:
                raise ValueError(f"invalid shape (stride < min_stride): {self.shape}")
            out[2 * i] = (factor - 1) & 0xFF
            out[2 * i + 1] = (length - 1) & 0xFF
            min_stride = stride * length
        return bytes(out)

    @classmethod
    def from_bytes(cls, data: bytes) -> "PeriodicPattern":
        if len(data) != PERIODIC_PATTERN_BYTES:
            raise ValueError(f"expected {PERIODIC_PATTERN_BYTES} bytes, got {len(data)}")
        shape: list[tuple[int, int]] = []
        min_stride = 1
        is_done = False
        for i in range(PERIODIC_PATTERN_NUM_DIMS):
            factor = 1 + data[2 * i]
            length = 1 + data[2 * i + 1]
            if length == 1 or is_done:
                if factor != 1 or length != 1:
                    raise ValueError("non-canonical PeriodicPattern bytes")
                is_done = True
            stride = factor * min_stride
            shape.append((stride, length))
            min_stride = stride * length
        return cls(shape=tuple(shape))  # type: ignore[arg-type]

    def size(self) -> int:
        return self.shape[0][1] * self.shape[1][1] * self.shape[2][1]

    def period(self) -> int:
        stride, length = self.shape[-1]
        return stride * length

    def indices_with_offset(self, offset: int) -> list[int]:
        return [i + offset for i in self.to_list()]


# ---------------------------------------------------------------------------- #
# MiningConfiguration                                                          #
# ---------------------------------------------------------------------------- #

@dataclass
class MiningConfiguration:
    common_dim: int                        # u32 — = k from pool's set_mining_params
    rank: int                              # u16 — = rank
    mma_type: int                          # u16 — 0 = Int7xInt7ToInt32
    rows_pattern: PeriodicPattern
    cols_pattern: PeriodicPattern
    reserved: bytes = field(default_factory=lambda: bytes(MINING_CONFIG_RESERVED_SIZE))

    def to_bytes(self) -> bytes:
        if self.common_dim < 0 or self.common_dim > 0xFFFFFFFF:
            raise ValueError("common_dim out of u32 range")
        if self.rank < 0 or self.rank > 0xFFFF:
            raise ValueError("rank out of u16 range")
        if self.mma_type not in (MMA_TYPE_INT7XINT7_TO_INT32,):
            raise ValueError(f"unsupported mma_type {self.mma_type}")
        if len(self.reserved) != MINING_CONFIG_RESERVED_SIZE:
            raise ValueError("reserved must be exactly 32 bytes")
        if self.reserved != bytes(MINING_CONFIG_RESERVED_SIZE):
            raise ValueError("reserved must be all zeros per current protocol")
        out = bytearray()
        out += struct.pack("<I", self.common_dim)
        out += struct.pack("<H", self.rank)
        out += struct.pack("<H", self.mma_type)
        out += self.rows_pattern.to_bytes()
        out += self.cols_pattern.to_bytes()
        out += self.reserved
        if len(out) != MINING_CONFIG_SERIALIZED_SIZE:
            raise AssertionError(f"serialized size {len(out)} != {MINING_CONFIG_SERIALIZED_SIZE}")
        return bytes(out)

    def dot_product_length(self) -> int:
        """Protocol dot-product length used by per-tile target scaling: the
        common dimension floored to the MMA quantum (128 for Int7xInt7ToInt32).
        Mirrors Akoya ``MiningConfiguration.DotProductLength`` (floor-div)."""
        quantum = 128 if self.mma_type == MMA_TYPE_INT7XINT7_TO_INT32 else 128
        return (self.common_dim // quantum) * quantum

    def difficulty_adjustment_factor(self) -> int:
        """Per-tile difficulty scale: ``rows.size * cols.size * dot_product_length``.
        Each found tile represents this many MACs, so the protocol scales the
        share target UP by it: the miner's real target = ``diff_target * DAF``.
        Mirrors Akoya ``MiningConfiguration.DifficultyAdjustmentFactor`` and the
        ``adjusted = diffTarget * DAF`` in Akoya ``GpuWorker.InstallSigmaHalf``.
        For the live shape (rows=2, cols=64, k=4096) this is 2*64*4096 = 2^19."""
        return (self.rows_pattern.size() * self.cols_pattern.size()
                * self.dot_product_length())

    @classmethod
    def from_bytes(cls, data: bytes) -> "MiningConfiguration":
        if len(data) != MINING_CONFIG_SERIALIZED_SIZE:
            raise ValueError(f"expected {MINING_CONFIG_SERIALIZED_SIZE} bytes")
        common_dim = struct.unpack("<I", data[0:4])[0]
        rank = struct.unpack("<H", data[4:6])[0]
        mma_type = struct.unpack("<H", data[6:8])[0]
        rows_pattern = PeriodicPattern.from_bytes(data[8:14])
        cols_pattern = PeriodicPattern.from_bytes(data[14:20])
        reserved = data[20:52]
        return cls(common_dim=common_dim, rank=rank, mma_type=mma_type,
                   rows_pattern=rows_pattern, cols_pattern=cols_pattern,
                   reserved=reserved)


# ---------------------------------------------------------------------------- #
# Job key                                                                      #
# ---------------------------------------------------------------------------- #

def compute_job_key(header_bytes: bytes, mining_config: MiningConfiguration) -> bytes:
    """``job_key = blake3(header_bytes || mining_config.to_bytes())`` (default,
    non-keyed BLAKE3, 32-byte output). ``header_bytes`` is the canonical
    76-byte form already delivered on the wire as
    ``mining.notify`` param[2] (hex-decoded).
    """
    if len(header_bytes) != HEADER_SERIALIZED_SIZE:
        raise ValueError(f"header must be {HEADER_SERIALIZED_SIZE} bytes, got {len(header_bytes)}")
    return blake3.blake3(header_bytes + mining_config.to_bytes()).digest()


# ---------------------------------------------------------------------------- #
# Pool wire bridge                                                             #
# ---------------------------------------------------------------------------- #

_MMA_TYPE_BY_NAME = {
    "Int7xInt7ToInt32": MMA_TYPE_INT7XINT7_TO_INT32,
}


def mining_config_from_pool_params(p: dict) -> MiningConfiguration:
    """Build a :class:`MiningConfiguration` from the pool's
    ``pearl.set_mining_params`` payload object.

    Accepts the inner dict (the pool sends ``params = [<dict>]``, an array
    wrapping a single object; ``StratumSession`` already unwraps it).
    """
    try:
        common_dim = int(p["k"])
        rank = int(p["rank"])
        mma_name = p["mma_type"]
        rows_pattern = PeriodicPattern.from_list(list(p["rows_pattern"]))
        cols_pattern = PeriodicPattern.from_list(list(p["cols_pattern"]))
    except KeyError as e:
        raise ValueError(f"set_mining_params is missing field {e}") from None
    if mma_name not in _MMA_TYPE_BY_NAME:
        raise ValueError(f"unsupported mma_type {mma_name!r}")
    return MiningConfiguration(
        common_dim=common_dim,
        rank=rank,
        mma_type=_MMA_TYPE_BY_NAME[mma_name],
        rows_pattern=rows_pattern,
        cols_pattern=cols_pattern,
    )


# ---------------------------------------------------------------------------- #
# Selftests                                                                    #
# ---------------------------------------------------------------------------- #

def _selftest_nbits() -> None:
    """All six test cases from ``zk-pow/src/api/proof_utils.rs`` mod
    ``difficulty_tests``."""
    cases: list[tuple[int, int]] = [
        # (nbits, expected_target)
        (0x1D00FFFF, int("00000000ffff0000000000000000000000000000000000000000000000000000", 16)),
        (0x1B0404CB, int("00000000000404cb000000000000000000000000000000000000000000000000", 16)),
        (0x03123456, 0x123456),
        (0x1D000000, 0),
        (0x2077FFFF, 0x77FFFF << (29 * 8)),
        (0x1D800000, 0),  # negative bit set
    ]
    for nbits, expected in cases:
        got = nbits_to_target(nbits)
        if got != expected:
            raise AssertionError(
                f"nbits_to_target({nbits:#010x}) = {got:#x}, expected {expected:#x}")
    print(f"  nbits: {len(cases)} test vectors match Rust reference")


def _selftest_pattern() -> None:
    """``PeriodicPattern`` round-trip + the live pool's patterns from the
    2026-05-24 capture, plus the test fixture from ``proof_utils.rs``."""
    cases: list[list[int]] = [
        [0, 32],                                                                # rows_pattern (live)
        list(range(64)),                                                        # cols_pattern (live)
        [0, 8, 64, 72],                                                         # rust test fixture rows
        [0, 1, 8, 9, 32, 33, 40, 41, 64, 65, 72, 73, 96, 97, 104, 105],         # rust test fixture cols
        [0],                                                                    # trivial
    ]
    for indices in cases:
        pat = PeriodicPattern.from_list(indices)
        encoded = pat.to_bytes()
        decoded = PeriodicPattern.from_bytes(encoded)
        if pat.shape != decoded.shape:
            raise AssertionError(f"shape round-trip failed for {indices}: "
                                 f"{pat.shape} != {decoded.shape}")
        if pat.to_list() != indices:
            raise AssertionError(f"to_list round-trip failed for {indices}: "
                                 f"got {pat.to_list()}")
        if len(encoded) != PERIODIC_PATTERN_BYTES:
            raise AssertionError(f"encoded size {len(encoded)} != 6")
    print(f"  pattern: {len(cases)} round-trips OK; "
          f"e.g. [0,32]->shape={PeriodicPattern.from_list([0,32]).shape}, "
          f"[0..63]->shape={PeriodicPattern.from_list(list(range(64))).shape}")


def _selftest_mining_config() -> None:
    """Build a config matching the live pool capture, check size + roundtrip."""
    pool_params = {
        "m": 131072, "n": 131072, "k": 4096, "rank": 128,
        "rows_pattern": [0, 32],
        "cols_pattern": list(range(64)),
        "mma_type": "Int7xInt7ToInt32",
    }
    mc = mining_config_from_pool_params(pool_params)
    encoded = mc.to_bytes()
    assert len(encoded) == MINING_CONFIG_SERIALIZED_SIZE
    decoded = MiningConfiguration.from_bytes(encoded)
    assert mc.to_bytes() == decoded.to_bytes()
    print(f"  mining_config: {MINING_CONFIG_SERIALIZED_SIZE} bytes, "
          f"round-trip OK, k={mc.common_dim} rank={mc.rank} "
          f"mma_type={mc.mma_type}")


def _selftest_job_key() -> None:
    """job_key is deterministic; no Rust reference available without building
    the crate, but we can at least check the inputs hash stably."""
    pool_params = {
        "m": 131072, "n": 131072, "k": 4096, "rank": 128,
        "rows_pattern": [0, 32],
        "cols_pattern": list(range(64)),
        "mma_type": "Int7xInt7ToInt32",
    }
    mc = mining_config_from_pool_params(pool_params)
    # A real header captured live on 2026-05-24:
    header_hex = ("00004020f8d70ee30e785506ed2ddeda7a2b19c583f9f7dce4eeb6e4a620c82570f81c"
                  "1a6f4972e98960f0dc29dd251aff1e75b7e8221d06b065b27591f35003efcfe2c"
                  "4c5fd116af7810318")
    header_bytes = bytes.fromhex(header_hex)
    assert len(header_bytes) == HEADER_SERIALIZED_SIZE, (
        f"captured header is {len(header_bytes)} bytes, expected {HEADER_SERIALIZED_SIZE}")
    jk = compute_job_key(header_bytes, mc)
    assert len(jk) == 32
    # Stable across re-runs:
    jk2 = compute_job_key(header_bytes, mc)
    assert jk == jk2
    print(f"  job_key: {jk.hex()} (deterministic, 32 bytes)")


def _selftest_all() -> None:
    _selftest_nbits()
    _selftest_pattern()
    _selftest_mining_config()
    _selftest_job_key()
    print("all selftests OK")


if __name__ == "__main__":
    _selftest_all()
