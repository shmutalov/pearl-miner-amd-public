"""Pure-Python encoder/decoder for the ``PlainProof`` bytes the Pearl pool
expects in ``mining.submit``. Mirrors the bincode v1 serialization of the
Rust struct defined in ``reference/zk-pow/src/ffi/plain_proof.rs``.

The format is fixed (no ZK here yet — that wrapper is on the gateway side):

  PlainProof {
      m, n, k, noise_rank: u64 LE         # 4 × 8 bytes = 32 bytes
      a:  MatrixMerkleProof
      bt: MatrixMerkleProof
  }

  MatrixMerkleProof {
      proof: MerkleProof
      row_indices: Vec<u64>               # u64-len prefix + N × u64 LE
  }

  MerkleProof {
      leaf_data:    Vec<Vec<u8>>          # u64 outer count, then for each:
                                          #   u64 inner-len (== 1024) + 1024 bytes
                                          # (Rust uses [u8; 1024] but the serde
                                          # helper wraps it as &[u8] → Vec<u8>)
      leaf_indices: Vec<u64>              # u64-len + N × u64 LE
      total_leaves: u64                   # 8 bytes
      root:         [u8; 32]              # 32 bytes (no length prefix)
      siblings:     Vec<[u8; 32]>         # u64-len + N × 32 bytes

Bincode v1 default config: little-endian, no varints (u64 always 8 bytes),
``Vec<T>`` = ``u64 length || items``, fixed-size arrays = raw bytes.
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import BinaryIO

BLAKE3_CHUNK_LEN = 1024
BLAKE3_DIGEST_SIZE = 32


# ---------------------------------------------------------------------------- #
# Low-level bincode primitives                                                 #
# ---------------------------------------------------------------------------- #

def _pack_u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def _unpack_u64(buf: BinaryIO) -> int:
    raw = buf.read(8)
    if len(raw) != 8:
        raise ValueError("short read for u64")
    return struct.unpack("<Q", raw)[0]


def _read_exact(buf: BinaryIO, n: int) -> bytes:
    data = buf.read(n)
    if len(data) != n:
        raise ValueError(f"short read: wanted {n}, got {len(data)}")
    return data


def _pack_vec_u64(items: list[int]) -> bytes:
    return _pack_u64(len(items)) + b"".join(_pack_u64(x) for x in items)


def _unpack_vec_u64(buf: BinaryIO) -> list[int]:
    n = _unpack_u64(buf)
    return [_unpack_u64(buf) for _ in range(n)]


def _pack_vec_fixed_bytes(items: list[bytes], elem_size: int) -> bytes:
    out = bytearray(_pack_u64(len(items)))
    for b in items:
        if len(b) != elem_size:
            raise ValueError(f"expected {elem_size}-byte entry, got {len(b)}")
        out += b
    return bytes(out)


def _unpack_vec_fixed_bytes(buf: BinaryIO, elem_size: int) -> list[bytes]:
    n = _unpack_u64(buf)
    return [_read_exact(buf, elem_size) for _ in range(n)]


def _pack_vec_byte_chunks(items: list[bytes], chunk_size: int = BLAKE3_CHUNK_LEN) -> bytes:
    """Special-case for the ``leaf_data`` field: Rust stores ``Vec<[u8; 1024]>``
    but the serde helper wraps each entry as ``&[u8]``, so each chunk gets
    its own length prefix on the wire (== 1024 in every well-formed proof)."""
    out = bytearray(_pack_u64(len(items)))
    for b in items:
        if len(b) != chunk_size:
            raise ValueError(f"chunk must be {chunk_size} bytes, got {len(b)}")
        out += _pack_u64(len(b))
        out += b
    return bytes(out)


def _unpack_vec_byte_chunks(buf: BinaryIO, chunk_size: int = BLAKE3_CHUNK_LEN) -> list[bytes]:
    n = _unpack_u64(buf)
    chunks: list[bytes] = []
    for _ in range(n):
        inner_len = _unpack_u64(buf)
        if inner_len != chunk_size:
            raise ValueError(f"chunk length on wire = {inner_len}, expected {chunk_size}")
        chunks.append(_read_exact(buf, inner_len))
    return chunks


# ---------------------------------------------------------------------------- #
# PlainProof data classes + codec                                              #
# ---------------------------------------------------------------------------- #

@dataclass
class MerkleProof:
    leaf_data: list[bytes]          # each entry is BLAKE3_CHUNK_LEN bytes
    leaf_indices: list[int]
    total_leaves: int
    root: bytes                     # BLAKE3_DIGEST_SIZE bytes
    siblings: list[bytes]           # each entry BLAKE3_DIGEST_SIZE bytes

    def encode(self) -> bytes:
        if len(self.root) != BLAKE3_DIGEST_SIZE:
            raise ValueError(f"root must be {BLAKE3_DIGEST_SIZE} bytes")
        return b"".join([
            _pack_vec_byte_chunks(self.leaf_data),
            _pack_vec_u64(self.leaf_indices),
            _pack_u64(self.total_leaves),
            self.root,
            _pack_vec_fixed_bytes(self.siblings, BLAKE3_DIGEST_SIZE),
        ])

    @classmethod
    def decode(cls, buf: BinaryIO) -> "MerkleProof":
        leaf_data = _unpack_vec_byte_chunks(buf)
        leaf_indices = _unpack_vec_u64(buf)
        total_leaves = _unpack_u64(buf)
        root = _read_exact(buf, BLAKE3_DIGEST_SIZE)
        siblings = _unpack_vec_fixed_bytes(buf, BLAKE3_DIGEST_SIZE)
        return cls(leaf_data=leaf_data, leaf_indices=leaf_indices,
                   total_leaves=total_leaves, root=root, siblings=siblings)


@dataclass
class MatrixMerkleProof:
    proof: MerkleProof
    row_indices: list[int]

    def encode(self) -> bytes:
        return self.proof.encode() + _pack_vec_u64(self.row_indices)

    @classmethod
    def decode(cls, buf: BinaryIO) -> "MatrixMerkleProof":
        proof = MerkleProof.decode(buf)
        row_indices = _unpack_vec_u64(buf)
        return cls(proof=proof, row_indices=row_indices)


@dataclass
class PlainProof:
    m: int
    n: int
    k: int
    noise_rank: int
    a: MatrixMerkleProof
    bt: MatrixMerkleProof

    def encode(self) -> bytes:
        return b"".join([
            _pack_u64(self.m),
            _pack_u64(self.n),
            _pack_u64(self.k),
            _pack_u64(self.noise_rank),
            self.a.encode(),
            self.bt.encode(),
        ])

    @classmethod
    def decode(cls, data: bytes) -> "PlainProof":
        buf = io.BytesIO(data)
        m = _unpack_u64(buf)
        n = _unpack_u64(buf)
        k = _unpack_u64(buf)
        noise_rank = _unpack_u64(buf)
        a = MatrixMerkleProof.decode(buf)
        bt = MatrixMerkleProof.decode(buf)
        # Reject any trailing junk — catches encoding bugs early
        rest = buf.read()
        if rest:
            raise ValueError(f"trailing {len(rest)} bytes after PlainProof")
        return cls(m=m, n=n, k=k, noise_rank=noise_rank, a=a, bt=bt)


# ---------------------------------------------------------------------------- #
# Round-trip selftest                                                          #
# ---------------------------------------------------------------------------- #

def _selftest() -> None:
    """Build a tiny but structurally-valid PlainProof, encode + decode, check
    the bytes round-trip and that the size matches our hand calculation."""
    chunk = b"\xab" * BLAKE3_CHUNK_LEN
    digest = b"\x42" * BLAKE3_DIGEST_SIZE

    inner = MerkleProof(
        leaf_data=[chunk, chunk],
        leaf_indices=[0, 1],
        total_leaves=4,
        root=digest,
        siblings=[digest, digest],
    )
    mat = MatrixMerkleProof(proof=inner, row_indices=[3, 5])
    proof = PlainProof(m=131072, n=131072, k=4096, noise_rank=128,
                       a=mat, bt=mat)

    encoded = proof.encode()
    # Hand-calculate expected size:
    # 4 × u64 (m,n,k,noise_rank)               = 32
    # per MerkleProof:
    #   leaf_data:  u64 count(=2) + 2*(u64 + 1024) = 8 + 2*(8 + 1024) = 2072
    #   leaf_indices: u64 count(=2) + 2*u64 = 24
    #   total_leaves: u64 = 8
    #   root: 32
    #   siblings: u64 count(=2) + 2*32 = 72
    #   = 2208 bytes
    # MatrixMerkleProof = MerkleProof(2208) + row_indices(8 + 2*8 = 24) = 2232
    # PlainProof = 32 + 2 * 2232 = 4496
    expected_size = 32 + 2 * (2208 + 24)
    assert len(encoded) == expected_size, (
        f"encoded size {len(encoded)} != expected {expected_size}")

    decoded = PlainProof.decode(encoded)
    re_encoded = decoded.encode()
    assert encoded == re_encoded, "round-trip mismatch"
    print(f"selftest OK: {len(encoded)} bytes, round-trips byte-identical")


if __name__ == "__main__":
    _selftest()
