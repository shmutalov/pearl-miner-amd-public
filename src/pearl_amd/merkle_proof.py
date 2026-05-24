"""Build a multi-leaf Merkle proof over a keyed-BLAKE3 tree.

Mirrors ``MerkleTree::get_multileaf_proof`` in
``reference/pearl-blake3/src/merkle.rs``. The output ``MerkleProof`` plugs
straight into :class:`pearl_amd.plain_proof_codec.MatrixMerkleProof`.

We hand-roll BLAKE3's ``compress`` here so we can compute the internal
chunk-CV and parent-CV values that the Python ``blake3`` package hides.
Pure-Python compress is slow (~50 µs per block in CPython), so this path
is only practical at small/medium shapes for now — the OpenCL port will
take over once the GEMM pipeline lands.

Layout assumptions match the Rust side exactly:
  - CHUNK_LEN = 1024 bytes per leaf.
  - Each chunk = 16 blocks of 64 bytes; chunk_index = position in the data.
  - Default-mode chaining: IV is the keyed-hash variant when ``key`` is set.
  - Parent CVs combine two children via the BLAKE3 ``parent`` compress.
  - The root CV at the final level sets the ROOT flag.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

# ---------------------------------------------------------------------------- #
# BLAKE3 compress (pure Python)                                                #
# ---------------------------------------------------------------------------- #

CHUNK_LEN = 1024
BLOCK_LEN = 64
OUT_LEN = 32

IV = (
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
)

CHUNK_START = 1 << 0
CHUNK_END   = 1 << 1
PARENT      = 1 << 2
ROOT        = 1 << 3
KEYED_HASH  = 1 << 4

MSG_SCHEDULE = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8),
    (3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1),
    (10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6),
    (12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4),
    (9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7),
    (11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13),
)

MASK32 = 0xFFFFFFFF


def _rotr32(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & MASK32


def _g(v: list[int], a: int, b: int, c: int, d: int, mx: int, my: int) -> None:
    v[a] = (v[a] + v[b] + mx) & MASK32
    v[d] = _rotr32(v[d] ^ v[a], 16)
    v[c] = (v[c] + v[d]) & MASK32
    v[b] = _rotr32(v[b] ^ v[c], 12)
    v[a] = (v[a] + v[b] + my) & MASK32
    v[d] = _rotr32(v[d] ^ v[a], 8)
    v[c] = (v[c] + v[d]) & MASK32
    v[b] = _rotr32(v[b] ^ v[c], 7)


def compress(cv: tuple[int, ...], block: bytes, counter: int,
             block_len: int, flags: int) -> list[int]:
    """One BLAKE3 block compression. Returns the 16-word post-XOR state:

      out[0..8]  = v[0..8]  XOR v[8..16]   <- chaining value words
      out[8..16] = v[8..16] XOR cv[0..8]   <- second half of XOF stream

    Callers that need the chaining CV take ``out[0..8]``; callers that need
    a 32-byte digest take ``out[0..8]`` packed little-endian; callers that
    need the full 64-byte XOF stream take ``out[0..16]``.
    """
    if len(block) != BLOCK_LEN:
        raise ValueError("block must be 64 bytes")
    m = list(struct.unpack("<16I", block))
    v = [
        cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
        IV[0], IV[1], IV[2], IV[3],
        counter & MASK32,
        (counter >> 32) & MASK32,
        block_len & MASK32,
        flags & MASK32,
    ]
    for s in MSG_SCHEDULE:
        _g(v, 0, 4,  8, 12, m[s[0]],  m[s[1]])
        _g(v, 1, 5,  9, 13, m[s[2]],  m[s[3]])
        _g(v, 2, 6, 10, 14, m[s[4]],  m[s[5]])
        _g(v, 3, 7, 11, 15, m[s[6]],  m[s[7]])
        _g(v, 0, 5, 10, 15, m[s[8]],  m[s[9]])
        _g(v, 1, 6, 11, 12, m[s[10]], m[s[11]])
        _g(v, 2, 7,  8, 13, m[s[12]], m[s[13]])
        _g(v, 3, 4,  9, 14, m[s[14]], m[s[15]])
    # Final XOR fold — the standard BLAKE3 compress output.
    for i in range(8):
        v[i] ^= v[i + 8]
        v[i + 8] ^= cv[i]
    return v


def _key_words(key: bytes) -> tuple[int, ...]:
    if len(key) != OUT_LEN:
        raise ValueError("key must be 32 bytes")
    return struct.unpack("<8I", key)


def chunk_cv(data: bytes, chunk_index: int, key: bytes,
             is_root: bool = False) -> bytes:
    """Compute the chaining value of a single BLAKE3 chunk (≤ 1024 bytes)
    with ``KEYED_HASH`` flag set. ``chunk_index`` becomes the counter.

    When ``is_root=True`` the final block of the chunk additionally gets
    the ``ROOT`` flag (used when the entire input fits in one chunk).
    """
    if len(data) > CHUNK_LEN:
        raise ValueError("chunk must be ≤ 1024 bytes")
    # Pad the LAST partial block (if any) with zeros to a full 64-byte block.
    n_blocks = (len(data) + BLOCK_LEN - 1) // BLOCK_LEN
    if n_blocks == 0:
        n_blocks = 1  # empty data still produces a CV via one zero block
    padded = data + b"\x00" * (n_blocks * BLOCK_LEN - len(data))
    cv = _key_words(key)
    bytes_remaining = len(data)
    for i in range(n_blocks):
        block = padded[i * BLOCK_LEN:(i + 1) * BLOCK_LEN]
        flags = KEYED_HASH
        if i == 0:
            flags |= CHUNK_START
        if i == n_blocks - 1:
            flags |= CHUNK_END
            if is_root:
                flags |= ROOT
        block_len = min(BLOCK_LEN, bytes_remaining) if bytes_remaining > 0 else 0
        v = compress(cv, block, chunk_index, block_len, flags)
        # compress already returns post-XOR words: cv_next = v[0..8].
        cv = tuple(v[j] & MASK32 for j in range(8))
        bytes_remaining -= block_len
    return struct.pack("<8I", *cv)


def parent_cv(left: bytes, right: bytes, key: bytes,
              is_root: bool = False) -> bytes:
    """Combine two child CVs into one parent CV via BLAKE3's parent compression.
    The output is the XOR-fold ``v[0..7] ^ v[8..15]`` per BLAKE3 spec."""
    if len(left) != OUT_LEN or len(right) != OUT_LEN:
        raise ValueError("child CVs must be 32 bytes")
    block = left + right  # 64 bytes
    flags = KEYED_HASH | PARENT
    if is_root:
        flags |= ROOT
    v = compress(_key_words(key), block, 0, BLOCK_LEN, flags)
    return struct.pack("<8I", *(v[j] & MASK32 for j in range(8)))


# ---------------------------------------------------------------------------- #
# Multi-leaf proof builder                                                     #
# ---------------------------------------------------------------------------- #

@dataclass
class MerkleProofData:
    leaf_data: list[bytes]       # each 1024 bytes (last may be padded)
    leaf_indices: list[int]
    total_leaves: int
    root: bytes
    siblings: list[bytes]


def _compute_layer_cvs(prev_layer: list[bytes], key: bytes,
                       is_top: bool) -> list[bytes]:
    """Combine adjacent CVs to produce the next layer up. If the previous
    layer has odd length, the last CV is carried through unchanged (the
    BLAKE3 spec's "unbalanced" promotion)."""
    out: list[bytes] = []
    i = 0
    is_root_layer = is_top and len(prev_layer) == 2
    while i + 1 < len(prev_layer):
        out.append(parent_cv(prev_layer[i], prev_layer[i + 1], key,
                             is_root=is_root_layer))
        i += 2
    if i < len(prev_layer):
        out.append(prev_layer[i])
    return out


def build_merkle_tree(data: bytes, key: bytes) -> list[list[bytes]]:
    """Compute all layers of the keyed Merkle tree over ``data``.

    Matches ``MerkleTree::new`` in ``reference/pearl-blake3``:
      - ≤ CHUNK_LEN: single ROOT chunk over the real (possibly partial) data.
      - > CHUNK_LEN: split into 1024-byte chunks (last may be partial),
        each hashed as a non-ROOT chunk CV; combine pairwise upward; the
        final two-into-one parent compress carries the ROOT flag.
    """
    if len(data) == 0:
        return [[]]
    if len(data) <= CHUNK_LEN:
        # Single chunk: use the real data length so block_len in the last
        # compression matches; ROOT flag applied to the final block.
        return [[chunk_cv(data, 0, key, is_root=True)]]
    n_leaves = (len(data) + CHUNK_LEN - 1) // CHUNK_LEN
    leaves: list[bytes] = []
    for i in range(n_leaves):
        chunk = data[i * CHUNK_LEN:(i + 1) * CHUNK_LEN]  # may be partial for last
        leaves.append(chunk_cv(chunk, i, key))
    layers: list[list[bytes]] = [leaves]
    while len(layers[-1]) > 1:
        is_top = len(layers[-1]) == 2
        layers.append(_compute_layer_cvs(layers[-1], key, is_top=is_top))
    return layers


def compute_leaf_indices_from_rows(row_indices: list[int], cols: int
                                   ) -> list[int]:
    """Which leaves cover the requested matrix rows (rows × cols bytes)?
    Mirrors ``MerkleTree::compute_leaf_indices_from_rows``.
    """
    indices: set[int] = set()
    for row in row_indices:
        first = (row * cols) // CHUNK_LEN
        last = ((row + 1) * cols - 1) // CHUNK_LEN
        indices.update(range(first, last + 1))
    return sorted(indices)


def get_multileaf_proof(layers: list[list[bytes]], matrix_bytes: bytes,
                        leaf_indices: list[int]) -> MerkleProofData:
    """Given a built Merkle ``layers`` and the underlying ``matrix_bytes``,
    extract a multi-leaf proof for ``leaf_indices``.
    """
    if not leaf_indices:
        raise ValueError("leaf_indices must be non-empty")
    n_leaves = len(layers[0])
    if leaf_indices[-1] >= n_leaves:
        raise ValueError("leaf index out of bounds")
    sorted_indices = sorted(set(leaf_indices))

    # Extract leaf data, pad last to 1024 if needed.
    leaf_data: list[bytes] = []
    for i in sorted_indices:
        start = i * CHUNK_LEN
        end = min(start + CHUNK_LEN, len(matrix_bytes))
        chunk = matrix_bytes[start:end]
        if len(chunk) < CHUNK_LEN:
            chunk = chunk + b"\x00" * (CHUNK_LEN - len(chunk))
        leaf_data.append(chunk)

    # Walk levels, collecting siblings.
    siblings: list[bytes] = []
    current = set(sorted_indices)
    level_len = n_leaves
    level = 0
    while level_len > 1 and current:
        level_nodes = layers[level]
        for i in sorted(current):
            if i % 2 == 1:
                if (i - 1) not in current:
                    siblings.append(level_nodes[i - 1])
            else:
                if (i + 1) not in current and (i + 1) < level_len:
                    siblings.append(level_nodes[i + 1])
        current = {i // 2 for i in current}
        level_len = (level_len + 1) // 2
        level += 1

    root = layers[-1][0] if layers and layers[-1] else b"\x00" * OUT_LEN
    return MerkleProofData(
        leaf_data=leaf_data,
        leaf_indices=sorted_indices,
        total_leaves=n_leaves,
        root=root,
        siblings=siblings,
    )


# ---------------------------------------------------------------------------- #
# Selftest                                                                     #
# ---------------------------------------------------------------------------- #

def _selftest_root_matches_library() -> None:
    """The Merkle root we compute MUST equal ``blake3-keyed(data, key=key)``."""
    import blake3
    key = bytes(range(32))
    for nbytes in [1, 1024, 2048, 3000, 10240]:
        data = bytes((i * 7 + 13) & 0xFF for i in range(nbytes))
        layers = build_merkle_tree(data, key)
        our_root = layers[-1][0]
        ref_root = blake3.blake3(data, key=key).digest()
        if our_root != ref_root:
            raise AssertionError(
                f"root mismatch at n={nbytes}: ours={our_root.hex()} ref={ref_root.hex()}")
    print(f"  Merkle root matches blake3-keyed for sizes [1, 1024, 2048, 3000, 10240]")


def _selftest_proof_roundtrip() -> None:
    """Build a proof, then re-derive the root from leaf_data + siblings
    using the same primitives, and verify it matches the tree's root."""
    key = bytes(range(32))
    n_leaves = 16
    data = bytes((i * 31 + 11) & 0xFF for i in range(n_leaves * CHUNK_LEN))
    layers = build_merkle_tree(data, key)
    proof = get_multileaf_proof(layers, data, [0, 3, 5])

    # Replay: compute leaf CVs from leaf_data, then walk up using siblings.
    leaf_cvs = {idx: chunk_cv(d, idx, key) for idx, d in zip(proof.leaf_indices, proof.leaf_data)}
    current = dict(leaf_cvs)
    level_len = proof.total_leaves
    sib_iter = iter(proof.siblings)
    while level_len > 1:
        next_layer: dict[int, bytes] = {}
        for i in sorted(current):
            if i % 2 == 0:
                left = current[i]
                if (i + 1) in current:
                    right = current[i + 1]
                elif i + 1 < level_len:
                    right = next(sib_iter)
                else:
                    next_layer[i // 2] = left
                    continue
                is_root = (level_len == 2)
                next_layer[i // 2] = parent_cv(left, right, key, is_root=is_root)
            else:
                if (i - 1) in current:
                    continue
                left = next(sib_iter)
                right = current[i]
                is_root = (level_len == 2)
                next_layer[i // 2] = parent_cv(left, right, key, is_root=is_root)
        current = next_layer
        level_len = (level_len + 1) // 2

    assert list(current.values())[0] == proof.root, "proof replay does not reach root"
    print(f"  multi-leaf proof for indices [0,3,5]: {len(proof.siblings)} siblings, "
          f"root reconstructs ✓")


def _selftest_compute_leaf_indices() -> None:
    cols = 4096
    # Rows 0 and 32 of a (m, k) matrix → each row spans 4 chunks
    idxs = compute_leaf_indices_from_rows([0, 32], cols)
    expected = [0, 1, 2, 3, 128, 129, 130, 131]
    assert idxs == expected, f"got {idxs}"
    print(f"  compute_leaf_indices_from_rows: rows [0,32] cols=4096 → leaves {idxs}")


if __name__ == "__main__":
    _selftest_compute_leaf_indices()
    _selftest_root_matches_library()
    _selftest_proof_roundtrip()
    print("all selftests OK")
