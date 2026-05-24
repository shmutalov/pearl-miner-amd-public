"""OpenCL host for BLAKE3-keyed Merkle layer construction.

Replaces the ~10 min/matrix pure-Python ``build_merkle_tree`` in
``merkle_proof.py`` with a GPU pipeline that runs in <100 ms at pool
shape (m*k = 512 MiB → 524288 leaves → 19 reduction levels).

Two kernels behind one class:
  - ``blake3_chunk_cvs``: one work-item per 1024-byte chunk, produces the
    full layer of leaf CVs in one launch.
  - ``blake3_merkle_layer``: combines adjacent CV pairs into parents.
    Called once per tree level; the final level gets the ROOT flag.

Output layout matches the host-side ``MerkleProofData`` consumers in
``merkle_proof.get_multileaf_proof``: a Python list of layers, each layer
a list of 32-byte CVs, in the order ``[leaves, level1, ..., root]``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyopencl as cl

from .device import find_gpu

CHUNK_LEN = 1024
OUT_LEN = 32

_KERNEL_PATH = Path(__file__).resolve().parent.parent / "kernels" / "blake3_merkle.cl"


def _pad_to_chunk(data: bytes) -> bytes:
    pad = (-len(data)) % CHUNK_LEN
    return data + b"\x00" * pad if pad else data


class _NumpyLayer:
    """List-like façade over a (n, 32) uint8 numpy array of CVs.

    Pool shape has 524288 leaves per matrix; materializing that as a Python
    list of 524288 ``bytes`` objects costs hundreds of milliseconds of
    pure-Python allocation. We keep the data as a single ndarray and lazily
    return ``bytes`` only on the few ``__getitem__`` calls actually made
    during proof construction (~128 leaves + ~20 siblings per hit).

    Supports the subset of ``list[bytes]`` operations that
    ``merkle_proof.get_multileaf_proof`` actually uses: ``__getitem__``,
    ``__len__``, ``__iter__`` (over bytes), and equality against either
    another ``_NumpyLayer`` or a plain ``list[bytes]`` (for cross-checks).
    """

    __slots__ = ("_arr",)

    def __init__(self, arr: np.ndarray) -> None:
        if arr.ndim != 2 or arr.shape[1] != OUT_LEN or arr.dtype != np.uint8:
            raise ValueError(
                f"_NumpyLayer expects (n, {OUT_LEN}) uint8, got {arr.shape} {arr.dtype}")
        self._arr = arr

    def __len__(self) -> int:
        return int(self._arr.shape[0])

    def __getitem__(self, i: int) -> bytes:
        return self._arr[i].tobytes()

    def __iter__(self):
        for i in range(self._arr.shape[0]):
            yield self._arr[i].tobytes()

    def to_list(self) -> list[bytes]:
        """Force-materialize as a list of bytes. Used in tests; avoid in
        hot paths since this is the expensive Python-allocation we are
        explicitly trying NOT to do."""
        return [self._arr[i].tobytes() for i in range(self._arr.shape[0])]

    def __eq__(self, other) -> bool:
        if isinstance(other, _NumpyLayer):
            return np.array_equal(self._arr, other._arr)
        if isinstance(other, list):
            if len(other) != self._arr.shape[0]:
                return False
            for i, b in enumerate(other):
                if self._arr[i].tobytes() != b:
                    return False
            return True
        return NotImplemented

    def __ne__(self, other) -> bool:
        eq = self.__eq__(other)
        return NotImplemented if eq is NotImplemented else not eq

    def __repr__(self) -> str:
        return f"_NumpyLayer({self._arr.shape[0]} CVs)"


class MerkleGpu:
    """Reusable PyOpenCL context + program for BLAKE3 Merkle construction.

    Construct once, call :meth:`build_layers` per job. The kernels are
    JIT-built on first use and cached for the lifetime of the instance.

    Pass ``context`` / ``queue`` to share OpenCL state with sibling
    classes; ``build_layers`` then accepts ``data_buf=`` to consume an
    already-on-device input matrix without re-uploading through PCIe.
    """

    def __init__(self, context: cl.Context | None = None,
                 queue: cl.CommandQueue | None = None,
                 device: cl.Device | None = None) -> None:
        if context is None:
            self.device = device or find_gpu()
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context, device=self.device)
        else:
            self.context = context
            self.queue = queue or cl.CommandQueue(context)
            self.device = device or self.queue.device
        src = _KERNEL_PATH.read_text(encoding="utf-8")
        self.program = cl.Program(self.context, src).build()
        self.k_chunk = self.program.blake3_chunk_cvs
        self.k_layer = self.program.blake3_merkle_layer
        self.k_single = self.program.blake3_single_chunk_root

    # ------------------------------------------------------------------- #
    # Public                                                              #
    # ------------------------------------------------------------------- #

    def build_layers(self, data: bytes | None, key: bytes,
                     *, data_buf: cl.Buffer | None = None,
                     n_bytes: int | None = None) -> list["_NumpyLayer"]:
        """Build all Merkle tree layers for the input keyed by ``key``.

        Either pass raw ``data`` bytes (the kernel uploads and pads them)
        OR pass ``data_buf`` + ``n_bytes`` to reuse an existing on-device
        buffer. Returns the same nested-list shape
        ``merkle_proof.build_merkle_tree`` produces: ``[leaves, level1,
        ..., [root]]``.

        The device buffer path requires ``n_bytes`` to be a multiple of
        ``CHUNK_LEN`` (= 1024) — at pool shape ``m*k = 512 MiB`` always
        is. Per-chunk tail-padding for non-multiple sizes is only
        implemented in the host-upload path.
        """
        if len(key) != OUT_LEN:
            raise ValueError("key must be 32 bytes")

        if data_buf is not None:
            if n_bytes is None or n_bytes <= 0:
                raise ValueError("n_bytes must be a positive int when data_buf is given")
            if n_bytes % CHUNK_LEN != 0:
                raise ValueError(
                    f"device-buffer path needs n_bytes multiple of {CHUNK_LEN}; "
                    f"got {n_bytes}")
            n_leaves = n_bytes // CHUNK_LEN
        else:
            if data is None:
                raise ValueError("either data or data_buf must be supplied")
            if not data:
                return [_NumpyLayer(np.empty((0, OUT_LEN), dtype=np.uint8))]
            if len(data) <= CHUNK_LEN:
                root_bytes = self._single_chunk_root(data, key)
                arr = np.frombuffer(root_bytes, dtype=np.uint8).reshape(1, OUT_LEN).copy()
                return [_NumpyLayer(arr)]
            n_leaves = (len(data) + CHUNK_LEN - 1) // CHUNK_LEN

        ctx, mf = self.context, cl.mem_flags
        key_words = np.frombuffer(key, dtype=np.uint32).copy()
        key_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=key_words)

        if data_buf is None:
            padded = _pad_to_chunk(data)
            data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                  hostbuf=np.frombuffer(padded, dtype=np.uint8))
        cvs0_buf = cl.Buffer(ctx, mf.READ_WRITE, size=n_leaves * OUT_LEN)
        self.k_chunk.set_args(data_buf, cvs0_buf, key_buf, np.uint32(n_leaves))
        cl.enqueue_nd_range_kernel(self.queue, self.k_chunk, (n_leaves,), None)

        # Iterate Merkle layers in-place: cvs_in → cvs_out, swap roles.
        layers_dev: list[cl.Buffer] = [cvs0_buf]
        cur_in = cvs0_buf
        cur_n = n_leaves
        while cur_n > 1:
            n_out = cur_n // 2
            promoted: bytes | None = None
            if cur_n & 1:
                # Odd → last CV is promoted to the next layer unchanged.
                # We allocate n_out+1 in the next buffer and copy the
                # promoted CV in after the kernel run.
                promote_idx = cur_n - 1
                # Read the promoted CV from cur_in into host so we can
                # copy it into the right slot in cur_out.
                tmp = np.empty(8, dtype=np.uint32)
                cl.enqueue_copy(self.queue, tmp, cur_in,
                                src_offset=promote_idx * OUT_LEN)
                self.queue.finish()
                promoted = tmp.tobytes()

            next_n = n_out + (1 if promoted else 0)
            cur_out = cl.Buffer(ctx, mf.READ_WRITE, size=next_n * OUT_LEN)
            is_root_layer = 1 if (cur_n == 2) else 0
            self.k_layer.set_args(cur_in, cur_out, key_buf,
                                  np.uint32(cur_n), np.int32(is_root_layer))
            cl.enqueue_nd_range_kernel(self.queue, self.k_layer, (n_out,), None)
            if promoted is not None:
                # Place promoted CV at index n_out.
                tmp = np.frombuffer(promoted, dtype=np.uint32).copy()
                cl.enqueue_copy(self.queue, cur_out, tmp,
                                dest_offset=n_out * OUT_LEN)

            layers_dev.append(cur_out)
            cur_in = cur_out
            cur_n = next_n
        self.queue.finish()

        # Read all layers back to host as ndarray. Pool shape: 524288 leaves
        # × 32 bytes = 16 MiB for the first layer; subsequent layers halve.
        # Wrapping each layer as a _NumpyLayer avoids the ~250 ms of pure-
        # Python bytes-list construction that was the dominant CPU cost on
        # the preflight critical path.
        host_layers: list[_NumpyLayer] = []
        n_for_layer = n_leaves
        for buf in layers_dev:
            raw = np.empty((n_for_layer, OUT_LEN), dtype=np.uint8)
            cl.enqueue_copy(self.queue, raw, buf)
            self.queue.finish()
            host_layers.append(_NumpyLayer(raw))
            n_for_layer = (n_for_layer + 1) // 2  # next layer size (with odd handling)
        return host_layers

    # ------------------------------------------------------------------- #
    # Internal: single-chunk root (≤ 1024 bytes) special-case             #
    # ------------------------------------------------------------------- #

    def _single_chunk_root(self, data: bytes, key: bytes) -> bytes:
        ctx, mf = self.context, cl.mem_flags
        data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.frombuffer(data, dtype=np.uint8))
        out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=OUT_LEN)
        key_words = np.frombuffer(key, dtype=np.uint32).copy()
        key_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=key_words)
        self.k_single.set_args(data_buf, out_buf, key_buf, np.uint32(len(data)))
        cl.enqueue_nd_range_kernel(self.queue, self.k_single, (1,), None)
        raw = np.empty(8, dtype=np.uint32)
        cl.enqueue_copy(self.queue, raw, out_buf)
        self.queue.finish()
        return raw.tobytes()


# ---------------------------------------------------------------------------- #
# Selftest                                                                     #
# ---------------------------------------------------------------------------- #

def _selftest_against_cpu() -> None:
    import time

    import blake3

    from .merkle_proof import build_merkle_tree, get_multileaf_proof

    gpu = MerkleGpu()
    print(f"  device: {gpu.device.name} ({gpu.device.max_compute_units} CUs)")

    key = bytes(range(32))
    sizes_with_short_oracle = [1, 1024, 2048, 65536, 1024 * 256]  # up to 256 KiB
    for n in sizes_with_short_oracle:
        data = bytes((i * 7 + 13) & 0xFF for i in range(n))
        host_layers = gpu.build_layers(data, key)
        ref_root = blake3.blake3(data, key=key).digest()
        gpu_root = host_layers[-1][0]
        if gpu_root != ref_root:
            raise AssertionError(
                f"n={n}: GPU root {gpu_root.hex()} != ref {ref_root.hex()}")
        # Also check vs the Python tree builder (slow at large n, so cap)
        if n <= 65536:
            cpu_layers = build_merkle_tree(data, key)
            assert len(cpu_layers) == len(host_layers), (
                f"layer count mismatch at n={n}: cpu={len(cpu_layers)} gpu={len(host_layers)}")
            for li, (cl_, gl_) in enumerate(zip(cpu_layers, host_layers)):
                if cl_ != gl_:
                    raise AssertionError(f"layer {li} mismatch at n={n}")
        print(f"  n={n:>8}: {len(host_layers)} layers, root={gpu_root[:8].hex()}... ✓")

    # Multi-leaf proof end-to-end check — build layers on GPU, then
    # construct proof using the Python helper; verify the reconstructed
    # root matches the GPU root.
    n = 16384  # 16 leaves at CHUNK_LEN
    data = bytes((i * 31 + 5) & 0xFF for i in range(n))
    host_layers = gpu.build_layers(data, key)
    proof = get_multileaf_proof(host_layers, data, [0, 3, 5, 14])
    print(f"  multi-leaf proof over GPU-built tree ({len(proof.siblings)} siblings)")
    # Re-derive root from leaf_data + siblings via merkle_proof.parent_cv...
    from .merkle_proof import chunk_cv, parent_cv
    cur = {idx: chunk_cv(d, idx, key) for idx, d in zip(proof.leaf_indices, proof.leaf_data)}
    sib_iter = iter(proof.siblings)
    level_len = proof.total_leaves
    while level_len > 1:
        nxt: dict[int, bytes] = {}
        for i in sorted(cur):
            if i % 2 == 0:
                left = cur[i]
                if (i + 1) in cur:
                    right = cur[i + 1]
                elif i + 1 < level_len:
                    right = next(sib_iter)
                else:
                    nxt[i // 2] = left
                    continue
                nxt[i // 2] = parent_cv(left, right, key, is_root=(level_len == 2))
            else:
                if (i - 1) in cur:
                    continue
                left = next(sib_iter)
                right = cur[i]
                nxt[i // 2] = parent_cv(left, right, key, is_root=(level_len == 2))
        cur = nxt
        level_len = (level_len + 1) // 2
    assert list(cur.values())[0] == proof.root, "proof replay fails"
    print(f"    proof reconstructs to GPU root ✓")


def _bench_pool_shape() -> None:
    import time

    import blake3

    gpu = MerkleGpu()
    key = bytes(range(32))

    # m * k at pool shape = 512 MiB
    n_bytes = 131072 * 4096
    print(f"  generating {n_bytes//(1024*1024)} MiB of test data via blake3 XOF... ",
          end="", flush=True)
    t0 = time.time()
    # Cheap fill: tile a small pattern so we don't spend selftest time on data gen.
    pattern = blake3.blake3(b"merkle bench").digest(length=4 * 1024 * 1024)
    n_tiles = (n_bytes + len(pattern) - 1) // len(pattern)
    data = (pattern * n_tiles)[:n_bytes]
    print(f"{time.time() - t0:.1f}s")

    # Warm-up
    print(f"  building Merkle tree on GPU ({n_bytes // CHUNK_LEN} leaves)...")
    t0 = time.time()
    layers = gpu.build_layers(data, key)
    dt = time.time() - t0
    print(f"  {len(layers)} layers built in {dt*1000:.0f}ms  "
          f"({n_bytes/(1024*1024)/dt:.0f} MiB/s effective)")

    # Sanity: GPU root must match blake3-keyed
    ref_root = blake3.blake3(data, key=key).digest()
    assert layers[-1][0] == ref_root, f"root mismatch (GPU: {layers[-1][0].hex()}, ref: {ref_root.hex()})"
    print(f"  root matches blake3-keyed reference ✓")


if __name__ == "__main__":
    print("== correctness ==")
    _selftest_against_cpu()
    print("\n== pool-shape bench ==")
    _bench_pool_shape()
