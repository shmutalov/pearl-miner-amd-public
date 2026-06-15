# pearl-miner-amd

Pearl PoUW miner ported to AMD GPUs via OpenCL. Targets the **Polaris**
generation (RX 470/480/570/580, gfx803) — the cheapest 8 GiB cards that
nobody else wants. Just-for-fun research project.

> **Status:** end-to-end pipeline works against real pool stratum
> (`eu1.alphapool.tech:5566`): handshake, BLAKE3 challenge, mining params,
> per-job preflight, GPU candidate search, PlainProof assembly, share
> submit. At ~26 000 candidate evaluations / second on an RX 570, this
> can't realistically win shares against modern Turing/Ada miners on the
> production pool's minimum difficulty (~44 000 years per share at
> `d=20000`) — but every stage is mathematically correct and
> bit-identical to the reference Rust implementation.

## What's interesting here

The journey from *36 attempts/sec on Python+CPU* to *~26 000 attempts/sec
end-to-end on a $80 used Polaris card* — without any hardware int8 dot-4
support (`V_DOT4_I32_I8` only appeared in gfx906). The repo is structured
as a sequence of small commits each with a measured speedup so you can
read it like a tutorial:

| Stage | Before | After | Notes |
|---|---|---|---|
| BLAKE3 anti-DoS challenge | CPU only | 958 MH/s | one-shot GPU solver |
| Matrix derivation (`derive_matrix`) | 5.4 s | 370 ms | GPU BLAKE3 XOF |
| BLAKE3-keyed Merkle layers (512 MiB) | 10 min Python | 210 ms | GPU chunk+layer kernels |
| Noise derivation (4 matrices) | 4.9 s | 10 ms | GPU keyed-BLAKE3 single-block |
| Jackpot evaluator | 12 ms/candidate (CPU) | 27 000 cand/s (batched GPU) | LDS A-strip cache, `mad24`, wavefront-sync XOR reduce |
| **Per-job preflight (pool shape)** | **~30 s** | **~410 ms** | shared OpenCL context, device-buffer reuse |
| **Search loop end-to-end** | ~80 cand/s (Python) | **~26 500 cand/s (GPU)** | `JackpotGpu.search` |

Other things you may find useful:

- **`docs/pearl-stratum-protocol.md`** — full reverse-engineered protocol
  spec for `pearl/v1` stratum dialect, including the BLAKE3 anti-DoS
  challenge format and the `PlainProof` bincode layout.
- **`src/pearl_amd/plain_proof_codec.py`** — pure-Python bincode v1
  encoder/decoder for `PlainProof`, plus all 6 Rust `nbits` test vectors.
- **`src/pearl_amd/jackpot.py`** — readable port of `pearl_noise.rs` +
  `jackpot/helper.rs` (CPU oracle used to verify the GPU path).
- **`src/kernels/*.cl`** — OpenCL kernels:
  `blake3_challenge.cl` (anti-DoS solver), `blake3_xof.cl`
  (`derive_matrix`), `blake3_merkle.cl` (chunk+layer), `pearl_noise.cl`
  (uniform + perm noise), `jackpot_search.cl` (batched evaluator), plus
  `jackpot_search_rdna3.cl` / `jackpot_search_rdna3_wtile.cl` (wave32
  RDNA3 evaluators — see the RDNA3 section below).

## Project structure

```
src/
├── kernels/                # OpenCL C kernels
│   ├── blake3_challenge.cl
│   ├── blake3_xof.cl
│   ├── blake3_merkle.cl
│   ├── pearl_noise.cl
│   ├── jackpot_search.cl
│   ├── jackpot_search_rdna3.cl       # wave32 RDNA3 evaluator
│   └── jackpot_search_rdna3_wtile.cl # + w-tiled for occupancy
└── pearl_amd/              # Python host + CPU oracles
    ├── device.py           # find_gpu()
    ├── stratum_client.py   # pearl/v1 JSON-RPC client
    ├── stratum_session.py  # high-level Work aggregator
    ├── mining_config.py    # MiningConfiguration + nbits decode + job_key
    ├── plain_proof_codec.py# bincode v1 encoder/decoder
    ├── proof_builder.py    # derive_AB + Merkle root
    ├── merkle_proof.py     # pure-Python BLAKE3 + multi-leaf proof
    ├── jackpot.py          # CPU noise + jackpot oracle
    ├── candidate_search.py # CPU search loop (oracle)
    ├── derive_matrix_gpu.py# GPU BLAKE3 XOF host
    ├── merkle_gpu.py       # GPU Merkle layer host
    ├── pearl_noise_gpu.py  # GPU noise derivation host
    ├── jackpot_gpu.py      # GPU batched evaluator + search
    ├── blake3_challenge_gpu.py
    └── miner.py            # PearlMiner orchestrator
scripts/
├── 34_miner_offline.py     # offline end-to-end smoke test
└── 35_run_miner_live.py    # live pool runner (dry-run by default)
docs/
└── pearl-stratum-protocol.md
```

## Quick start

Tested on Windows 10 + RX 570 8 GiB + AMD Adrenalin OpenCL driver.
Linux with ROCm or AMDGPU-Pro OpenCL should also work but isn't
regularly tested.

```bash
git clone <this-repo>
cd pearl-miner-amd
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

# Verify GPU discovery
.venv/Scripts/python.exe -c "from src.pearl_amd.device import list_devices; list_devices()"

# Run the offline pipeline smoke test (no network needed)
.venv/Scripts/python.exe -m src.scripts.34_miner_offline

# Bench any single component at pool shape (m=n=131072, k=4096):
.venv/Scripts/python.exe -m src.pearl_amd.derive_matrix_gpu
.venv/Scripts/python.exe -m src.pearl_amd.merkle_gpu
.venv/Scripts/python.exe -m src.pearl_amd.pearl_noise_gpu
.venv/Scripts/python.exe -m src.pearl_amd.jackpot_gpu

# Live run against a real pool (dry-run = does NOT submit shares):
.venv/Scripts/python.exe src/scripts/35_run_miner_live.py \
    --address <your-prl1...-wallet> \
    --worker rx570 \
    --password 'x;d=20000' \
    --observe-seconds 90 \
    --max-attempts-per-job 100000
```

To actually submit shares, add `--submit`. Beware: at pool minimum
difficulty (20 000 on alphapool.tech) finding a single share takes
~10⁴⁴ seconds on an RX 570. This is normal — the code is correct, the
hardware is just slow for this workload.

## Limitations / what's NOT here

- **No share verifier**: we build `PlainProof` bytes but don't have a
  local Merkle+jackpot verifier. The pool does that; locally we only
  check that our `PlainProof` round-trips byte-identically through the
  bincode codec.
- **No vardiff handling**: we honor whatever `share_nbits` the pool
  sends but don't adjust strategy based on it.
- **No multi-GPU**: single device only.
- **No native int8 dot-4**: gfx803 lacks the instruction. The AMD
  Windows OpenCL driver doesn't expose `cl_khr_integer_dot_product`
  even on RDNA3, so DP4A isn't reachable from a clean builtin — and the
  inner loop turns out to be *gather*-bound (sparse-permutation noise),
  not MAC-bound, so DP4A wouldn't be the win anyway.

## RDNA3 (gfx11xx) kernel

Two wave32-native variants of the jackpot evaluator, auto-selected on
gfx10/11/12 parts (override with
`JackpotGpu(..., variant="polaris"|"rdna3"|"rdna3_wtile")`).

First, the GCN kernel is **already bit-correct on RDNA3 wave32** — its
final XOR tree sits exactly at the 32-lane boundary, after a barrier, so
it never reads an unsynchronized subgroup. The old "different SIMD width
will misbehave" warning was overly cautious.

The speedup is algorithmic, not from DP4A. The per-candidate jackpot is
the tiny GEMM `acc[u][v] = Σ_l (A+noise_a)[u][l]·(B+noise_b)[v][l]`, and
the GCN kernel recomputes the noisy `pa[u][l]` operand 64× (once per `v`).
`jackpot_search_rdna3.cl` builds the `pa`/`pb` strips cooperatively in LDS
once per outer iter, leaving a clean char4-vectorized int8 dot inner loop.

That doubled the LDS footprint (17 KiB/WG) and halved occupancy — and the
kernel is **gather/latency-bound**, so occupancy matters. `…_rdna3_wtile.cl`
tiles the `w` dimension (default `PEARL_NTILES_W=4`) so the two big LDS
buffers shrink to `TW·r`, dropping LDS to ~5.4 KiB/WG (~11 workgroups per
WGP) — the measured occupancy sweet spot. Measured on an RX 7900 XT
(gfx1100) at pool shape (m=n=131072, k=4096), all three bit-identical:

| Kernel | LDS/WG | cand/s | vs GCN |
|---|---|---|---|
| `jackpot_search.cl` (GCN) | 9.1 KiB | ~117 000 | 1× |
| `jackpot_search_rdna3.cl` | 17.1 KiB | ~205 000 | ~1.75× |
| `jackpot_search_rdna3_wtile.cl` | 5.4 KiB | **~348 000** | **~3.0×** |

For reference the same workload runs at ~27 000 cand/s on the RX 570
(gfx803) the repo was first written for, so the 7900 XT lands ~13× ahead
end-to-end — well short of its ~10× raw-FP32 edge alone, because this PoW
is bound by sparse-permutation noise gathers, not arithmetic.

## Acknowledgements

- Pearl Research Labs for the reference implementation and the
  spec-quality test vectors.
- The Decred developers for the consensus / SPV building blocks Pearl
  builds on.

## License

[ISC](LICENSE), matching the upstream Pearl reference repository.
