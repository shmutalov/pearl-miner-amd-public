# pearl-miner-amd

Pearl PoUW miner for AMD GPUs. Started on **Polaris** (RX 470/480/570/580,
gfx803) via OpenCL — the cheapest 8 GiB cards nobody wants — and now runs the
hot path on **RDNA3** (RX 7900 XT, gfx1100) tensor cores via Vulkan
`cooperative_matrix`. Just-for-fun research project.

> **Status: it earns.** The full pipeline runs against real pool stratum
> (`eu1.alphapool.tech:5566`) — handshake, BLAKE3 challenge, mining params,
> per-job preflight, GPU candidate search, PlainProof assembly, share submit —
> and **submits shares the pool accepts** (validated live: 40/40 accepted,
> continuously). The amortized-GEMM + cooperative_matrix evaluator hits
> **~37–38 M candidates/s on an RX 7900 XT** (~37 TH/s by the protocol's
> `attempts × 2³² / s` metric). Every stage is mathematically correct and
> bit-identical to the reference Rust implementation.
>
> Getting there required two correctness fixes that the small-shape tests had
> masked (both now fixed + verified at the live pool's `rank=128`):
> - **perm `noise_rank`** used `NOISE_RANGE//2` (=64) instead of `r` → wrong
>   noise (= invalid proofs) whenever `rank ≠ 64`.
> - **share target** is `2²⁵⁶/D = nbits_target << 32`, *not* the raw `nbits`
>   target. Pearl credits each attempt as `H_per_attempt = 2³²` hash-equivalents,
>   so a share is accepted when `2²⁵⁶/hash ≥ D`. Searching at the raw target was
>   2³² too strict and found ~nothing; the scaled target makes expected
>   candidates/share = `D` (≈ a few ms of GPU time).
>
> Economics are still modest — one 7900 XT is a small contributor, roughly
> break-even on power — but it produces *accepted* shares, which is the point.

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
    ├── jackpot_gpu.py      # GPU batched evaluator + search (OpenCL)
    ├── jackpot_vk.py       # Vulkan per-candidate evaluator (--vulkan)
    ├── jackpot_coopmat.py  # tensor-core amortized-GEMM evaluator (--coopmat)
    ├── vk/                 # Vulkan/GLSL kernels + C-ABI DLLs (build.sh)
    │   ├── jackpot.comp / jackpot_vk.cpp          # per-candidate Vulkan path
    │   ├── pmat.comp                              # builds global PA/PB
    │   ├── jackpot_coopmat.comp / *_vk.cpp        # cooperative_matrix path
    │   └── build.sh                               # glslc + MinGW g++
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
    --observe-seconds 90 \
    --max-attempts-per-job 100000
```

On an RX 7900 XT, build the Vulkan bits once and mine on tensor cores —
this submits accepted shares continuously (omit `d` in the password to let
the pool's VarDiff pick the difficulty):

```bash
bash src/pearl_amd/vk/build.sh        # one-time: glslc + MinGW g++

.venv/Scripts/python.exe src/scripts/35_run_miner_live.py \
    --address <your-prl1...-wallet> --worker rx7900xt \
    --coopmat --submit --coopmat-batch 64
```

Add `--submit` to post shares (default is dry-run). `--coopmat` routes the
search through the tensor-core evaluator (falls back to `--vulkan`, then
OpenCL, then CPU if unavailable).

## Limitations / what's NOT here

- **No share verifier**: we build `PlainProof` bytes but don't have a
  local Merkle+jackpot verifier. The pool does that; locally we only
  check that our `PlainProof` round-trips byte-identically through the
  bincode codec.
- **VarDiff**: we honor whatever difficulty the pool assigns (omit `d` in the
  password) and search against the correctly-scaled share target, but don't
  otherwise adjust strategy based on it.
- **No multi-GPU**: single device only.
- **Coopmat is pattern-specialized**: the tensor-core kernel hardcodes the live
  pool's `rows=[0,32]`/`cols=[0..63]` tile geometry; other patterns fall back to
  the Vulkan/OpenCL evaluators.
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

## Vulkan + cooperative_matrix (tensor cores) — the fast path

The OpenCL kernels recompute the noisy GEMM per candidate. The AMD Windows
OpenCL driver exposes neither subgroup arithmetic nor `cl_khr_integer_dot_product`,
so the real win — RDNA3's WMMA units — is unreachable from OpenCL. Vulkan
compute reaches them.

Two Vulkan stages live in `src/pearl_amd/vk/` (built by `build.sh`, artifacts
gitignored), each a ctypes drop-in selected by a miner flag:

- **`jackpot_vk` (`--vulkan`)** — a straight Vulkan port of the per-candidate
  kernel. Packing 4×int8 per `shared uint` lets LLPC fuse the inner byte-dot
  (likely `v_dot4_i32_i8`): **~0.9 M cand/s**, ~2.6× the OpenCL kernel.

- **`jackpot_coopmat` (`--coopmat`)** — the algorithm real miners use. The noise
  depends only on the row (for A) or column (for B), so `PA = A+noise_a` (m×k)
  and `PB = B+noise_b` (n×k) are **global per job**. Build them once, then the
  per-r-slice product `G = PA·PBᵀ` is a batched int8 GEMM run on tensor cores
  (`VK_KHR_cooperative_matrix`, `s8×s8→s32` 16×16×16), with each candidate a
  cheap gather + XOR-reduce + BLAKE3. One workgroup = one (64-row band)×(64-col
  block) = 32 candidates; `G` never leaves LDS. Tuned to **WG=256 + wave32**:

  | Path | cand/s (RX 7900 XT, pool shape) | vs OpenCL wtile |
  |---|---|---|
  | `jackpot_search_rdna3_wtile.cl` (OpenCL) | ~348 000 | 1× |
  | `jackpot_vk` (Vulkan, `--vulkan`) | ~0.9 M | ~2.6× |
  | **`jackpot_coopmat` (`--coopmat`)** | **~37–38 M** | **~110×** |

  Specialized to the live pool pattern (`rows=[0,32]` h=2, `cols=[0..63]` w=64);
  other patterns fall back to Vulkan/OpenCL automatically. By the protocol's
  display metric (`attempts × 2³² / s / 1e12`), 37 M cand/s ≈ **37 TH/s**.

The whole chain is validated bit-identical to `evaluate_candidate` at the pool
shape, and end-to-end against the live pool (accepted shares). Build:

```bash
bash src/pearl_amd/vk/build.sh    # needs LunarG Vulkan SDK (glslc) + MinGW g++
```

Research harness, oracle, and per-phase validators live in
`experiments/vk_coopmat/` (`amortized_oracle.py`, `live_share_probe.py`,
`live_mine_continuous.py`, `ARCHITECTURE.md`).

## Acknowledgements

- Pearl Research Labs for the reference implementation and the
  spec-quality test vectors.
- The Decred developers for the consensus / SPV building blocks Pearl
  builds on.

## Donations

If this project was useful to you, PRL donations are welcome (wallet created at
[compute.pearlresearch.ai/wallet](https://compute.pearlresearch.ai/wallet)):

```
prl1p5vtjsxajasd805qtc2xp5zp3tl99egklxzfr0th7m0v8ue858uvs7hrhhs
```

## License

[ISC](LICENSE), matching the upstream Pearl reference repository.
