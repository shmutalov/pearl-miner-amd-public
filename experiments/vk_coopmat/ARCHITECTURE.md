# Amortized-GEMM + cooperative_matrix Pearl miner (RDNA3)

Branch `amortized-gemm-wmma`. Goal: replace the per-candidate noisy-GEMM kernel
(`src/pearl_amd/vk/jackpot.comp`, ~0.9M cand/s) with the algorithm real miners
use — precompute the noised matrices once, run the int8 GEMM on tensor cores
amortized across candidates, and finish each candidate with a cheap
gather+XOR+BLAKE3.

## The PoW, restated for amortization

Per job (one (A,B) commitment), pool shape:
`m = n = 131072, k = 4096, rank R = 128, rows_pattern=[0,32] (h=2),
cols_pattern=[0..63] (w=64)`, n_iters = k/R = **32** r-slices.

A **candidate** is `(t_r, t_c)`:
- rows used = `{t_r, t_r+32}` for valid `t_r` (t_r % 64 < 32) → 65536 row-pairs
- cols used = `{t_c .. t_c+63}` for `t_c` a multiple of 64 → 2048 col-blocks
- → **134,217,728 candidates / job**.

Hash:
```
tile_acc[iter] = XOR_{u in pair, v in block} ( sum_{l in slice iter} PA[row_u][l] * PB[col_v][l] )
jackpot_msg[iter % 16] = rotl13(jackpot_msg[iter%16]) ^ tile_acc[iter]     (folded over 32 iters)
hash = BLAKE3_compress(job_key, jackpot_msg)     ; accept if LE-uint256(hash) < target
```
where (proven from `jackpot.comp:150/163`, candidate-independent):
```
PA[row][l] = A[row][l] + e_al[row][ear[l].x] - e_al[row][ear[l].y]      (m x k, int8)
PB[col][l] = B[col][l] + e_br[col][ebl[l].x] - e_br[col][ebl[l].y]      (n x k, int8)
```

## Where the speedup comes from (and where it does NOT)

- **No flop reduction.** Each (row,col) cell is touched by exactly one candidate
  (every row in one pair, every col in one block), so the GEMM has *no cell reuse*:
  full G = m·n·k = **7.04e13 MACs/job**, same as the naive total.
- **The win is two constant factors:**
  1. **PA/PB computed once** (m×k + n×k int8 = 2×536 MB, fits 20 GB VRAM). The
     current kernel rebuilds PA[row][:] in all 2048 col-blocks and PB[col][:] in
     all 65536 pairs — that redundant noise-add dominates its runtime.
  2. **GEMM on WMMA** (coopmat `s8×s8→s32` 16×16×16, confirmed config [9]) instead
     of scalar `v_dot4`. RDNA3 int8 WMMA peak ≈ 200 TOPS.

Honest ceiling: GEMM-bound. 7.04e13 MACs/job ÷ ~50 TMAC/s sustained ≈ **1.4 s/job**
→ ~95M candidates/s, i.e. **~60–115× the current 0.9M/s**. (The pool table's
"35 TH/s" for a 7900 XT is a tensor-throughput unit, *not* candidates/s — our
~1e14 MAC/s would land in that regime; it is not 35e12 candidates/s.)

## Fused kernel design (no G round-trip through VRAM)

Materializing all of G is 9 PB; round-tripping it is ~18 PB of VRAM traffic
(hours). So G is **never stored** — each workgroup computes its G-tile in LDS and
finishes its candidates immediately.

**One workgroup = one (row-band of 64 rows) × one (col-block of 64 cols).**
That unit contains exactly 32 candidates (32 row-pairs `(64g+j, 64g+j+32)`,
j=0..31, all sharing the col-block). 2048 bands × 2048 blocks = 4.19M workgroups.

Per workgroup:
```
persistent LDS: jackpot_msg[32 candidates][16]            (2 KB)
for iter in 0..31:                                        # r-slices
    load PA_strip = PA[band 64 rows][k slice 128]  (64x128 s8, 8 KB LDS)
    load PB_strip = PB[block 64 cols][k slice 128] (64x128 s8, 8 KB LDS)
    G = PA_strip @ PB_strip^T  via coopmat 16x16x16        (64x64 s32, 16 KB LDS)
        # 4x4 output tiles x 8 K-steps = 128 wmma ops
    barrier
    for cand j in 0..31:                                   # row-pair (j, j+32)
        acc = XOR_{v=0..63} ( G[j][v] ^ G[j+32][v] )
        jackpot_msg[j][iter%16] = rotl13(jackpot_msg[j][iter%16]) ^ acc
    barrier
for cand j in 0..31:
    h = BLAKE3_compress(job_key, jackpot_msg[j])
    if LE(h) < target: record hit (t_r, t_c)
```
LDS ≈ 34 KB (< 64 KB). 4.19M WGs × 128 wmma = 1.7e10 wmma-16³ = 7.04e13 MACs ✓.

## Correctness strategy (bit-identical, validated per phase)

- **Phase A — PA/PB precompute.** New kernel builds PA (m×k), PB (n×k) int8.
  Validate: for sampled (row, slice), PA == the `pa[]` the current jackpot.comp
  builds for a candidate touching that row. CPU oracle from `pearl_noise` /
  `jackpot.py`.
- **Phase B — coopmat GEMM.** Compute one G-tile; assert
  `G[u][v] == sum_l PA[row][l]*PB[col][l]` vs numpy int32 (wrap) per slice.
- **Phase C — finish.** gather+XOR+rotl+BLAKE3 from a CPU-supplied G; assert
  `hash == jackpot.comp` output for the same candidate.
- **Phase D — fuse + search + wire.** Whole-kernel hash must match the existing
  `JackpotVk`/CPU oracle candidate-for-candidate, then benchmark cand/s and wire
  behind a `use_coopmat` flag with fallback to the packed kernel.

## Device facts (RX 7900 XT, this box)

- `VK_KHR_cooperative_matrix` rev 2, compute stage, `cooperativeMatrix=true`.
- Config [9]: **A=s8 B=s8 C=s32 R=s32, 16×16×16, scope=Subgroup, sat=0** (exact
  two's-complement wrap — matches our int32 XOR). Also u8/mixed variants [3..8].
  f16→f32 [0] available too.
- subgroupSize = 64 (wave64). coopmat is subgroup-scoped → one subgroup owns a
  16×16 tile. (Can pin wave32 via VK_EXT_subgroup_size_control if profiling wants.)
- `shaderIntegerDotProduct` 8-bit signed + 4×8 packed = accelerated (the
  current packed kernel's `v_dot4` fallback path).
- Probe: `experiments/vk_coopmat/probe.cpp` (`build.sh` to rebuild).
