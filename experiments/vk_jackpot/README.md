# Vulkan jackpot microbench

A research spike that ports the jackpot evaluator to **Vulkan compute (GLSL)**
to answer one question the OpenCL build can't: *does moving the WG-wide XOR
reduce off LDS and onto subgroup ops help?* The AMD Windows OpenCL driver does
not expose `cl_khr_subgroups` arithmetic (`subgroupXor`) or shuffle; the Vulkan
driver does. This isolates that one lever.

## Result (RX 7900 XT / gfx1100, pool shape m=n=131072, k=4096)

All variants verified **bit-identical** to the OpenCL/CPU reference.

| w-tiles | V0: LDS-tree reduce | V1: `subgroupXor` (wave32) | V1 gain |
|---|---|---|---|
| 4  | 167k cand/s | 202k | +21% |
| 8  | 218k cand/s | 276k | +27% |
| 16 | 266k cand/s | 281k | +6%  |

**`subgroupXor` beats the LDS-tree reduce at every tiling** — the reduce was on
the critical path, and Vulkan's subgroup arithmetic (unreachable from this
OpenCL driver) is a genuine win for this latency-bound kernel.

For reference the production OpenCL `jackpot_search_rdna3_wtile.cl` does ~338k
cand/s. The Vulkan port trails it **only because of the int8-shared caveat
below**, not the reduce — see caveats.

## What's here

- `jackpot.comp` — GLSL port of `jackpot_search_rdna3_wtile.cl`. Compile-time
  macros: `PEARL_R`, `PEARL_NTILES_W`, `REDUCE_MODE` (0 = LDS tree, 1 =
  `subgroupXor`).
- `host.cpp` — Vulkan host (volk). Device-local SSBOs via staging, optional
  `requiredSubgroupSize` (wave32 pin), timestamp timing, readback + bit-identical
  compare against `ref.bin`.
- `dump_job.py` — generates kernel inputs + reference hashes (from the validated
  OpenCL kernel) into a job dir.
- `smoke.cpp` / `vecadd.comp` — host-harness smoke test.
- `verify_msg.py` — compares the shader's pre-hash `jackpot_msg` to the CPU
  oracle (used during bring-up).
- `build.sh`, `run.sh` — build everything / dump + run the A/B sweep.

## Build & run

```bash
# needs LunarG Vulkan SDK (glslc) + MinGW g++; volk is fetched on first build
./build.sh
./run.sh pool 16384     # dumps a job via the Python/OpenCL reference, runs the sweep
./run.sh small 256      # fast correctness gate
```

## Caveats

- **int shared, not int8.** The strips live in 32-bit `shared int` arrays.
  int8 *shared* needs `VK_KHR_workgroup_memory_explicit_layout` blocks, which
  miscompiled here (member aliasing — `tile_acc` writes landed in
  `jackpot_msg`), so we fell back to int and recovered occupancy by tiling
  harder (`NTILES_W` up to 16). This 4x-larger LDS is the whole reason the
  Vulkan port (281k) trails the OpenCL int8 kernel (338k); it is **not** the
  reduce. Closing the gap = compact LDS via manual `4×int8→uint` packing in
  plain shared (TODO).
- **V0 baseline is pessimistic**: it uses a full `barrier()` between every tree
  step, more than the OpenCL kernel's lockstep-optimized reduce. The honest
  takeaway is the *V1-over-V0 delta*, and that V1 reaches a reduce the OpenCL
  driver can't.

## Next step: RGP

The remaining question — *is the noise gather LDS-bank-conflict bound?* (which
would justify a `subgroupShuffle` gather rewrite) — needs Radeon GPU Profiler:

1. Launch **Radeon Developer Panel**, enable profiling.
2. Run `host.exe <spv> <job_dir> --reps 200` so the dispatch repeats long
   enough to capture.
3. Capture an RGP profile; read **LDS bank-conflict %**, wave occupancy, and the
   VALU/LDS/VMEM instruction mix.
   - High bank-conflict % on the pa/pb build → green-light a `subgroupShuffle`
     gather (V3) and likely a full Vulkan port.
   - Low → we're near the form's ceiling; keep the OpenCL kernel in production.
