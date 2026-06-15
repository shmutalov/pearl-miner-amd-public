# Vulkan jackpot microbench

A research spike that ports the jackpot evaluator to **Vulkan compute (GLSL)**
to answer one question the OpenCL build can't: *does moving the WG-wide XOR
reduce off LDS and onto subgroup ops help?* The AMD Windows OpenCL driver does
not expose `cl_khr_subgroups` arithmetic (`subgroupXor`) or shuffle; the Vulkan
driver does. This isolates that one lever.

## Result (RX 7900 XT / gfx1100, pool shape m=n=131072, k=4096)

All variants verified **bit-identical** to the OpenCL/CPU reference. Throughput
is pure GPU time (Vulkan timestamps / OpenCL CL profiling events), batch=16384.

**Headline — packed int8 LDS, apples-to-apples GPU time:**

| Kernel | cand/s |
|---|---|
| OpenCL `jackpot_search_rdna3_wtile.cl` (int8, char4) | 382k |
| **Vulkan packed int8, NTILES=4** | **989k (~2.6×)** |

The win is the **inner loop**, not the reduce. Packing 4×int8 into each
`shared uint` and doing a 4-wide byte dot lets the Vulkan LLPC backend schedule
it far better than the AMD OpenCL compiler (very likely auto-emitting
`v_dot4_i32_i8` — the int8 hardware dot we thought was unreachable, reached via
the compiler rather than an extension). Still only ~0.5% of int8 peak, so the
kernel remains gather-bound; this is a scheduling/ISA win, not hitting compute.

**The `subgroupXor` lever depends on the inner loop being the bottleneck:**

| LDS strips | V0 LDS-tree | V1 `subgroupXor` |
|---|---|---|
| int (32-bit), NTILES=8 | 218k | 276k (**+27%**) |
| packed int8, NTILES=4  | 988k | 989k (**~0%**) |

When the inner loop is inefficient (int shared), the reduce is a big fraction
and `subgroupXor` helps a lot. Once the inner loop is fast (packed int8), the
reduce is no longer the bottleneck and the subgroup advantage vanishes. Honest
takeaway: do the int8 packing first; `subgroupXor` only matters if you don't.

## Productionized: `JackpotVk` (drop-in for `JackpotGpu`)

`jackpot_vk.cpp` builds to `jackpot_vk.dll` (a C ABI: create / set_job /
evaluate / destroy, with the device + pipeline + per-job buffers persistent and
batch buffers reused). `jackpot_vk.py` wraps it via ctypes as **`JackpotVk`**,
matching `JackpotGpu` (`set_job` derives noise via `PearlNoiseGpu` then uploads;
`set_job_raw` takes inputs directly; `evaluate_batch` → `(B,32)` hashes).
`validate_vk.py` checks it **bit-identical to `JackpotGpu`** and benches it
(987k cand/s through the Python/ctypes boundary — the DLL holds A/B/noise on the
device, so per-call overhead is just the tiny t_rows/t_cols upload + readback).
This is the piece that makes the 2.6x usable from the miner; wiring `search()` +
`miner.py` is the next phase.

```bash
python validate_vk.py pool 16384   # bit-identical gate + throughput
```

## What's here

- `jackpot.comp` — GLSL port of `jackpot_search_rdna3_wtile.cl`. Compile-time
  macros: `PEARL_R`, `PEARL_NTILES_W`, `REDUCE_MODE` (0 = LDS tree, 1 =
  `subgroupXor`).
- `jackpot_vk.cpp` / `jackpot_vk.py` / `validate_vk.py` — the DLL, ctypes
  wrapper, and bit-identical validation (above).
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

## Notes / caveats

- **int8 in shared = manual packing, not blocks.** int8 *shared* via
  `VK_KHR_workgroup_memory_explicit_layout` blocks miscompiled here (member
  aliasing — `tile_acc` writes landed in `jackpot_msg`). The working approach is
  4×int8 packed into each plain `shared uint`, with each thread writing a full
  uint (no read-modify-write race) and the inner product unpacking 4-at-a-time
  via `bitfieldExtract`. This both restores the compact LDS footprint *and* is
  what unlocks the LLPC inner-loop win above.
- **Fair timing.** Vulkan = GPU timestamps; OpenCL = CL profiling events
  (`measure_ocl_gpu.py`) — both pure kernel time, no host/alloc/readback. The
  ~338k you may see from `JackpotGpu.evaluate_batch` wall-time is burdened by
  per-batch buffer alloc + readback + Python; the kernel itself is 382k.
- This is a spike, not productionized: single fixed shape per `.spv`, no
  multi-GPU, host hardcodes device[0].

## Next step: RGP

Two open questions for the Radeon GPU Profiler:

1. **Confirm the inner-loop win's mechanism** — disassemble the packed shader's
   ISA and check whether LLPC emitted `v_dot4_i32_i8` for the byte-dot. If so,
   the same packing trick could be hand-forced in OpenCL (or at least explains
   the 2.6×).
2. **Is the pa/pb build LDS-bank-conflict bound?** If yes, a `subgroupShuffle`
   gather (keeping the noise rows in registers) is the next lever; if not, we're
   near the form's ceiling.

How: launch **Radeon Developer Panel**, run `host.exe <spv> <job_dir>
--reps 200`, capture, and read the ISA + **LDS bank-conflict %**, occupancy, and
VALU/LDS/VMEM mix.
