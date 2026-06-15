#!/usr/bin/env bash
# Dump a job (via the OpenCL/Python reference) then run the Vulkan A/B sweep.
# Usage: run.sh [small|pool] [batch]
set -euo pipefail
cd "$(dirname "$0")"
SHAPE="${1:-pool}"; BATCH="${2:-16384}"
PY="${PY:-../../.venv/Scripts/python.exe}"
JOB="${TMP:-/d/tmp}/vkjob_${SHAPE}"
R=$([ "$SHAPE" = small ] && echo 64 || echo 128)

echo "== dumping $SHAPE job (batch=$BATCH) =="
"$PY" dump_job.py "$JOB" --shape "$SHAPE" --batch "$BATCH" | tail -2

echo; echo "== V0 (LDS tree reduce, default subgroup) =="
for nt in 4 8 16; do printf "NTILES=%-2d " $nt
  ./host.exe "jackpot_r${R}_n${nt}_red0.spv" "$JOB" --sgsize 0 --reps 15 | grep -E "bit-identical|best"|tr '\n' ' '; echo; done

echo; echo "== V1 (subgroupXor reduce, wave32) =="
for nt in 4 8 16; do printf "NTILES=%-2d " $nt
  ./host.exe "jackpot_r${R}_n${nt}_red1.spv" "$JOB" --sgsize 32 --reps 15 | grep -E "bit-identical|best"|tr '\n' ' '; echo; done
