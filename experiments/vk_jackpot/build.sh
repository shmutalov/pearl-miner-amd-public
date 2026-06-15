#!/usr/bin/env bash
# Build the research/benchmark harness (smoke.exe + host.exe microbench). The
# production library + kernel now live in ../../src/pearl_amd/vk/ (build it with
# that dir's build.sh); the jackpot shaders here are compiled from the canonical
# kernel source so the standalone microbench host can load them.
set -euo pipefail
cd "$(dirname "$0")"

: "${VULKAN_SDK:=/c/VulkanSDK/1.4.350.0}"
GLSLC="${GLSLC:-$VULKAN_SDK/Bin/glslc.exe}"
GXX="${GXX:-$HOME/scoop/apps/mingw/current/bin/g++.exe}"
INC="$VULKAN_SDK/Include"
KERNEL="../../src/pearl_amd/vk/jackpot.comp"   # canonical kernel source

if [ ! -f volk.h ]; then
  echo "fetching volk..."
  curl -L -sS -o volk.h https://raw.githubusercontent.com/zeux/volk/master/volk.h
  curl -L -sS -o volk.c https://raw.githubusercontent.com/zeux/volk/master/volk.c
fi

echo "compiling shaders (from $KERNEL)..."
"$GLSLC" vecadd.comp -o vecadd.spv
for r in 64 128; do for nt in 4 8 16; do for red in 0 1; do
  "$GLSLC" --target-env=vulkan1.3 -DPEARL_R=$r -DPEARL_NTILES_W=$nt -DREDUCE_MODE=$red \
           "$KERNEL" -o "jackpot_r${r}_n${nt}_red${red}.spv"
done; done; done

echo "compiling microbench hosts..."
"$GXX" -std=c++17 -O2 -I"$INC" -I. smoke.cpp volk.c -o smoke.exe
"$GXX" -std=c++17 -O2 -I"$INC" -I. host.cpp  volk.c -o host.exe
echo "done. (production lib: build ../../src/pearl_amd/vk/build.sh)"
