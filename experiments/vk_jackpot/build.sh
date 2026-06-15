#!/usr/bin/env bash
# Build the Vulkan jackpot microbench: fetch volk, compile shaders + host.
# Requires: LunarG Vulkan SDK (glslc) and a C++ compiler (MinGW g++ tested).
set -euo pipefail
cd "$(dirname "$0")"

: "${VULKAN_SDK:=/c/VulkanSDK/1.4.350.0}"
GLSLC="${GLSLC:-$VULKAN_SDK/Bin/glslc.exe}"
GXX="${GXX:-$HOME/scoop/apps/mingw/current/bin/g++.exe}"
INC="$VULKAN_SDK/Include"

# Vendored volk (meta-loader) — fetched on first build.
if [ ! -f volk.h ]; then
  echo "fetching volk..."
  curl -L -sS -o volk.h https://raw.githubusercontent.com/zeux/volk/master/volk.h
  curl -L -sS -o volk.c https://raw.githubusercontent.com/zeux/volk/master/volk.c
fi

echo "compiling shaders..."
# vecadd smoke
"$GLSLC" vecadd.comp -o vecadd.spv
# jackpot: small (r=64) gate + pool (r=128); V0=LDS tree, V1=subgroupXor; NTILES sweep
for r in 64 128; do for nt in 4 8 16; do for red in 0 1; do
  "$GLSLC" --target-env=vulkan1.3 -DPEARL_R=$r -DPEARL_NTILES_W=$nt -DREDUCE_MODE=$red \
           jackpot.comp -o "jackpot_r${r}_n${nt}_red${red}.spv"
done; done; done

echo "compiling hosts..."
"$GXX" -std=c++17 -O2 -I"$INC" -I. smoke.cpp volk.c -o smoke.exe
"$GXX" -std=c++17 -O2 -I"$INC" -I. host.cpp  volk.c -o host.exe
echo "done."
