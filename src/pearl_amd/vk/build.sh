#!/usr/bin/env bash
# Build the native Vulkan jackpot artifacts (jackpot_vk.dll + SPIR-V shaders)
# consumed by ../jackpot_vk.py. Requires the LunarG Vulkan SDK (glslc) and a
# C++ compiler (MinGW g++ tested). volk is fetched on first build.
#
# After this, `from pearl_amd.jackpot_vk import JackpotVk` works.
set -euo pipefail
cd "$(dirname "$0")"

: "${VULKAN_SDK:=/c/VulkanSDK/1.4.350.0}"
GLSLC="${GLSLC:-$VULKAN_SDK/Bin/glslc.exe}"
GXX="${GXX:-$HOME/scoop/apps/mingw/current/bin/g++.exe}"
INC="$VULKAN_SDK/Include"

if [ ! -f volk.h ]; then
  echo "fetching volk..."
  curl -L -sS -o volk.h https://raw.githubusercontent.com/zeux/volk/master/volk.h
  curl -L -sS -o volk.c https://raw.githubusercontent.com/zeux/volk/master/volk.c
fi

echo "compiling shaders (r=64,128 x ntiles=4,8,16 x reduce=0,1)..."
for r in 64 128; do for nt in 4 8 16; do for red in 0 1; do
  "$GLSLC" --target-env=vulkan1.3 -DPEARL_R=$r -DPEARL_NTILES_W=$nt -DREDUCE_MODE=$red \
           jackpot.comp -o "jackpot_r${r}_n${nt}_red${red}.spv"
done; done; done

echo "compiling jackpot_vk.dll (static CRT for ctypes)..."
"$GXX" -std=c++17 -O2 -shared -static -static-libgcc -static-libstdc++ \
       -I"$INC" -I. jackpot_vk.cpp volk.c -o jackpot_vk.dll
echo "done -> $(pwd)/jackpot_vk.dll"
