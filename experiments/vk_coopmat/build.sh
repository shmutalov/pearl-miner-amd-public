#!/usr/bin/env bash
# Build the cooperative_matrix capability probe.
set -euo pipefail
cd "$(dirname "$0")"
: "${VULKAN_SDK:=/c/VulkanSDK/1.4.350.0}"
GXX="${GXX:-$HOME/scoop/apps/mingw/current/bin/g++.exe}"
if [ ! -f volk.h ]; then
  curl -L -sS -o volk.h https://raw.githubusercontent.com/zeux/volk/master/volk.h
  curl -L -sS -o volk.c https://raw.githubusercontent.com/zeux/volk/master/volk.c
fi
"$GXX" -std=c++17 -O2 -I"$VULKAN_SDK/Include" -I. probe.cpp volk.c -o probe.exe
echo "done -> $(pwd)/probe.exe"
