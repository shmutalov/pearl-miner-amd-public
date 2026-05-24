"""OpenCL device discovery and selection helpers."""
from __future__ import annotations

import pyopencl as cl


def list_devices() -> None:
    for i, platform in enumerate(cl.get_platforms()):
        print(f"Platform {i}: {platform.name}")
        print(f"  vendor:  {platform.vendor}")
        print(f"  version: {platform.version}")
        for j, device in enumerate(platform.get_devices()):
            dtype = cl.device_type.to_string(device.type)
            print(f"  Device {j} ({dtype}): {device.name}")
            print(f"    compute units:   {device.max_compute_units}")
            print(f"    max work group:  {device.max_work_group_size}")
            print(f"    global mem:      {device.global_mem_size / (1024**3):.2f} GiB")
            print(f"    local mem:       {device.local_mem_size / 1024:.0f} KiB")
            print(f"    OpenCL C:        {device.opencl_c_version}")
            ext = device.extensions.split()
            interesting = [
                e for e in ext
                if any(k in e.lower() for k in ("amd", "fp16", "media", "atomic", "dot", "int8"))
            ]
            if interesting:
                print(f"    interesting ext: {' '.join(interesting)}")


def find_gpu(prefer_vendor: str = "Advanced Micro Devices") -> cl.Device:
    """Return a GPU device, preferring the given vendor substring."""
    for platform in cl.get_platforms():
        if prefer_vendor.lower() in platform.vendor.lower():
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if devices:
                return devices[0]
    for platform in cl.get_platforms():
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if devices:
            return devices[0]
    raise RuntimeError("No OpenCL GPU device found")
