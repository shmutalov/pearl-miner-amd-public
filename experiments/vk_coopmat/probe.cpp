// Enumerate VK_KHR_cooperative_matrix configurations on the default device.
// We need to know whether int8(signed)xint8(signed)->int32 is supported, at
// what MxNxK tile, and in what scope (Subgroup expected on RDNA3). This decides
// the GEMM tiling for the amortized Pearl miner.
//
// Build:  g++ -std=c++17 -O2 -I"$VULKAN_SDK/Include" -I. probe.cpp volk.c -o probe.exe
// Run:    ./probe.exe
#define VK_NO_PROTOTYPES
#include "volk.h"
#include <cstdio>
#include <cstdint>
#include <vector>

static const char* comp_type(VkComponentTypeKHR t) {
    switch (t) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
        case VK_COMPONENT_TYPE_FLOAT64_KHR: return "f64";
        case VK_COMPONENT_TYPE_SINT8_KHR:   return "s8";
        case VK_COMPONENT_TYPE_SINT16_KHR:  return "s16";
        case VK_COMPONENT_TYPE_SINT32_KHR:  return "s32";
        case VK_COMPONENT_TYPE_SINT64_KHR:  return "s64";
        case VK_COMPONENT_TYPE_UINT8_KHR:   return "u8";
        case VK_COMPONENT_TYPE_UINT16_KHR:  return "u16";
        case VK_COMPONENT_TYPE_UINT32_KHR:  return "u32";
        case VK_COMPONENT_TYPE_UINT64_KHR:  return "u64";
        default: return "?";
    }
}
static const char* scope_name(VkScopeKHR s) {
    switch (s) {
        case VK_SCOPE_DEVICE_KHR:      return "Device";
        case VK_SCOPE_WORKGROUP_KHR:   return "Workgroup";
        case VK_SCOPE_SUBGROUP_KHR:    return "Subgroup";
        case VK_SCOPE_QUEUE_FAMILY_KHR:return "QueueFamily";
        default: return "?";
    }
}

int main() {
    if (volkInitialize() != VK_SUCCESS) { printf("volkInitialize failed\n"); return 1; }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) { printf("createInstance failed\n"); return 1; }
    volkLoadInstance(inst);

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(inst, &ndev, devs.data());

    for (uint32_t d = 0; d < ndev; ++d) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devs[d], &props);

        // subgroup size
        VkPhysicalDeviceSubgroupProperties sgp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &sgp;
        vkGetPhysicalDeviceProperties2(devs[d], &p2);

        printf("=== device %u: %s (subgroupSize=%u) ===\n", d, props.deviceName, sgp.subgroupSize);

        auto fpEnum = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
            vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        if (!fpEnum) { printf("  (no cooperative_matrix entrypoint)\n"); continue; }

        uint32_t n = 0;
        fpEnum(devs[d], &n, nullptr);
        std::vector<VkCooperativeMatrixPropertiesKHR> cfgs(n,
            {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
        fpEnum(devs[d], &n, cfgs.data());
        printf("  %u cooperative-matrix configs:\n", n);
        for (uint32_t i = 0; i < n; ++i) {
            auto& c = cfgs[i];
            printf("   [%2u] %ux%ux%u  A=%-3s B=%-3s C=%-3s R=%-3s  scope=%s sat=%d\n",
                   i, c.MSize, c.NSize, c.KSize,
                   comp_type(c.AType), comp_type(c.BType),
                   comp_type(c.CType), comp_type(c.ResultType),
                   scope_name(c.scope), (int)c.saturatingAccumulation);
        }
        // Highlight the int8->int32 ones we care about.
        printf("  --- int8(s8/u8) x int8 -> int32 candidates: ---\n");
        for (uint32_t i = 0; i < n; ++i) {
            auto& c = cfgs[i];
            bool a8 = (c.AType==VK_COMPONENT_TYPE_SINT8_KHR||c.AType==VK_COMPONENT_TYPE_UINT8_KHR);
            bool b8 = (c.BType==VK_COMPONENT_TYPE_SINT8_KHR||c.BType==VK_COMPONENT_TYPE_UINT8_KHR);
            bool c32 = (c.CType==VK_COMPONENT_TYPE_SINT32_KHR||c.CType==VK_COMPONENT_TYPE_UINT32_KHR);
            if (a8 && b8 && c32)
                printf("   *** [%2u] %ux%ux%u A=%s B=%s C=%s R=%s scope=%s\n",
                       i, c.MSize, c.NSize, c.KSize, comp_type(c.AType),
                       comp_type(c.BType), comp_type(c.CType), comp_type(c.ResultType),
                       scope_name(c.scope));
        }
    }
    return 0;
}
