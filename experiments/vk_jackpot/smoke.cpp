// Vector-add smoke test for the Vulkan compute host harness.
// Validates: volk init, physical-device pick (gfx1100), logical device + queue,
// host-visible SSBOs, descriptor set, compute pipeline from SPIR-V, dispatch,
// GPU timestamp query, and readback. If this prints "OK", the jackpot host can
// reuse the same patterns.
#include "volk.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    std::fprintf(stderr, "VK_CHECK failed %d at %s:%d\n", _r, __FILE__, __LINE__); std::exit(1);} } while(0)

static std::vector<char> readFile(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> buf(n);
    if (std::fread(buf.data(), 1, n, f) != (size_t)n) { std::fprintf(stderr, "short read\n"); std::exit(1); }
    std::fclose(f);
    return buf;
}

static uint32_t findMemType(VkPhysicalDevice pd, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    std::fprintf(stderr, "no suitable memory type\n"); std::exit(1);
}

struct Buf { VkBuffer buf{}; VkDeviceMemory mem{}; void* mapped{}; VkDeviceSize size{}; };

static Buf makeHostBuffer(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize size) {
    Buf b; b.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev, &bi, nullptr, &b.buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b.buf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = findMemType(pd, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
    VK_CHECK(vkMapMemory(dev, b.mem, 0, size, 0, &b.mapped));
    return b;
}

int main(int argc, char** argv) {
    const char* spv = argc > 1 ? argv[1] : "vecadd.spv";
    const uint32_t N = 1u << 20; // 1M elements

    VK_CHECK(volkInitialize());
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst{}; VK_CHECK(vkCreateInstance(&ici, nullptr, &inst));
    volkLoadInstance(inst);

    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst, &n, pds.data());
    VkPhysicalDevice pd = pds[0];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    std::printf("device: %s\n", props.deviceName);

    // Find a compute queue family.
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qn); vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qfs.data());
    uint32_t qfi = ~0u;
    for (uint32_t i = 0; i < qn; ++i) if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    if (qfi == ~0u) { std::fprintf(stderr, "no compute queue\n"); return 1; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice dev{}; VK_CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
    volkLoadDevice(dev);
    VkQueue queue; vkGetDeviceQueue(dev, qfi, 0, &queue);

    // Buffers
    Buf A = makeHostBuffer(pd, dev, (VkDeviceSize)N * 4);
    Buf B = makeHostBuffer(pd, dev, (VkDeviceSize)N * 4);
    Buf C = makeHostBuffer(pd, dev, (VkDeviceSize)N * 4);
    float* a = (float*)A.mapped; float* b = (float*)B.mapped;
    for (uint32_t i = 0; i < N; ++i) { a[i] = (float)i; b[i] = 2.0f * (float)i; }

    // Descriptor set layout: 3 storage buffers
    VkDescriptorSetLayoutBinding binds[3]{};
    for (int i = 0; i < 3; ++i) { binds[i].binding = i; binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1; binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 3; dslci.pBindings = binds;
    VkDescriptorSetLayout dsl; VK_CHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; VK_CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

    auto code = readFile(spv);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size(); smci.pCode = (const uint32_t*)code.data();
    VkShaderModule sm; VK_CHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));

    VkPipelineShaderStageCreateInfo ssci{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT; ssci.module = sm; ssci.pName = "main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = ssci; cpci.layout = pl;
    VkPipeline pipe; VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

    // Descriptor pool + set
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
    VkDescriptorPool dp; VK_CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet ds; VK_CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
    VkDescriptorBufferInfo bi3[3] = {{A.buf,0,A.size},{B.buf,0,B.size},{C.buf,0,C.size}};
    VkWriteDescriptorSet w[3]{};
    for (int i = 0; i < 3; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = ds;
        w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi3[i]; }
    vkUpdateDescriptorSets(dev, 3, w, 0, nullptr);

    // Timestamp query pool
    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP; qpci.queryCount = 2;
    VkQueryPool qp; VK_CHECK(vkCreateQueryPool(dev, &qpci, nullptr, &qp));
    double tsPeriod = props.limits.timestampPeriod; // ns per tick

    // Command buffer
    VkCommandPoolCreateInfo cpci2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci2.queueFamilyIndex = qfi;
    VkCommandPool cp; VK_CHECK(vkCreateCommandPool(dev, &cpci2, nullptr, &cp));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cp; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));

    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK_CHECK(vkBeginCommandBuffer(cb, &cbbi));
    vkCmdResetQueryPool(cb, qp, 0, 2);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &N);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
    vkCmdDispatch(cb, (N + 255) / 256, 1, 1);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
    VK_CHECK(vkEndCommandBuffer(cb));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));

    uint64_t ts[2]{};
    VK_CHECK(vkGetQueryPoolResults(dev, qp, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    double gpu_ms = (ts[1] - ts[0]) * tsPeriod / 1e6;

    // Verify
    float* c = (float*)C.mapped; uint32_t bad = 0;
    for (uint32_t i = 0; i < N; ++i) if (c[i] != a[i] + b[i]) { if (bad < 4) std::printf("  mismatch[%u] %f != %f\n", i, c[i], a[i]+b[i]); bad++; }
    std::printf("vecadd N=%u  gpu=%.3f ms  mismatches=%u  => %s\n", N, gpu_ms, bad, bad ? "FAIL" : "OK");
    return bad ? 1 : 0;
}
