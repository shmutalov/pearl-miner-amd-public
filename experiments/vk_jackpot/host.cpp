// Vulkan host for the jackpot microbench. Loads a job dumped by dump_job.py,
// runs jackpot.comp, compares against ref.bin (bit-identical gate), and times
// the dispatch. Buffers are device-local (staging upload) so VRAM-bound reads
// are measured honestly.
//
// Usage: host <jackpot.spv> <job_dir> [--sgsize 0|32|64] [--reps N]
#include "volk.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>

#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    std::fprintf(stderr, "VK_CHECK failed %d at %s:%d\n", _r, __FILE__, __LINE__); std::exit(1);} } while(0)

static std::vector<uint8_t> readFile(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(1); }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> b(n);
    if (n && std::fread(b.data(), 1, n, f) != (size_t)n) { std::fprintf(stderr, "short read %s\n", p.c_str()); std::exit(1); }
    std::fclose(f); return b;
}

static VkPhysicalDevice g_pd;
static VkDevice g_dev;
static VkQueue g_queue;
static uint32_t g_qfi;
static VkCommandPool g_cp;

static uint32_t findMemType(uint32_t bits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(g_pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    std::fprintf(stderr, "no mem type\n"); std::exit(1);
}

struct Buffer { VkBuffer buf{}; VkDeviceMemory mem{}; VkDeviceSize size{}; };

static Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mp) {
    Buffer b; b.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size ? size : 4; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(g_dev, &bi, nullptr, &b.buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_dev, b.buf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size; ai.memoryTypeIndex = findMemType(mr.memoryTypeBits, mp);
    VK_CHECK(vkAllocateMemory(g_dev, &ai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(g_dev, b.buf, b.mem, 0));
    return b;
}

static void runOnce(VkCommandBuffer cb) {
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(g_queue));
}

// Device-local buffer initialised from host data via a staging copy.
static Buffer uploadDeviceLocal(const void* data, VkDeviceSize size) {
    Buffer dst = createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buffer stg = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* m; VK_CHECK(vkMapMemory(g_dev, stg.mem, 0, size ? size : 4, 0, &m));
    if (size) std::memcpy(m, data, size);
    vkUnmapMemory(g_dev, stg.mem);

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = g_cp; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(g_dev, &cbai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    VkBufferCopy bc{0, 0, size ? size : 4}; vkCmdCopyBuffer(cb, stg.buf, dst.buf, 1, &bc);
    VK_CHECK(vkEndCommandBuffer(cb));
    runOnce(cb);
    vkFreeCommandBuffers(g_dev, g_cp, 1, &cb);
    vkDestroyBuffer(g_dev, stg.buf, nullptr); vkFreeMemory(g_dev, stg.mem, nullptr);
    return dst;
}

static Buffer uploadFile(const std::string& path) {
    auto d = readFile(path); return uploadDeviceLocal(d.data(), d.size());
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: host <spv> <job_dir> [--sgsize N] [--reps N]\n"); return 2; }
    std::string spv = argv[1], dir = argv[2];
    uint32_t sgsize = 0; int reps = 20;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--sgsize") && i+1 < argc) sgsize = (uint32_t)std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--reps") && i+1 < argc) reps = std::atoi(argv[++i]);
    }
    auto meta = readFile(dir + "/meta.txt");
    int m,n,k,r,h,w,batch,nca,ncb;
    if (std::sscanf((const char*)meta.data(), "%d %d %d %d %d %d %d %d %d",
                    &m,&n,&k,&r,&h,&w,&batch,&nca,&ncb) != 9) { std::fprintf(stderr,"bad meta\n"); return 1; }
    uint32_t WG = (uint32_t)(h * w);

    VK_CHECK(volkInitialize());
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    VkInstance inst; VK_CHECK(vkCreateInstance(&ici, nullptr, &inst)); volkLoadInstance(inst);
    uint32_t nd = 0; vkEnumeratePhysicalDevices(inst, &nd, nullptr);
    std::vector<VkPhysicalDevice> pds(nd); vkEnumeratePhysicalDevices(inst, &nd, pds.data());
    g_pd = pds[0];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(g_pd, &props);

    VkPhysicalDeviceSubgroupSizeControlProperties sscp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; p2.pNext = &sscp;
    vkGetPhysicalDeviceProperties2(g_pd, &p2);

    uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(g_pd,&qn,nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qn); vkGetPhysicalDeviceQueueFamilyProperties(g_pd,&qn,qfs.data());
    g_qfi=~0u; for(uint32_t i=0;i<qn;++i) if(qfs[i].queueFlags&VK_QUEUE_COMPUTE_BIT){g_qfi=i;break;}

    // Chain the features we need.
    VkPhysicalDeviceSubgroupSizeControlFeatures sscf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    sscf.subgroupSizeControl = VK_TRUE; sscf.computeFullSubgroups = VK_TRUE;
    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR wmf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR};
    wmf.workgroupMemoryExplicitLayout = VK_TRUE; wmf.workgroupMemoryExplicitLayout8BitAccess = VK_TRUE;
    wmf.workgroupMemoryExplicitLayoutScalarBlockLayout = VK_TRUE; wmf.pNext = &sscf;
    VkPhysicalDevice8BitStorageFeatures s8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    s8.storageBuffer8BitAccess = VK_TRUE; s8.pNext = &wmf;
    VkPhysicalDeviceShaderFloat16Int8Features fi8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    fi8.shaderInt8 = VK_TRUE; fi8.pNext = &s8;

    const char* exts[] = {
        VK_KHR_8BIT_STORAGE_EXTENSION_NAME, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
    };
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex=g_qfi; qci.queueCount=1; qci.pQueuePriorities=&prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext=&fi8; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
    dci.enabledExtensionCount=4; dci.ppEnabledExtensionNames=exts;
    VK_CHECK(vkCreateDevice(g_pd,&dci,nullptr,&g_dev)); volkLoadDevice(g_dev);
    vkGetDeviceQueue(g_dev,g_qfi,0,&g_queue);
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex=g_qfi; cpci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(g_dev,&cpci,nullptr,&g_cp));

    // Upload all inputs (bindings 0..10) + output (11).
    Buffer bufs[12];
    bufs[0]=uploadFile(dir+"/A.bin");          bufs[1]=uploadFile(dir+"/B.bin");
    bufs[2]=uploadFile(dir+"/e_al.bin");       bufs[3]=uploadFile(dir+"/e_br_t.bin");
    bufs[4]=uploadFile(dir+"/e_ar_t.bin");     bufs[5]=uploadFile(dir+"/e_bl.bin");
    bufs[6]=uploadFile(dir+"/t_rows.bin");     bufs[7]=uploadFile(dir+"/t_cols.bin");
    bufs[8]=uploadFile(dir+"/row_pattern.bin");bufs[9]=uploadFile(dir+"/col_pattern.bin");
    bufs[10]=uploadFile(dir+"/key.bin");
    VkDeviceSize outSize = (VkDeviceSize)batch * 8 * 4;
    bufs[11]=createBuffer(outSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buffer outStg = createBuffer(outSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Descriptor set layout: 12 storage buffers.
    VkDescriptorSetLayoutBinding binds[12]{};
    for (int i=0;i<12;++i){binds[i].binding=i;binds[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount=1;binds[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount=12; dslci.pBindings=binds;
    VkDescriptorSetLayout dsl; VK_CHECK(vkCreateDescriptorSetLayout(g_dev,&dslci,nullptr,&dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,3*sizeof(int32_t)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount=1; plci.pSetLayouts=&dsl; plci.pushConstantRangeCount=1; plci.pPushConstantRanges=&pcr;
    VkPipelineLayout pl; VK_CHECK(vkCreatePipelineLayout(g_dev,&plci,nullptr,&pl));

    auto code=readFile(spv);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize=code.size(); smci.pCode=(const uint32_t*)code.data();
    VkShaderModule sm; VK_CHECK(vkCreateShaderModule(g_dev,&smci,nullptr,&sm));
    VkPipelineShaderStageCreateInfo ssci{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ssci.stage=VK_SHADER_STAGE_COMPUTE_BIT; ssci.module=sm; ssci.pName="main";
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    if (sgsize) { rss.requiredSubgroupSize = sgsize; ssci.pNext = &rss; }
    VkComputePipelineCreateInfo cpci2{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci2.stage=ssci; cpci2.layout=pl;
    VkPipeline pipe; VK_CHECK(vkCreateComputePipelines(g_dev,VK_NULL_HANDLE,1,&cpci2,nullptr,&pipe));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,12};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets=1; dpci.poolSizeCount=1; dpci.pPoolSizes=&ps;
    VkDescriptorPool dp; VK_CHECK(vkCreateDescriptorPool(g_dev,&dpci,nullptr,&dp));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool=dp; dsai.descriptorSetCount=1; dsai.pSetLayouts=&dsl;
    VkDescriptorSet ds; VK_CHECK(vkAllocateDescriptorSets(g_dev,&dsai,&ds));
    VkDescriptorBufferInfo dbi[12]; VkWriteDescriptorSet wr[12]{};
    for(int i=0;i<12;++i){dbi[i]={bufs[i].buf,0,VK_WHOLE_SIZE};
        wr[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[i].dstSet=ds; wr[i].dstBinding=i;
        wr[i].descriptorCount=1; wr[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo=&dbi[i];}
    vkUpdateDescriptorSets(g_dev,12,wr,0,nullptr);

    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType=VK_QUERY_TYPE_TIMESTAMP; qpci.queryCount=2;
    VkQueryPool qp; VK_CHECK(vkCreateQueryPool(g_dev,&qpci,nullptr,&qp));
    double tsPeriod = props.limits.timestampPeriod;

    int32_t pcvals[3] = {nca, ncb, k};
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool=g_cp; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(g_dev,&cbai,&cb));

    auto record = [&](bool copyOut){
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cb,&bi));
        vkCmdResetQueryPool(cb,qp,0,2);
        vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
        vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,nullptr);
        vkCmdPushConstants(cb,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pcvals),pcvals);
        vkCmdWriteTimestamp(cb,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,qp,0);
        vkCmdDispatch(cb,(uint32_t)batch,1,1);
        vkCmdWriteTimestamp(cb,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,qp,1);
        if (copyOut) {
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&mb,0,nullptr,0,nullptr);
            VkBufferCopy bc{0,0,outSize}; vkCmdCopyBuffer(cb,bufs[11].buf,outStg.buf,1,&bc);
        }
        VK_CHECK(vkEndCommandBuffer(cb));
    };

    // Warm-up + correctness run (with readback).
    record(true); runOnce(cb);
    std::vector<uint8_t> got(outSize); void* om;
    VK_CHECK(vkMapMemory(g_dev,outStg.mem,0,outSize,0,&om)); std::memcpy(got.data(),om,outSize); vkUnmapMemory(g_dev,outStg.mem);
    { FILE* f=std::fopen((dir+"/out.bin").c_str(),"wb"); if(f){std::fwrite(got.data(),1,got.size(),f);std::fclose(f);} }
    auto ref = readFile(dir+"/ref.bin");
    uint32_t bad=0; int firstBad=-1;
    for (VkDeviceSize i=0;i<outSize;++i) if (got[i]!=ref[i]) { if(bad<3){ if(firstBad<0)firstBad=(int)(i/32);} bad++; }
    std::printf("device: %s  reqSubgroupSize=%u (dev min=%u max=%u)\n",
                props.deviceName, sgsize, sscp.minSubgroupSize, sscp.maxSubgroupSize);
    std::printf("shape m=%d n=%d k=%d r=%d h=%d w=%d batch=%d\n", m,n,k,r,h,w,batch);
    std::printf("bit-identical: %s  (mismatched bytes=%u%s)\n",
                bad?"FAIL":"OK", bad, firstBad>=0?(", first cand "+std::to_string(firstBad)).c_str():"");

    // Timing: best-of-reps GPU time (no readback in the timed body).
    record(false);
    double best = 1e30;
    for (int i=0;i<reps;++i){
        runOnce(cb);
        uint64_t ts[2]; VK_CHECK(vkGetQueryPoolResults(g_dev,qp,0,2,sizeof(ts),ts,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT));
        double ms=(ts[1]-ts[0])*tsPeriod/1e6; best=std::min(best,ms);
    }
    std::printf("best dispatch: %.3f ms  =>  %.0f cand/s\n", best, batch/(best/1000.0));
    return bad?1:0;
}
