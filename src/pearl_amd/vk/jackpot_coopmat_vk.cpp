// libjackpot_coopmat_vk: amortized-GEMM + cooperative_matrix Pearl miner as a C
// ABI for ctypes. Builds the global noised matrices PA (m x k) / PB (n x k)
// device-local once per job (pmat.spv), then searches the (band x block) grid
// on tensor cores (jackpot_coopmat.spv), comparing each candidate's hash
// (LE-uint256) to the target and recording hits.
//
// Specialized to the live pool pattern: rows_pattern=[0,32] (h=2), cols_pattern
// =[0..63] (w=64). One workgroup = one (64-row band) x (64-col block) = 32
// candidates (row-pairs (j, j+32), j=0..31). Grid = (m/64) x (n/64) workgroups,
// which enumerates exactly the valid (t_rows,t_cols) set. The host tiles the
// search dispatch (TDR-safe) and aggregates hits.
//
// Two layers of host/GPU overlap:
//   * NSLOT search ping-pong slots: submit one tile while draining the other's
//     hits (proof build + pool submit) -> host work overlaps the GPU search.
//   * NJOB double-buffered job slots (PA/PB/KEY): build the next round's matrices
//     into the idle job slot on a prefetch thread while the current round still
//     searches the active slot -> the per-round preflight rebuild overlaps too.
// Cross-thread safety: setup (uploads + pmat) runs from its own command pool with
// a setup fence (never vkQueueWaitIdle, which would stall the search), and every
// vkQueueSubmit is serialized by a spinlock since a VkQueue is externally synced.
//
// Build (MinGW): g++ -std=c++17 -O2 -shared -static -static-libgcc -static-libstdc++ \
//                -I<sdk>/Include -I. jackpot_coopmat_vk.cpp volk.c -o jackpot_coopmat_vk.dll
#include "volk.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <atomic>
#include <vector>

#define CHECK(x)  do{ VkResult _r=(x); if(_r!=VK_SUCCESS){ std::fprintf(stderr,"[jcm] VK %d at %s:%d\n",_r,__FILE__,__LINE__); return -1; } }while(0)
#define CHECKP(x) do{ VkResult _r=(x); if(_r!=VK_SUCCESS){ std::fprintf(stderr,"[jcm] VK %d at %s:%d\n",_r,__FILE__,__LINE__); return nullptr; } }while(0)

namespace {

struct Buf { VkBuffer buf{}; VkDeviceMemory mem{}; void* map{}; VkDeviceSize size{}; };

// Spinlock guarding vkQueueSubmit (the queue is externally synchronized and the
// setup/prefetch thread submits concurrently with the search/main thread).
struct SpinLock {
    std::atomic<bool> f{false};
    void lock()   { bool e=false; while(!f.compare_exchange_weak(e,true,std::memory_order_acquire)) e=false; }
    void unlock() { f.store(false,std::memory_order_release); }
};

static const int NSLOT = 2;   // search ping-pong slots (per-tile)
static const int NJOB  = 2;   // double-buffered job slots (PA/PB/KEY, per-round)

struct Ctx {
    VkInstance inst{}; VkPhysicalDevice pd{}; VkDevice dev{}; VkQueue queue{}; uint32_t qfi{};
    VkCommandPool cp{};        // search command pool (main thread)
    VkCommandPool setupCp{};   // setup command pool (prefetch thread: uploads + pmat)
    VkCommandBuffer cb{};      // setup command buffer (from setupCp)
    double tsPeriod{};
    SpinLock qlock;            // serializes vkQueueSubmit across threads
    VkFence setupFence{};      // setup (upload/pmat) completion
    VkPipeline pmatPipe{}, searchPipe{};
    VkPipelineLayout pmatPl{}, searchPl{};
    VkDescriptorSetLayout pmatDsl{}, searchDsl{};
    VkShaderModule pmatSm{}, searchSm{};
    VkDescriptorPool dpool{};
    VkDescriptorSet pmatDsA{}, pmatDsB{};
    // Per-slot search resources (ping-pong).
    VkCommandBuffer scb[NSLOT]{};
    VkDescriptorSet searchDs[NSLOT][NJOB]{};   // [search slot][job slot]
    VkFence fence[NSLOT]{};
    VkQueryPool qpool[NSLOT]{};
    Buf dTGT[NSLOT], dHIT[NSLOT];
    int k{}, r{}, m{}, n{}, nbands{}, nblocks{}, maxHits{};
    // Shared transient pmat inputs (rebuilt each set_job; never read by search).
    Buf dA,dB,dEAL,dEBR,dEAR,dEBL;
    // Double-buffered per-job outputs + key (read by search).
    Buf dPA[NJOB], dPB[NJOB], dKEY[NJOB];
    bool jobSet[NJOB]{};
};

uint32_t findMem(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags p){
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
    for(uint32_t i=0;i<mp.memoryTypeCount;++i) if((bits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&p)==p) return i;
    return ~0u;
}
int makeBuf(Ctx* c, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mp, Buf& o){
    o.size=size; VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size=size?size:4; bi.usage=usage;
    CHECK(vkCreateBuffer(c->dev,&bi,nullptr,&o.buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(c->dev,o.buf,&mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=mr.size; ai.memoryTypeIndex=findMem(c->pd,mr.memoryTypeBits,mp);
    CHECK(vkAllocateMemory(c->dev,&ai,nullptr,&o.mem)); CHECK(vkBindBufferMemory(c->dev,o.buf,o.mem,0));
    if(mp&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) CHECK(vkMapMemory(c->dev,o.mem,0,size?size:4,0,&o.map));
    return 0;
}
void destroyBuf(Ctx* c, Buf& b){ if(b.buf)vkDestroyBuffer(c->dev,b.buf,nullptr); if(b.mem)vkFreeMemory(c->dev,b.mem,nullptr); b=Buf{}; }

// Submit cb under the queue spinlock (held only for the submit call itself).
VkResult qsubmit(Ctx* c, VkCommandBuffer cb, VkFence fence){
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb;
    c->qlock.lock(); VkResult r=vkQueueSubmit(c->queue,1,&si,fence); c->qlock.unlock(); return r;
}
// Setup-path submit+wait: fence-based (NOT vkQueueWaitIdle, which would also wait
// on in-flight searches on the other thread). Only the brief submit is serialized.
int submitWaitFence(Ctx* c, VkCommandBuffer cb){
    CHECK(vkResetFences(c->dev,1,&c->setupFence));
    CHECK(qsubmit(c,cb,c->setupFence));
    CHECK(vkWaitForFences(c->dev,1,&c->setupFence,VK_TRUE,UINT64_MAX));
    return 0;
}
int uploadDL(Ctx* c, const void* data, VkDeviceSize size, Buf& dst){
    if(makeBuf(c,size,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,dst)) return -1;
    Buf stg; if(makeBuf(c,size,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,stg)) return -1;
    std::memcpy(stg.map,data,size);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK(vkResetCommandBuffer(c->cb,0)); CHECK(vkBeginCommandBuffer(c->cb,&bi));
    VkBufferCopy bc{0,0,size}; vkCmdCopyBuffer(c->cb,stg.buf,dst.buf,1,&bc); CHECK(vkEndCommandBuffer(c->cb));
    if(submitWaitFence(c,c->cb)) return -1;
    destroyBuf(c,stg); return 0;
}
std::vector<char> readFile(const char* p){ FILE* f=std::fopen(p,"rb"); if(!f) return {};
    std::fseek(f,0,SEEK_END); long n=std::ftell(f); std::fseek(f,0,SEEK_SET); std::vector<char> b(n); if(n){size_t rd=std::fread(b.data(),1,n,f);(void)rd;} std::fclose(f); return b; }

int mkPipe(Ctx* c, const char* spv, VkPipelineLayout pl, uint32_t sgSize, VkShaderModule& sm, VkPipeline& pipe){
    auto code=readFile(spv); if(code.empty()){ std::fprintf(stderr,"[jcm] cannot read %s\n",spv); return -1; }
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize=code.size(); smci.pCode=(const uint32_t*)code.data();
    CHECK(vkCreateShaderModule(c->dev,&smci,nullptr,&sm));
    VkPipelineShaderStageCreateInfo ss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; ss.stage=VK_SHADER_STAGE_COMPUTE_BIT; ss.module=sm; ss.pName="main";
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    if(sgSize){ rss.requiredSubgroupSize=sgSize; ss.pNext=&rss; }
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cp.stage=ss; cp.layout=pl;
    CHECK(vkCreateComputePipelines(c->dev,VK_NULL_HANDLE,1,&cp,nullptr,&pipe)); return 0;
}

VkDescriptorSetLayout mkDsl(Ctx* c, int n){
    std::vector<VkDescriptorSetLayoutBinding> b(n);
    for(int i=0;i<n;++i){b[i]={}; b[i].binding=i; b[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b[i].descriptorCount=1; b[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dl.bindingCount=n; dl.pBindings=b.data();
    VkDescriptorSetLayout o; if(vkCreateDescriptorSetLayout(c->dev,&dl,nullptr,&o)!=VK_SUCCESS) return VK_NULL_HANDLE; return o;
}
void writeSet(Ctx* c, VkDescriptorSet s, std::vector<VkBuffer> bufs){
    std::vector<VkDescriptorBufferInfo> bi(bufs.size()); std::vector<VkWriteDescriptorSet> w(bufs.size());
    for(size_t i=0;i<bufs.size();++i){ bi[i]={bufs[i],0,VK_WHOLE_SIZE}; w[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[i].dstSet=s; w[i].dstBinding=(uint32_t)i; w[i].descriptorCount=1; w[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo=&bi[i]; }
    vkUpdateDescriptorSets(c->dev,(uint32_t)w.size(),w.data(),0,nullptr);
}
void freeInputs(Ctx* c){
    destroyBuf(c,c->dA);destroyBuf(c,c->dB);destroyBuf(c,c->dEAL);
    destroyBuf(c,c->dEBR);destroyBuf(c,c->dEAR);destroyBuf(c,c->dEBL);
}
void freeSlot(Ctx* c, int J){
    destroyBuf(c,c->dPA[J]);destroyBuf(c,c->dPB[J]);destroyBuf(c,c->dKEY[J]); c->jobSet[J]=false;
}

} // namespace

extern "C" {

__declspec(dllexport) Ctx* jcm_create(const char* pmat_spv, const char* search_spv,
                                      int k, int r, int subgroup_size, int max_hits){
    if(volkInitialize()!=VK_SUCCESS){ std::fprintf(stderr,"[jcm] no vulkan-1\n"); return nullptr; }
    Ctx* c=new Ctx(); c->k=k; c->r=r; c->maxHits=max_hits>0?max_hits:4096;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&app;
    CHECKP(vkCreateInstance(&ici,nullptr,&c->inst)); volkLoadInstance(c->inst);
    uint32_t nd=0; vkEnumeratePhysicalDevices(c->inst,&nd,nullptr); if(!nd){std::fprintf(stderr,"[jcm] no device\n");return nullptr;}
    std::vector<VkPhysicalDevice> pds(nd); vkEnumeratePhysicalDevices(c->inst,&nd,pds.data()); c->pd=pds[0];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(c->pd,&props); c->tsPeriod=props.limits.timestampPeriod;
    uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(c->pd,&qn,nullptr); std::vector<VkQueueFamilyProperties> qfs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(c->pd,&qn,qfs.data()); c->qfi=~0u;
    for(uint32_t i=0;i<qn;++i) if(qfs[i].queueFlags&VK_QUEUE_COMPUTE_BIT){c->qfi=i;break;}

    // The coopmat SPIR-V declares OpCapability VulkanMemoryModel / OpMemoryModel
    // Logical Vulkan (from GL_KHR_memory_scope_semantics + cooperative_matrix), so
    // the device must enable the vulkanMemoryModel feature (core since Vulkan 1.2;
    // device-scope variant is NOT declared, so it is not needed).
    VkPhysicalDeviceVulkanMemoryModelFeatures vmm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES}; vmm.vulkanMemoryModel=VK_TRUE;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR}; cmf.cooperativeMatrix=VK_TRUE; cmf.pNext=&vmm;
    VkPhysicalDevice8BitStorageFeatures s8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES}; s8.storageBuffer8BitAccess=VK_TRUE; s8.pNext=&cmf;
    VkPhysicalDeviceShaderFloat16Int8Features fi8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES}; fi8.shaderInt8=VK_TRUE; fi8.pNext=&s8;
    VkPhysicalDeviceSubgroupSizeControlFeatures sscf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES}; sscf.subgroupSizeControl=VK_TRUE; sscf.computeFullSubgroups=VK_TRUE; sscf.pNext=&fi8;
    const char* exts[]={VK_KHR_8BIT_STORAGE_EXTENSION_NAME,VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME};
    float prio=1.0f; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex=c->qfi; qci.queueCount=1; qci.pQueuePriorities=&prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.pNext=&sscf; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci; dci.enabledExtensionCount=4; dci.ppEnabledExtensionNames=exts;
    CHECKP(vkCreateDevice(c->pd,&dci,nullptr,&c->dev)); volkLoadDevice(c->dev); vkGetDeviceQueue(c->dev,c->qfi,0,&c->queue);
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci.queueFamilyIndex=c->qfi; cpci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CHECKP(vkCreateCommandPool(c->dev,&cpci,nullptr,&c->cp));        // search (main thread)
    CHECKP(vkCreateCommandPool(c->dev,&cpci,nullptr,&c->setupCp));   // setup (prefetch thread)
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cbai.commandPool=c->setupCp; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    CHECKP(vkAllocateCommandBuffers(c->dev,&cbai,&c->cb));           // setup cb (separate pool)
    cbai.commandPool=c->cp; cbai.commandBufferCount=NSLOT; CHECKP(vkAllocateCommandBuffers(c->dev,&cbai,c->scb)); // per-slot search
    VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qp.queryType=VK_QUERY_TYPE_TIMESTAMP; qp.queryCount=2;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; fci.flags=VK_FENCE_CREATE_SIGNALED_BIT;
    CHECKP(vkCreateFence(c->dev,&fci,nullptr,&c->setupFence));
    for(int s=0;s<NSLOT;++s){ CHECKP(vkCreateQueryPool(c->dev,&qp,nullptr,&c->qpool[s])); CHECKP(vkCreateFence(c->dev,&fci,nullptr,&c->fence[s])); }

    c->pmatDsl=mkDsl(c,4); c->searchDsl=mkDsl(c,5);
    if(!c->pmatDsl||!c->searchDsl) return nullptr;
    VkPushConstantRange pcr1{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(int32_t)};
    VkPipelineLayoutCreateInfo pl1{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pl1.setLayoutCount=1; pl1.pSetLayouts=&c->pmatDsl; pl1.pushConstantRangeCount=1; pl1.pPushConstantRanges=&pcr1;
    CHECKP(vkCreatePipelineLayout(c->dev,&pl1,nullptr,&c->pmatPl));
    VkPushConstantRange pcr2{VK_SHADER_STAGE_COMPUTE_BIT,0,4*sizeof(int32_t)};
    VkPipelineLayoutCreateInfo pl2{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pl2.setLayoutCount=1; pl2.pSetLayouts=&c->searchDsl; pl2.pushConstantRangeCount=1; pl2.pPushConstantRanges=&pcr2;
    CHECKP(vkCreatePipelineLayout(c->dev,&pl2,nullptr,&c->searchPl));
    if(mkPipe(c,pmat_spv,c->pmatPl,0,c->pmatSm,c->pmatPipe)) return nullptr;
    if(mkPipe(c,search_spv,c->searchPl,(uint32_t)subgroup_size,c->searchSm,c->searchPipe)) return nullptr;

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,(uint32_t)(4+4+5*NSLOT*NJOB)};
    VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dp.maxSets=2+NSLOT*NJOB; dp.poolSizeCount=1; dp.pPoolSizes=&ps;
    CHECKP(vkCreateDescriptorPool(c->dev,&dp,nullptr,&c->dpool));
    auto alloc=[&](VkDescriptorSetLayout l)->VkDescriptorSet{ VkDescriptorSetAllocateInfo a{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; a.descriptorPool=c->dpool; a.descriptorSetCount=1; a.pSetLayouts=&l; VkDescriptorSet s; vkAllocateDescriptorSets(c->dev,&a,&s); return s; };
    c->pmatDsA=alloc(c->pmatDsl); c->pmatDsB=alloc(c->pmatDsl);
    for(int s=0;s<NSLOT;++s) for(int j=0;j<NJOB;++j) c->searchDs[s][j]=alloc(c->searchDsl);

    auto HV=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for(int s=0;s<NSLOT;++s){
        if(makeBuf(c,32,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,HV,c->dTGT[s])) return nullptr;
        if(makeBuf(c,(VkDeviceSize)(1+c->maxHits*10)*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,HV,c->dHIT[s])) return nullptr;
    }
    return c;
}

// Build PA/PB/KEY for one job slot J. key = a_noise_seed as 8 LE u32. Safe to run
// on a prefetch thread concurrently with searches reading a *different* job slot:
// uploads/pmat use the setup command pool + setup fence (no vkQueueWaitIdle), and
// every submit is serialized by the queue spinlock.
__declspec(dllexport) int jcm_set_job_slot(Ctx* c, int J,
    const int8_t* A, const int8_t* B, const int8_t* e_al, const int8_t* e_br_t,
    const uint32_t* e_ar_t, const uint32_t* e_bl, const uint32_t* key, int m, int n){
    if(J<0||J>=NJOB) return -3;
    // CONTRACT: the caller must not rebuild a job slot while a search reads it.
    // set_job_slot(J) only touches slot-J buffers + the shared (search-invisible)
    // pmat inputs, so it is safe to run on a prefetch thread concurrently with a
    // search reading a *different* slot. With NJOB=2 double-buffering, a slot is
    // reused only two rounds after its last search, which has long since drained.
    freeInputs(c); freeSlot(c,J);
    c->m=m; c->n=n; c->nbands=m/64; c->nblocks=n/64;
    int k=c->k, r=c->r;
    if(uploadDL(c,A,(VkDeviceSize)m*k,c->dA)) return -1;
    if(uploadDL(c,B,(VkDeviceSize)n*k,c->dB)) return -1;
    if(uploadDL(c,e_al,(VkDeviceSize)m*r,c->dEAL)) return -1;
    if(uploadDL(c,e_br_t,(VkDeviceSize)n*r,c->dEBR)) return -1;
    if(uploadDL(c,e_ar_t,(VkDeviceSize)k*2*4,c->dEAR)) return -1;
    if(uploadDL(c,e_bl,(VkDeviceSize)k*2*4,c->dEBL)) return -1;
    if(uploadDL(c,key,32,c->dKEY[J])) return -1;
    if(makeBuf(c,(VkDeviceSize)m*k,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,c->dPA[J])) return -1;
    if(makeBuf(c,(VkDeviceSize)n*k,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,c->dPB[J])) return -1;
    writeSet(c,c->pmatDsA,{c->dA.buf,c->dEAL.buf,c->dEAR.buf,c->dPA[J].buf});
    writeSet(c,c->pmatDsB,{c->dB.buf,c->dEBR.buf,c->dEBL.buf,c->dPB[J].buf});
    for(int s=0;s<NSLOT;++s)
        writeSet(c,c->searchDs[s][J],{c->dPA[J].buf,c->dPB[J].buf,c->dKEY[J].buf,c->dTGT[s].buf,c->dHIT[s].buf});

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK(vkResetCommandBuffer(c->cb,0)); CHECK(vkBeginCommandBuffer(c->cb,&bi));
    vkCmdBindPipeline(c->cb,VK_PIPELINE_BIND_POINT_COMPUTE,c->pmatPipe);
    int32_t kk=k;
    vkCmdBindDescriptorSets(c->cb,VK_PIPELINE_BIND_POINT_COMPUTE,c->pmatPl,0,1,&c->pmatDsA,0,nullptr);
    vkCmdPushConstants(c->cb,c->pmatPl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(kk),&kk);
    vkCmdDispatch(c->cb,(uint32_t)m,1,1);
    vkCmdBindDescriptorSets(c->cb,VK_PIPELINE_BIND_POINT_COMPUTE,c->pmatPl,0,1,&c->pmatDsB,0,nullptr);
    vkCmdDispatch(c->cb,(uint32_t)n,1,1);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(c->cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,nullptr,0,nullptr);
    CHECK(vkEndCommandBuffer(c->cb));
    if(submitWaitFence(c,c->cb)) return -1;
    c->jobSet[J]=true;
    return 0;
}

// Back-compat: single-job callers build (and search) job slot 0.
__declspec(dllexport) int jcm_set_job(Ctx* c,
    const int8_t* A, const int8_t* B, const int8_t* e_al, const int8_t* e_br_t,
    const uint32_t* e_ar_t, const uint32_t* e_bl, const uint32_t* key, int m, int n){
    return jcm_set_job_slot(c,0,A,B,e_al,e_br_t,e_ar_t,e_bl,key,m,n);
}

// Record + submit one search slot's tile over WGs [wg_off, wg_off+wg_cnt), reading
// job slot `jslot`, with the search slot's own fence; returns immediately (no
// idle-wait). The caller MUST jcm_search_collect(slot) before re-submitting the
// same search slot. Fences start signaled so the leading wait is a no-op first use.
__declspec(dllexport) int jcm_search_submit(Ctx* c, int slot, int jslot,
    const uint8_t* target_le, int64_t wg_off, int wg_cnt){
    if(slot<0||slot>=NSLOT||jslot<0||jslot>=NJOB) return -3;
    if(!c->jobSet[jslot]) return -2;
    CHECK(vkWaitForFences(c->dev,1,&c->fence[slot],VK_TRUE,UINT64_MAX));
    CHECK(vkResetFences(c->dev,1,&c->fence[slot]));
    std::memcpy(c->dTGT[slot].map, target_le, 32);
    ((uint32_t*)c->dHIT[slot].map)[0]=0u;
    int32_t pcv[4]={c->nblocks, c->k, (int32_t)wg_off, c->maxHits};
    VkCommandBuffer cb=c->scb[slot];
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK(vkResetCommandBuffer(cb,0)); CHECK(vkBeginCommandBuffer(cb,&bi));
    vkCmdResetQueryPool(cb,c->qpool[slot],0,2);
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,c->searchPipe);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,c->searchPl,0,1,&c->searchDs[slot][jslot],0,nullptr);
    vkCmdPushConstants(cb,c->searchPl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pcv),pcv);
    vkCmdWriteTimestamp(cb,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,c->qpool[slot],0);
    vkCmdDispatch(cb,(uint32_t)wg_cnt,1,1);
    vkCmdWriteTimestamp(cb,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,c->qpool[slot],1);
    CHECK(vkEndCommandBuffer(cb));
    CHECK(qsubmit(c,cb,c->fence[slot]));
    return 0;
}

// Wait for slot's submission, then read up to maxHits records of 10 u32
// {t_r, t_c, hash[8]} into out_records; *out_count = total hits in the tile
// (may exceed maxHits). Leaves the fence signaled for the next submit.
__declspec(dllexport) int jcm_search_collect(Ctx* c, int slot,
    uint32_t* out_records, int* out_count){
    if(slot<0||slot>=NSLOT) return -3;
    CHECK(vkWaitForFences(c->dev,1,&c->fence[slot],VK_TRUE,UINT64_MAX));
    uint32_t* hit=(uint32_t*)c->dHIT[slot].map; uint32_t cnt=hit[0];
    *out_count=(int)cnt;
    uint32_t saved=cnt<(uint32_t)c->maxHits?cnt:(uint32_t)c->maxHits;
    if(saved) std::memcpy(out_records, hit+1, (size_t)saved*10*4);
    return 0;
}

// Back-compat synchronous tile = submit(search slot 0, job slot 0) + collect(0).
__declspec(dllexport) int jcm_search_tile(Ctx* c, const uint8_t* target_le,
    int64_t wg_off, int wg_cnt, uint32_t* out_records, int* out_count){
    int rc=jcm_search_submit(c,0,0,target_le,wg_off,wg_cnt); if(rc) return rc;
    return jcm_search_collect(c,0,out_records,out_count);
}

__declspec(dllexport) double jcm_last_gpu_ms(Ctx* c, int slot){
    if(slot<0||slot>=NSLOT) return -1;
    uint64_t ts[2]; if(vkGetQueryPoolResults(c->dev,c->qpool[slot],0,2,sizeof(ts),ts,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT)!=VK_SUCCESS) return -1;
    return (ts[1]-ts[0])*c->tsPeriod/1e6;
}

__declspec(dllexport) void jcm_destroy(Ctx* c){
    if(!c) return;
    if(c->dev){ vkDeviceWaitIdle(c->dev);
        freeInputs(c); for(int j=0;j<NJOB;++j) freeSlot(c,j);
        for(int s=0;s<NSLOT;++s){
            destroyBuf(c,c->dTGT[s]); destroyBuf(c,c->dHIT[s]);
            if(c->fence[s])vkDestroyFence(c->dev,c->fence[s],nullptr);
            if(c->qpool[s])vkDestroyQueryPool(c->dev,c->qpool[s],nullptr);
        }
        if(c->setupFence)vkDestroyFence(c->dev,c->setupFence,nullptr);
        if(c->pmatPipe)vkDestroyPipeline(c->dev,c->pmatPipe,nullptr);
        if(c->searchPipe)vkDestroyPipeline(c->dev,c->searchPipe,nullptr);
        if(c->pmatSm)vkDestroyShaderModule(c->dev,c->pmatSm,nullptr);
        if(c->searchSm)vkDestroyShaderModule(c->dev,c->searchSm,nullptr);
        if(c->pmatPl)vkDestroyPipelineLayout(c->dev,c->pmatPl,nullptr);
        if(c->searchPl)vkDestroyPipelineLayout(c->dev,c->searchPl,nullptr);
        if(c->dpool)vkDestroyDescriptorPool(c->dev,c->dpool,nullptr);
        if(c->pmatDsl)vkDestroyDescriptorSetLayout(c->dev,c->pmatDsl,nullptr);
        if(c->searchDsl)vkDestroyDescriptorSetLayout(c->dev,c->searchDsl,nullptr);
        if(c->cp)vkDestroyCommandPool(c->dev,c->cp,nullptr);
        if(c->setupCp)vkDestroyCommandPool(c->dev,c->setupCp,nullptr);
        vkDestroyDevice(c->dev,nullptr);
    }
    if(c->inst)vkDestroyInstance(c->inst,nullptr);
    delete c;
}

} // extern "C"
