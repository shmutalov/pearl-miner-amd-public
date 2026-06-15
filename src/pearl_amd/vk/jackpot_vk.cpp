// libjackpot_vk: Vulkan jackpot evaluator exposed as a C ABI for ctypes.
// Persistent context: device + pipeline + per-job device buffers are created
// once; evaluate() reuses pre-allocated batch buffers and a command buffer so
// the search loop's repeated calls are cheap.
//
// Build (MinGW): g++ -std=c++17 -O2 -shared -I<sdk>/Include -I. \
//                jackpot_vk.cpp volk.c -o jackpot_vk.dll
#include "volk.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#define CHECK(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS){ \
    std::fprintf(stderr,"[jvk] VK fail %d at %s:%d\n",_r,__FILE__,__LINE__); return -1; } } while(0)
#define CHECKP(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS){ \
    std::fprintf(stderr,"[jvk] VK fail %d at %s:%d\n",_r,__FILE__,__LINE__); return nullptr; } } while(0)

namespace {

struct Buf { VkBuffer buf{}; VkDeviceMemory mem{}; VkDeviceSize size{}; };

struct Ctx {
    VkInstance inst{};
    VkPhysicalDevice pd{};
    VkDevice dev{};
    VkQueue queue{};
    uint32_t qfi{};
    VkCommandPool cp{};
    VkPipeline pipe{};
    VkPipelineLayout pl{};
    VkDescriptorSetLayout dsl{};
    VkDescriptorPool dpool{};
    VkDescriptorSet ds{};
    VkShaderModule sm{};
    VkCommandBuffer cb{};
    double tsPeriod{};
    VkQueryPool qpool{};
    int h{}, w{}, r{}, k{};
    // per-job buffers, bindings 0..10
    Buf job[11];
    bool jobSet{false};
    // reusable batch buffers (bindings 6,7 = t_rows,t_cols; 11 = out)
    Buf trDev, tcDev, outDev;       // device-local
    Buf trStg, tcStg, outStg;       // host-visible staging
    int cap{0};                     // current batch capacity
};

uint32_t findMem(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags p) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
    for (uint32_t i=0;i<mp.memoryTypeCount;++i)
        if ((bits&(1u<<i)) && (mp.memoryTypes[i].propertyFlags&p)==p) return i;
    return ~0u;
}
int makeBuf(Ctx* c, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mp, Buf& out) {
    out.size=size; VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size=size?size:4; bi.usage=usage; bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    CHECK(vkCreateBuffer(c->dev,&bi,nullptr,&out.buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(c->dev,out.buf,&mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize=mr.size; ai.memoryTypeIndex=findMem(c->pd,mr.memoryTypeBits,mp);
    CHECK(vkAllocateMemory(c->dev,&ai,nullptr,&out.mem));
    CHECK(vkBindBufferMemory(c->dev,out.buf,out.mem,0));
    return 0;
}
void destroyBuf(Ctx* c, Buf& b){ if(b.buf)vkDestroyBuffer(c->dev,b.buf,nullptr); if(b.mem)vkFreeMemory(c->dev,b.mem,nullptr); b=Buf{}; }

int submitWait(Ctx* c, VkCommandBuffer cb) {
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb;
    CHECK(vkQueueSubmit(c->queue,1,&si,VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(c->queue));
    return 0;
}
// One-shot device-local upload from host data.
int uploadDL(Ctx* c, const void* data, VkDeviceSize size, Buf& dst) {
    if (makeBuf(c,size,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,dst)) return -1;
    Buf stg; if(makeBuf(c,size,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,stg)) return -1;
    void* m; CHECK(vkMapMemory(c->dev,stg.mem,0,size?size:4,0,&m));
    if(size) std::memcpy(m,data,size); vkUnmapMemory(c->dev,stg.mem);
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool=c->cp; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    VkCommandBuffer cb; CHECK(vkAllocateCommandBuffers(c->dev,&cbai,&cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK(vkBeginCommandBuffer(cb,&bi)); VkBufferCopy bc{0,0,size?size:4}; vkCmdCopyBuffer(cb,stg.buf,dst.buf,1,&bc);
    CHECK(vkEndCommandBuffer(cb)); if(submitWait(c,cb)) return -1;
    vkFreeCommandBuffers(c->dev,c->cp,1,&cb); destroyBuf(c,stg);
    return 0;
}

std::vector<char> readFile(const char* p){ FILE* f=std::fopen(p,"rb"); if(!f) return {};
    std::fseek(f,0,SEEK_END); long n=std::ftell(f); std::fseek(f,0,SEEK_SET);
    std::vector<char> b(n); if(n) { size_t rd=std::fread(b.data(),1,n,f); (void)rd; } std::fclose(f); return b; }

} // namespace

extern "C" {

// spv_path = packed shader for the chosen (r, ntiles, reduce). subgroup_size 0=default.
__declspec(dllexport) Ctx* jvk_create(int h, int w, int r, const char* spv_path, int subgroup_size) {
    if (volkInitialize()!=VK_SUCCESS){ std::fprintf(stderr,"[jvk] no vulkan-1\n"); return nullptr; }
    Ctx* c = new Ctx(); c->h=h; c->w=w; c->r=r;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&app;
    CHECKP(vkCreateInstance(&ici,nullptr,&c->inst)); volkLoadInstance(c->inst);
    uint32_t nd=0; vkEnumeratePhysicalDevices(c->inst,&nd,nullptr); if(!nd){std::fprintf(stderr,"[jvk] no device\n");return nullptr;}
    std::vector<VkPhysicalDevice> pds(nd); vkEnumeratePhysicalDevices(c->inst,&nd,pds.data()); c->pd=pds[0];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(c->pd,&props); c->tsPeriod=props.limits.timestampPeriod;
    uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(c->pd,&qn,nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qn); vkGetPhysicalDeviceQueueFamilyProperties(c->pd,&qn,qfs.data());
    c->qfi=~0u; for(uint32_t i=0;i<qn;++i) if(qfs[i].queueFlags&VK_QUEUE_COMPUTE_BIT){c->qfi=i;break;}

    VkPhysicalDeviceSubgroupSizeControlFeatures sscf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    sscf.subgroupSizeControl=VK_TRUE; sscf.computeFullSubgroups=VK_TRUE;
    VkPhysicalDevice8BitStorageFeatures s8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    s8.storageBuffer8BitAccess=VK_TRUE; s8.pNext=&sscf;
    VkPhysicalDeviceShaderFloat16Int8Features fi8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    fi8.shaderInt8=VK_TRUE; fi8.pNext=&s8;
    const char* exts[]={VK_KHR_8BIT_STORAGE_EXTENSION_NAME,VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
                        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME};
    float prio=1.0f; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex=c->qfi; qci.queueCount=1; qci.pQueuePriorities=&prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext=&fi8; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci; dci.enabledExtensionCount=3; dci.ppEnabledExtensionNames=exts;
    CHECKP(vkCreateDevice(c->pd,&dci,nullptr,&c->dev)); volkLoadDevice(c->dev);
    vkGetDeviceQueue(c->dev,c->qfi,0,&c->queue);
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex=c->qfi; cpci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CHECKP(vkCreateCommandPool(c->dev,&cpci,nullptr,&c->cp));

    // 12 storage bindings + push constants (n_cols_A,n_cols_B,k)
    VkDescriptorSetLayoutBinding binds[12]{};
    for(int i=0;i<12;++i){binds[i].binding=i;binds[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount=1;binds[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dl.bindingCount=12; dl.pBindings=binds;
    CHECKP(vkCreateDescriptorSetLayout(c->dev,&dl,nullptr,&c->dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,3*sizeof(int32_t)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount=1; plci.pSetLayouts=&c->dsl; plci.pushConstantRangeCount=1; plci.pPushConstantRanges=&pcr;
    CHECKP(vkCreatePipelineLayout(c->dev,&plci,nullptr,&c->pl));
    auto code=readFile(spv_path); if(code.empty()){std::fprintf(stderr,"[jvk] cannot read %s\n",spv_path);return nullptr;}
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize=code.size(); smci.pCode=(const uint32_t*)code.data();
    CHECKP(vkCreateShaderModule(c->dev,&smci,nullptr,&c->sm));
    VkPipelineShaderStageCreateInfo ss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ss.stage=VK_SHADER_STAGE_COMPUTE_BIT; ss.module=c->sm; ss.pName="main";
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    if(subgroup_size){ rss.requiredSubgroupSize=(uint32_t)subgroup_size; ss.pNext=&rss; }
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cp.stage=ss; cp.layout=c->pl;
    CHECKP(vkCreateComputePipelines(c->dev,VK_NULL_HANDLE,1,&cp,nullptr,&c->pipe));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,12};
    VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dp.maxSets=1; dp.poolSizeCount=1; dp.pPoolSizes=&ps;
    CHECKP(vkCreateDescriptorPool(c->dev,&dp,nullptr,&c->dpool));
    VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; da.descriptorPool=c->dpool; da.descriptorSetCount=1; da.pSetLayouts=&c->dsl;
    CHECKP(vkAllocateDescriptorSets(c->dev,&da,&c->ds));
    VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qp.queryType=VK_QUERY_TYPE_TIMESTAMP; qp.queryCount=2;
    CHECKP(vkCreateQueryPool(c->dev,&qp,nullptr,&c->qpool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool=c->cp; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    CHECKP(vkAllocateCommandBuffers(c->dev,&cbai,&c->cb));
    return c;
}

__declspec(dllexport) int jvk_set_job(Ctx* c,
    const int8_t* A, const int8_t* B, const int8_t* e_al, const int8_t* e_br_t,
    const uint32_t* e_ar_t, const uint32_t* e_bl, const int32_t* row_pat, const int32_t* col_pat,
    const uint32_t* key, int m, int n, int k) {
    for (int i=0;i<11;++i) destroyBuf(c, c->job[i]);
    c->k=k;
    if (uploadDL(c,A,(VkDeviceSize)m*k,c->job[0])) return -1;
    if (uploadDL(c,B,(VkDeviceSize)n*k,c->job[1])) return -1;
    if (uploadDL(c,e_al,(VkDeviceSize)m*c->r,c->job[2])) return -1;
    if (uploadDL(c,e_br_t,(VkDeviceSize)n*c->r,c->job[3])) return -1;
    if (uploadDL(c,e_ar_t,(VkDeviceSize)k*2*4,c->job[4])) return -1;
    if (uploadDL(c,e_bl,(VkDeviceSize)k*2*4,c->job[5])) return -1;
    if (uploadDL(c,row_pat,(VkDeviceSize)c->h*4,c->job[8])) return -1;
    if (uploadDL(c,col_pat,(VkDeviceSize)c->w*4,c->job[9])) return -1;
    if (uploadDL(c,key,32,c->job[10])) return -1;
    c->jobSet=true;
    return 0;
}

static int ensureBatch(Ctx* c, int batch) {
    if (batch<=c->cap) return 0;
    destroyBuf(c,c->trDev);destroyBuf(c,c->tcDev);destroyBuf(c,c->outDev);
    destroyBuf(c,c->trStg);destroyBuf(c,c->tcStg);destroyBuf(c,c->outStg);
    VkDeviceSize idx=(VkDeviceSize)batch*4, out=(VkDeviceSize)batch*8*4;
    auto DL=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT; auto HV=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if(makeBuf(c,idx,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,DL,c->trDev))return -1;
    if(makeBuf(c,idx,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,DL,c->tcDev))return -1;
    if(makeBuf(c,out,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,DL,c->outDev))return -1;
    if(makeBuf(c,idx,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,HV,c->trStg))return -1;
    if(makeBuf(c,idx,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,HV,c->tcStg))return -1;
    if(makeBuf(c,out,VK_BUFFER_USAGE_TRANSFER_DST_BIT,HV,c->outStg))return -1;
    c->cap=batch;
    return 0;
}

// Dispatch `batch` candidates whose offsets are already staged in trStg/tcStg;
// hashes land in outStg. Caller fills trStg/tcStg and reads outStg.
static int runBatch(Ctx* c, int batch) {
    VkDeviceSize idx=(VkDeviceSize)batch*4, outsz=(VkDeviceSize)batch*8*4;
    Buf* b[12]={&c->job[0],&c->job[1],&c->job[2],&c->job[3],&c->job[4],&c->job[5],
                &c->trDev,&c->tcDev,&c->job[8],&c->job[9],&c->job[10],&c->outDev};
    VkDescriptorBufferInfo dbi[12]; VkWriteDescriptorSet wr[12]{};
    for(int i=0;i<12;++i){dbi[i]={b[i]->buf,0,VK_WHOLE_SIZE};
        wr[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[i].dstSet=c->ds; wr[i].dstBinding=i;
        wr[i].descriptorCount=1; wr[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo=&dbi[i];}
    vkUpdateDescriptorSets(c->dev,12,wr,0,nullptr);
    int32_t pc[3]={c->k,c->k,c->k};
    CHECK(vkResetCommandBuffer(c->cb,0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CHECK(vkBeginCommandBuffer(c->cb,&bi));
    VkBufferCopy bc{0,0,idx};
    vkCmdCopyBuffer(c->cb,c->trStg.buf,c->trDev.buf,1,&bc);
    vkCmdCopyBuffer(c->cb,c->tcStg.buf,c->tcDev.buf,1,&bc);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(c->cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,nullptr,0,nullptr);
    vkCmdResetQueryPool(c->cb,c->qpool,0,2);
    vkCmdBindPipeline(c->cb,VK_PIPELINE_BIND_POINT_COMPUTE,c->pipe);
    vkCmdBindDescriptorSets(c->cb,VK_PIPELINE_BIND_POINT_COMPUTE,c->pl,0,1,&c->ds,0,nullptr);
    vkCmdPushConstants(c->cb,c->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pc),pc);
    vkCmdWriteTimestamp(c->cb,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,c->qpool,0);
    vkCmdDispatch(c->cb,(uint32_t)batch,1,1);
    vkCmdWriteTimestamp(c->cb,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,c->qpool,1);
    VkMemoryBarrier mb2{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb2.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; mb2.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(c->cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&mb2,0,nullptr,0,nullptr);
    VkBufferCopy bo{0,0,outsz}; vkCmdCopyBuffer(c->cb,c->outDev.buf,c->outStg.buf,1,&bo);
    CHECK(vkEndCommandBuffer(c->cb));
    return submitWait(c,c->cb);
}

// Evaluate `batch` candidates; out_hashes must hold batch*8 uint32 (=batch*32 B).
__declspec(dllexport) int jvk_evaluate(Ctx* c, const int32_t* t_rows, const int32_t* t_cols, int batch, uint32_t* out_hashes) {
    if (!c->jobSet) return -2;
    if (ensureBatch(c,batch)) return -1;
    VkDeviceSize idx=(VkDeviceSize)batch*4, outsz=(VkDeviceSize)batch*8*4; void* m;
    CHECK(vkMapMemory(c->dev,c->trStg.mem,0,idx,0,&m)); std::memcpy(m,t_rows,idx); vkUnmapMemory(c->dev,c->trStg.mem);
    CHECK(vkMapMemory(c->dev,c->tcStg.mem,0,idx,0,&m)); std::memcpy(m,t_cols,idx); vkUnmapMemory(c->dev,c->tcStg.mem);
    if (runBatch(c,batch)) return -1;
    CHECK(vkMapMemory(c->dev,c->outStg.mem,0,outsz,0,&m)); std::memcpy(out_hashes,m,outsz); vkUnmapMemory(c->dev,c->outStg.mem);
    return 0;
}

struct JvkHit { int32_t found; int32_t t_rows; int32_t t_cols; uint8_t hash[32]; int64_t attempts; };

static inline uint64_t ld64(const uint8_t* p){ uint64_t v; std::memcpy(&v,p,8); return v; }
static inline bool ltLE(const uint8_t* h, const uint64_t tw[4]){
    uint64_t h3=ld64(h+24); if(h3!=tw[3])return h3<tw[3];
    uint64_t h2=ld64(h+16); if(h2!=tw[2])return h2<tw[2];
    uint64_t h1=ld64(h+8);  if(h1!=tw[1])return h1<tw[1];
    return ld64(h)<tw[0];
}
// Step one axis (matches candidate_search.enumerate_valid_offsets).
static inline bool axisNext(int& base,int& delta,int mod,int win,int upper,int& o){
    while(base<upper){ int lim=(win<upper-base)?win:upper-base; if(delta<lim){o=base+delta;++delta;return true;} base+=mod; delta=0; }
    return false;
}

// Native search: enumerate valid (t_rows,t_cols), evaluate in batches, return the
// first hash that is < target (LE uint256). Offsets never leave the process.
__declspec(dllexport) int jvk_search(Ctx* c,
    int r_mod,int r_win,int r_upper, int c_mod,int c_win,int c_upper,
    const uint8_t* target_le, int batch, int64_t max_attempts, JvkHit* out) {
    if(!c->jobSet) return -2;
    if(ensureBatch(c,batch)) return -1;
    uint64_t tw[4]={ld64(target_le),ld64(target_le+8),ld64(target_le+16),ld64(target_le+24)};
    int ra_base=0,ra_delta=0,ca_base=0,ca_delta=0,cur_tr=0; bool have_tr=false;
    auto nextOff=[&](int& tr,int& tc)->bool{
        for(;;){
            if(!have_tr){ if(!axisNext(ra_base,ra_delta,r_mod,r_win,r_upper,cur_tr)) return false; ca_base=0;ca_delta=0; have_tr=true; }
            if(axisNext(ca_base,ca_delta,c_mod,c_win,c_upper,tc)){ tr=cur_tr; return true; }
            have_tr=false;
        }
    };
    out->found=0; out->attempts=0;
    std::vector<int32_t> hr(batch), hc(batch);
    int64_t attempts=0;
    for(;;){
        int remaining = (max_attempts>0) ? (int)((max_attempts-attempts < batch) ? (max_attempts-attempts) : batch) : batch;
        if(remaining<=0) break;
        int cnt=0; int tr,tc;
        for(; cnt<remaining; ++cnt){ if(!nextOff(tr,tc)) break; hr[cnt]=tr; hc[cnt]=tc; }
        if(cnt==0) break;
        void* m;
        CHECK(vkMapMemory(c->dev,c->trStg.mem,0,(VkDeviceSize)cnt*4,0,&m)); std::memcpy(m,hr.data(),(size_t)cnt*4); vkUnmapMemory(c->dev,c->trStg.mem);
        CHECK(vkMapMemory(c->dev,c->tcStg.mem,0,(VkDeviceSize)cnt*4,0,&m)); std::memcpy(m,hc.data(),(size_t)cnt*4); vkUnmapMemory(c->dev,c->tcStg.mem);
        if(runBatch(c,cnt)) return -1;
        uint8_t* hp; CHECK(vkMapMemory(c->dev,c->outStg.mem,0,(VkDeviceSize)cnt*32,0,(void**)&hp));
        int hit=-1;
        for(int i=0;i<cnt;++i){ if(ltLE(hp+(size_t)i*32,tw)){hit=i;break;} }
        if(hit>=0){ out->found=1; out->t_rows=hr[hit]; out->t_cols=hc[hit];
            std::memcpy(out->hash,hp+(size_t)hit*32,32); out->attempts=attempts+hit+1;
            vkUnmapMemory(c->dev,c->outStg.mem); return 0; }
        vkUnmapMemory(c->dev,c->outStg.mem);
        attempts+=cnt;
    }
    out->attempts=attempts;
    return 0;
}

__declspec(dllexport) double jvk_last_gpu_ms(Ctx* c) {
    uint64_t ts[2]; if(vkGetQueryPoolResults(c->dev,c->qpool,0,2,sizeof(ts),ts,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT)!=VK_SUCCESS) return -1;
    return (ts[1]-ts[0])*c->tsPeriod/1e6;
}

__declspec(dllexport) void jvk_destroy(Ctx* c) {
    if(!c) return;
    if(c->dev){ vkDeviceWaitIdle(c->dev);
        for(int i=0;i<11;++i) destroyBuf(c,c->job[i]);
        destroyBuf(c,c->trDev);destroyBuf(c,c->tcDev);destroyBuf(c,c->outDev);
        destroyBuf(c,c->trStg);destroyBuf(c,c->tcStg);destroyBuf(c,c->outStg);
        if(c->qpool)vkDestroyQueryPool(c->dev,c->qpool,nullptr);
        if(c->pipe)vkDestroyPipeline(c->dev,c->pipe,nullptr);
        if(c->sm)vkDestroyShaderModule(c->dev,c->sm,nullptr);
        if(c->pl)vkDestroyPipelineLayout(c->dev,c->pl,nullptr);
        if(c->dpool)vkDestroyDescriptorPool(c->dev,c->dpool,nullptr);
        if(c->dsl)vkDestroyDescriptorSetLayout(c->dev,c->dsl,nullptr);
        if(c->cp)vkDestroyCommandPool(c->dev,c->cp,nullptr);
        vkDestroyDevice(c->dev,nullptr);
    }
    if(c->inst)vkDestroyInstance(c->inst,nullptr);
    delete c;
}

} // extern "C"
