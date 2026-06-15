// Phase D throughput + search host (DEVICE-LOCAL PA/PB).
//   1. upload raw inputs (A,B,EAL,EBR,EAR,EBL) to device-local
//   2. build PA,PB device-local via pmat_rR.spv  (Phase A)
//   3. run pearl_search_rR.spv over a WG range, GPU-timed, record hits < target
//
// Build: g++ -std=c++17 -O2 -I"$VULKAN_SDK/Include" -I. search_host.cpp volk.c -o search_host.exe
// Run:   ./search_host.exe <jobdir> <pmat.spv> <search.spv> [wg_count] [target_lz]
//        target_lz = leading-zero bits required (LE). 0 => no hits (pure bench).
#define VK_NO_PROTOTYPES
#include "volk.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>

#define CHECK(x) do{ VkResult _r=(x); if(_r!=VK_SUCCESS){ printf("FAIL %s = %d\n", #x, _r); return 1; } }while(0)
static VkDevice dev; static VkPhysicalDevice pd; static VkQueue queue; static uint32_t qfi;
static VkCommandPool cpool; static float tsPeriod=1.0f;

static uint32_t pickMem(uint32_t bits, VkMemoryPropertyFlags want){
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
    for(uint32_t i=0;i<mp.memoryTypeCount;++i) if((bits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&want)==want) return i;
    return ~0u;
}
struct Buf{ VkBuffer buf=VK_NULL_HANDLE; VkDeviceMemory mem=VK_NULL_HANDLE; void* map=nullptr; VkDeviceSize sz=0; };
static int mkBuf(VkDeviceSize sz, VkBufferUsageFlags usage, VkMemoryPropertyFlags mp, Buf& o){
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.size=sz; bci.usage=usage;
    CHECK(vkCreateBuffer(dev,&bci,nullptr,&o.buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,o.buf,&mr);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize=mr.size; mai.memoryTypeIndex=pickMem(mr.memoryTypeBits,mp);
    CHECK(vkAllocateMemory(dev,&mai,nullptr,&o.mem)); CHECK(vkBindBufferMemory(dev,o.buf,o.mem,0)); o.sz=sz;
    if(mp&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) CHECK(vkMapMemory(dev,o.mem,0,sz,0,&o.map));
    return 0;
}
static int submitWait(VkCommandBuffer cb){
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb;
    CHECK(vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE)); CHECK(vkQueueWaitIdle(queue)); return 0;
}
static VkCommandBuffer beginCB(){
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cbai.commandPool=cpool; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    VkCommandBuffer cb; vkAllocateCommandBuffers(dev,&cbai,&cb);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb,&bi); return cb;
}
// upload host data into a fresh device-local buffer (storage|transfer_dst) via staging
static int uploadDL(const void* data, VkDeviceSize sz, VkBufferUsageFlags extra, Buf& o){
    if(mkBuf(sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|extra,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, o)) return 1;
    Buf stg; if(mkBuf(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stg)) return 1;
    memcpy(stg.map,data,sz);
    VkCommandBuffer cb=beginCB(); VkBufferCopy bc{0,0,sz}; vkCmdCopyBuffer(cb,stg.buf,o.buf,1,&bc); vkEndCommandBuffer(cb);
    if(submitWait(cb)) return 1; vkFreeCommandBuffers(dev,cpool,1,&cb);
    vkDestroyBuffer(dev,stg.buf,nullptr); vkUnmapMemory(dev,stg.mem); vkFreeMemory(dev,stg.mem,nullptr);
    return 0;
}
static std::vector<char> readFile(const std::string& p){
    FILE* f=fopen(p.c_str(),"rb"); if(!f){printf("cannot open %s\n",p.c_str());exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); std::vector<char> b(n); size_t rd=fread(b.data(),1,n,f);(void)rd; fclose(f); return b;
}
static long metaInt(const std::string& s,const char* k){ std::string pat=std::string("\"")+k+"\""; size_t p=s.find(pat); if(p==std::string::npos){printf("meta missing %s\n",k);exit(1);} p=s.find(':',p)+1; return strtol(s.c_str()+p,nullptr,10); }

static VkPipeline mkPipe(const std::string& spv, VkPipelineLayout pl){
    auto code=readFile(spv);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize=code.size(); smci.pCode=(const uint32_t*)code.data();
    VkShaderModule sm; vkCreateShaderModule(dev,&smci,nullptr,&sm);
    VkPipelineShaderStageCreateInfo ss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; ss.stage=VK_SHADER_STAGE_COMPUTE_BIT; ss.module=sm; ss.pName="main";
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cp.stage=ss; cp.layout=pl;
    VkPipeline pipe; if(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cp,nullptr,&pipe)!=VK_SUCCESS){printf("pipeline fail %s\n",spv.c_str());exit(1);} return pipe;
}

int main(int argc,char** argv){
    if(argc<4){ printf("usage: search_host <jobdir> <pmat.spv> <search.spv> [wg_count] [target_lz]\n"); return 1; }
    std::string dir=argv[1], pmatSpv=argv[2], searchSpv=argv[3];
    auto meta=readFile(dir+"/meta.json"); std::string ms(meta.begin(),meta.end());
    long M=metaInt(ms,"m"),N=metaInt(ms,"n"),K=metaInt(ms,"k"),R=metaInt(ms,"r"),nbands=metaInt(ms,"nbands"),nblocks=metaInt(ms,"nblocks");
    long totalWG=nbands*nblocks;
    long wg_cnt = argc>4? atol(argv[4]) : totalWG;
    int target_lz = argc>5? atoi(argv[5]) : 0;
    if(wg_cnt>totalWG) wg_cnt=totalWG;
    printf("m=%ld n=%ld k=%ld r=%ld totalWG=%ld run=%ld target_lz=%d\n",M,N,K,R,totalWG,wg_cnt,target_lz);

    if(volkInitialize()!=VK_SUCCESS){printf("no vulkan-1\n");return 1;}
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&app;
    VkInstance inst; CHECK(vkCreateInstance(&ici,nullptr,&inst)); volkLoadInstance(inst);
    uint32_t nd=0; vkEnumeratePhysicalDevices(inst,&nd,nullptr); std::vector<VkPhysicalDevice> pds(nd); vkEnumeratePhysicalDevices(inst,&nd,pds.data()); pd=pds[0];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd,&props); tsPeriod=props.limits.timestampPeriod; printf("device: %s\n",props.deviceName);
    uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&qn,nullptr); std::vector<VkQueueFamilyProperties> qfs(qn); vkGetPhysicalDeviceQueueFamilyProperties(pd,&qn,qfs.data());
    qfi=~0u; for(uint32_t i=0;i<qn;++i) if(qfs[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qfi=i;break;}
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR}; cmf.cooperativeMatrix=VK_TRUE;
    VkPhysicalDevice8BitStorageFeatures s8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES}; s8.storageBuffer8BitAccess=VK_TRUE; s8.pNext=&cmf;
    VkPhysicalDeviceShaderFloat16Int8Features fi8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES}; fi8.shaderInt8=VK_TRUE; fi8.pNext=&s8;
    const char* exts[]={VK_KHR_8BIT_STORAGE_EXTENSION_NAME,VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME};
    float prio=1.0f; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex=qfi; qci.queueCount=1; qci.pQueuePriorities=&prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.pNext=&fi8; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci; dci.enabledExtensionCount=3; dci.ppEnabledExtensionNames=exts;
    CHECK(vkCreateDevice(pd,&dci,nullptr,&dev)); volkLoadDevice(dev); vkGetDeviceQueue(dev,qfi,0,&queue);
    VkCommandPoolCreateInfo cpc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpc.queueFamilyIndex=qfi; cpc.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CHECK(vkCreateCommandPool(dev,&cpc,nullptr,&cpool));
    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qpci.queryType=VK_QUERY_TYPE_TIMESTAMP; qpci.queryCount=2;
    VkQueryPool qpool; CHECK(vkCreateQueryPool(dev,&qpci,nullptr,&qpool));

    // ---- upload raw inputs device-local ----
    auto A=readFile(dir+"/A.bin"),B=readFile(dir+"/B.bin"),EAL=readFile(dir+"/EAL.bin"),EBR=readFile(dir+"/EBR.bin"),EAR=readFile(dir+"/EAR.bin"),EBL=readFile(dir+"/EBL.bin"),KEY=readFile(dir+"/key.bin");
    Buf dA,dB,dEAL,dEBR,dEAR,dEBL,dPA,dPB,dKEY,dTGT,dHIT;
    printf("uploading raw inputs device-local...\n");
    if(uploadDL(A.data(),A.size(),0,dA))return 1;  if(uploadDL(B.data(),B.size(),0,dB))return 1;
    if(uploadDL(EAL.data(),EAL.size(),0,dEAL))return 1; if(uploadDL(EBR.data(),EBR.size(),0,dEBR))return 1;
    if(uploadDL(EAR.data(),EAR.size(),0,dEAR))return 1; if(uploadDL(EBL.data(),EBL.size(),0,dEBL))return 1;
    if(uploadDL(KEY.data(),KEY.size(),0,dKEY))return 1;
    if(mkBuf((VkDeviceSize)M*K,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,dPA))return 1;
    if(mkBuf((VkDeviceSize)N*K,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,dPB))return 1;
    // target (8 u32 LE) from target_lz: target = 2^(256-lz). host-visible.
    uint32_t tgt[8]={0,0,0,0,0,0,0,0};
    if(target_lz<=0){ for(int i=0;i<8;++i) tgt[i]=0xFFFFFFFFu; }   // everything < target -> all hits
    else { int bit=256-target_lz; tgt[bit/32]=1u<<(bit%32); }       // target = 2^(256-lz)
    if(mkBuf(32,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,dTGT))return 1; memcpy(dTGT.map,tgt,32);
    const int MAX_HITS=1024; VkDeviceSize hitSz=(1+MAX_HITS*10)*sizeof(uint32_t);
    if(mkBuf(hitSz,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,dHIT))return 1; memset(dHIT.map,0,hitSz);

    // ---- pmat pipeline (4 storage bindings, push int k) ----
    VkDescriptorSetLayoutBinding pb4[4]{}; for(int i=0;i<4;++i){pb4[i].binding=i;pb4[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;pb4[i].descriptorCount=1;pb4[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
    VkDescriptorSetLayoutCreateInfo dl4{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dl4.bindingCount=4; dl4.pBindings=pb4;
    VkDescriptorSetLayout dslP; CHECK(vkCreateDescriptorSetLayout(dev,&dl4,nullptr,&dslP));
    VkPushConstantRange pcrP{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(int32_t)};
    VkPipelineLayoutCreateInfo plP{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; plP.setLayoutCount=1; plP.pSetLayouts=&dslP; plP.pushConstantRangeCount=1; plP.pPushConstantRanges=&pcrP;
    VkPipelineLayout plPmat; CHECK(vkCreatePipelineLayout(dev,&plP,nullptr,&plPmat));
    VkPipeline pipePmat=mkPipe(pmatSpv,plPmat);

    // ---- search pipeline (5 storage bindings, push 4 ints) ----
    VkDescriptorSetLayoutBinding sb5[5]{}; for(int i=0;i<5;++i){sb5[i].binding=i;sb5[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;sb5[i].descriptorCount=1;sb5[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
    VkDescriptorSetLayoutCreateInfo dl5{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dl5.bindingCount=5; dl5.pBindings=sb5;
    VkDescriptorSetLayout dslS; CHECK(vkCreateDescriptorSetLayout(dev,&dl5,nullptr,&dslS));
    VkPushConstantRange pcrS{VK_SHADER_STAGE_COMPUTE_BIT,0,4*sizeof(int32_t)};
    VkPipelineLayoutCreateInfo plS{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; plS.setLayoutCount=1; plS.pSetLayouts=&dslS; plS.pushConstantRangeCount=1; plS.pPushConstantRanges=&pcrS;
    VkPipelineLayout plSearch; CHECK(vkCreatePipelineLayout(dev,&plS,nullptr,&plSearch));
    VkPipeline pipeSearch=mkPipe(searchSpv,plSearch);

    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4+4+5};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpci.maxSets=3; dpci.poolSizeCount=1; dpci.pPoolSizes=&dps;
    VkDescriptorPool dpool; CHECK(vkCreateDescriptorPool(dev,&dpci,nullptr,&dpool));
    auto allocSet=[&](VkDescriptorSetLayout l){ VkDescriptorSetAllocateInfo a{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; a.descriptorPool=dpool; a.descriptorSetCount=1; a.pSetLayouts=&l; VkDescriptorSet s; vkAllocateDescriptorSets(dev,&a,&s); return s; };
    auto writeSet=[&](VkDescriptorSet s, std::vector<VkBuffer> bufs){
        std::vector<VkDescriptorBufferInfo> bi(bufs.size()); std::vector<VkWriteDescriptorSet> w(bufs.size());
        for(size_t i=0;i<bufs.size();++i){ bi[i]={bufs[i],0,VK_WHOLE_SIZE}; w[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[i].dstSet=s; w[i].dstBinding=(uint32_t)i; w[i].descriptorCount=1; w[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo=&bi[i]; }
        vkUpdateDescriptorSets(dev,(uint32_t)w.size(),w.data(),0,nullptr);
    };
    VkDescriptorSet dsPA=allocSet(dslP), dsPB=allocSet(dslP), dsS=allocSet(dslS);
    writeSet(dsPA,{dA.buf,dEAL.buf,dEAR.buf,dPA.buf});
    writeSet(dsPB,{dB.buf,dEBR.buf,dEBL.buf,dPB.buf});
    writeSet(dsS ,{dPA.buf,dPB.buf,dKEY.buf,dTGT.buf,dHIT.buf});

    // ---- build PA/PB (timed) ----
    VkCommandBuffer cb=beginCB();
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipePmat);
    int32_t kk=(int32_t)K;
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,plPmat,0,1,&dsPA,0,nullptr);
    vkCmdPushConstants(cb,plPmat,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(kk),&kk);
    vkCmdDispatch(cb,(uint32_t)M,1,1);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,plPmat,0,1,&dsPB,0,nullptr);
    vkCmdDispatch(cb,(uint32_t)N,1,1);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,nullptr,0,nullptr);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb;
    vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE); vkQueueWaitIdle(queue); vkFreeCommandBuffers(dev,cpool,1,&cb);
    printf("PA/PB built device-local.\n");

    // ---- search (GPU-timed) ----
    int32_t pcv[4]={(int32_t)nblocks,(int32_t)K,0,(int32_t)MAX_HITS};
    VkCommandBuffer cs=beginCB();
    vkCmdResetQueryPool(cs,qpool,0,2);
    vkCmdBindPipeline(cs,VK_PIPELINE_BIND_POINT_COMPUTE,pipeSearch);
    vkCmdBindDescriptorSets(cs,VK_PIPELINE_BIND_POINT_COMPUTE,plSearch,0,1,&dsS,0,nullptr);
    vkCmdPushConstants(cs,plSearch,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pcv),pcv);
    vkCmdWriteTimestamp(cs,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,qpool,0);
    vkCmdDispatch(cs,(uint32_t)wg_cnt,1,1);
    vkCmdWriteTimestamp(cs,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,qpool,1);
    vkEndCommandBuffer(cs);
    if(submitWait(cs)) return 1; vkFreeCommandBuffers(dev,cpool,1,&cs);

    uint64_t ts[2]={0,0}; vkGetQueryPoolResults(dev,qpool,0,2,sizeof(ts),ts,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT);
    double gpu_s=(ts[1]-ts[0])*(double)tsPeriod*1e-9;
    double cand=(double)wg_cnt*32.0;
    uint32_t* hit=(uint32_t*)dHIT.map; uint32_t nhit=hit[0];
    printf("SEARCH: %.0f candidates in %.4f s GPU = %.2f M cand/s (device-local PA/PB)\n", cand, gpu_s, cand/gpu_s/1e6);
    printf("hits: %u (target_lz=%d)\n", nhit, target_lz);
    uint32_t show = nhit<5?nhit:5;
    for(uint32_t i=0;i<show;++i){ uint32_t* r=hit+1+i*10; printf("  hit t_r=%u t_c=%u hash=", r[0],r[1]); for(int j=0;j<8;++j) printf("%08x", r[2+j]); printf("\n"); }
    // Dump recorded hits for the python validator: count + min(nhit,MAX_HITS) records.
    uint32_t saved = nhit<(uint32_t)MAX_HITS?nhit:(uint32_t)MAX_HITS;
    FILE* hf=fopen((dir+"/hits_out.bin").c_str(),"wb");
    fwrite(&nhit,sizeof(uint32_t),1,hf);
    fwrite(hit+1,sizeof(uint32_t),(size_t)saved*10,hf);
    fclose(hf);
    return 0;
}
