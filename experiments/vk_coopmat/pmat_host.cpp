// Phase A validation host: build PA (from A,EAL,EAR) and PB (from B,EBR,EBL) on
// the GPU via pmat_rR.spv, then byte-compare against the trusted numpy PA.bin /
// PB.bin. Self-contained (no Python needed for the compare).
//
// Build: g++ -std=c++17 -O2 -I"$VULKAN_SDK/Include" -I. pmat_host.cpp volk.c -o pmat_host.exe
// Run:   ./pmat_host.exe <jobdir> <spv>
#define VK_NO_PROTOTYPES
#include "volk.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>

#define CHECK(x) do{ VkResult _r=(x); if(_r!=VK_SUCCESS){ printf("FAIL %s = %d\n", #x, _r); return 1; } }while(0)
static uint32_t pickMem(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want){
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
    for(uint32_t i=0;i<mp.memoryTypeCount;++i) if((bits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&want)==want) return i;
    return ~0u;
}
struct Buf{ VkBuffer buf; VkDeviceMemory mem; void* map; };
static std::vector<char> readFile(const std::string& p){
    FILE* f=fopen(p.c_str(),"rb"); if(!f){printf("cannot open %s\n",p.c_str());exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<char> b(n); size_t rd=fread(b.data(),1,n,f);(void)rd; fclose(f); return b;
}
static long metaInt(const std::string& s, const char* key){
    std::string pat=std::string("\"")+key+"\""; size_t p=s.find(pat);
    if(p==std::string::npos){printf("meta missing %s\n",key);exit(1);} p=s.find(':',p)+1;
    return strtol(s.c_str()+p,nullptr,10);
}

static VkDevice dev; static VkPhysicalDevice pd; static VkQueue queue; static uint32_t qfi;
static VkMemoryPropertyFlags HV=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
static int mkBuf(VkDeviceSize sz, Buf& o){
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.size=sz; bci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    CHECK(vkCreateBuffer(dev,&bci,nullptr,&o.buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,o.buf,&mr);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize=mr.size;
    mai.memoryTypeIndex=pickMem(pd,mr.memoryTypeBits,HV);
    CHECK(vkAllocateMemory(dev,&mai,nullptr,&o.mem)); CHECK(vkBindBufferMemory(dev,o.buf,o.mem,0));
    CHECK(vkMapMemory(dev,o.mem,0,sz,0,&o.map)); return 0;
}

int main(int argc, char** argv){
    if(argc<3){ printf("usage: pmat_host <jobdir> <spv>\n"); return 1; }
    std::string dir=argv[1], spv=argv[2];
    auto meta=readFile(dir+"/meta.json"); std::string ms(meta.begin(),meta.end());
    long M=metaInt(ms,"m"), N=metaInt(ms,"n"), K=metaInt(ms,"k"), R=metaInt(ms,"r");
    printf("m=%ld n=%ld k=%ld r=%ld\n",M,N,K,R);

    if(volkInitialize()!=VK_SUCCESS){printf("no vulkan-1\n");return 1;}
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&app;
    VkInstance inst; CHECK(vkCreateInstance(&ici,nullptr,&inst)); volkLoadInstance(inst);
    uint32_t nd=0; vkEnumeratePhysicalDevices(inst,&nd,nullptr); std::vector<VkPhysicalDevice> pds(nd);
    vkEnumeratePhysicalDevices(inst,&nd,pds.data()); pd=pds[0];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd,&props); printf("device: %s\n",props.deviceName);
    uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&qn,nullptr); std::vector<VkQueueFamilyProperties> qfs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd,&qn,qfs.data()); qfi=~0u;
    for(uint32_t i=0;i<qn;++i) if(qfs[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qfi=i;break;}
    VkPhysicalDevice8BitStorageFeatures s8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES}; s8.storageBuffer8BitAccess=VK_TRUE;
    VkPhysicalDeviceShaderFloat16Int8Features fi8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES}; fi8.shaderInt8=VK_TRUE; fi8.pNext=&s8;
    const char* exts[]={VK_KHR_8BIT_STORAGE_EXTENSION_NAME,VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME};
    float prio=1.0f; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex=qfi; qci.queueCount=1; qci.pQueuePriorities=&prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.pNext=&fi8; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci; dci.enabledExtensionCount=2; dci.ppEnabledExtensionNames=exts;
    CHECK(vkCreateDevice(pd,&dci,nullptr,&dev)); volkLoadDevice(dev); vkGetDeviceQueue(dev,qfi,0,&queue);

    // pipeline
    VkDescriptorSetLayoutBinding bnd[4]{};
    for(int i=0;i<4;++i){bnd[i].binding=i;bnd[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bnd[i].descriptorCount=1;bnd[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dl.bindingCount=4; dl.pBindings=bnd;
    VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev,&dl,nullptr,&dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(int32_t)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; plci.setLayoutCount=1; plci.pSetLayouts=&dsl; plci.pushConstantRangeCount=1; plci.pPushConstantRanges=&pcr;
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev,&plci,nullptr,&pl));
    auto code=readFile(spv);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize=code.size(); smci.pCode=(const uint32_t*)code.data();
    VkShaderModule sm; CHECK(vkCreateShaderModule(dev,&smci,nullptr,&sm));
    VkPipelineShaderStageCreateInfo ss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; ss.stage=VK_SHADER_STAGE_COMPUTE_BIT; ss.module=sm; ss.pName="main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpci.stage=ss; cpci.layout=pl;
    VkPipeline pipe; CHECK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpci,nullptr,&pipe));
    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4*2};
    VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dp.maxSets=2; dp.poolSizeCount=1; dp.pPoolSizes=&psz;
    VkDescriptorPool dpool; CHECK(vkCreateDescriptorPool(dev,&dp,nullptr,&dpool));
    VkCommandPoolCreateInfo cpc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpc.queueFamilyIndex=qfi;
    VkCommandPool cpool; CHECK(vkCreateCommandPool(dev,&cpc,nullptr,&cpool));

    auto runOne=[&](const char* sName,const char* eName,const char* pName,long rows,
                    const std::string& refName)->int{
        auto S=readFile(dir+"/"+sName), E=readFile(dir+"/"+eName), P=readFile(dir+"/"+pName);
        Buf bS,bE,bP,bO;
        if(mkBuf(S.size(),bS))return 1; memcpy(bS.map,S.data(),S.size());
        if(mkBuf(E.size(),bE))return 1; memcpy(bE.map,E.data(),E.size());
        if(mkBuf(P.size(),bP))return 1; memcpy(bP.map,P.data(),P.size());      // perm (k x 2 u32)
        if(mkBuf((VkDeviceSize)rows*K,bO))return 1; memset(bO.map,0,(size_t)rows*K);
        VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsa.descriptorPool=dpool; dsa.descriptorSetCount=1; dsa.pSetLayouts=&dsl;
        VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(dev,&dsa,&ds));
        VkDescriptorBufferInfo bi[4]={{bS.buf,0,VK_WHOLE_SIZE},{bE.buf,0,VK_WHOLE_SIZE},{bP.buf,0,VK_WHOLE_SIZE},{bO.buf,0,VK_WHOLE_SIZE}};
        VkWriteDescriptorSet wr[4]{}; for(int i=0;i<4;++i){wr[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;wr[i].dstSet=ds;wr[i].dstBinding=i;wr[i].descriptorCount=1;wr[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;wr[i].pBufferInfo=&bi[i];}
        vkUpdateDescriptorSets(dev,4,wr,0,nullptr);
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cbai.commandPool=cpool; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
        VkCommandBuffer cb; CHECK(vkAllocateCommandBuffers(dev,&cbai,&cb));
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; cbi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK(vkBeginCommandBuffer(cb,&cbi));
        vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
        vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,nullptr);
        int32_t kk=(int32_t)K; vkCmdPushConstants(cb,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(kk),&kk);
        vkCmdDispatch(cb,(uint32_t)rows,1,1);
        CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb;
        CHECK(vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE)); CHECK(vkQueueWaitIdle(queue));
        // compare bO vs reference P matrix file
        auto REF=readFile(dir+"/"+refName);
        const int8_t* g=(const int8_t*)bO.map; const int8_t* rref=(const int8_t*)REF.data();
        long bad=0, firsti=-1; for(long i=0;i<rows*K;++i){ if(g[i]!=rref[i]){ if(firsti<0)firsti=i; ++bad; } }
        printf("  %s: %ld/%ld bytes match%s\n", refName.c_str(), rows*K-bad, rows*K,
               bad? "" : "  OK");
        if(bad) printf("    first mismatch at %ld: gpu=%d ref=%d\n", firsti,(int)g[firsti],(int)rref[firsti]);
        return bad?2:0;
    };

    int rc=0;
    printf("building PA (A,EAL,EAR)...\n");  rc|=runOne("A.bin","EAL.bin","EAR.bin", M, "PA.bin");
    printf("building PB (B,EBR,EBL)...\n");  rc|=runOne("B.bin","EBR.bin","EBL.bin", N, "PB.bin");
    printf(rc? "PHASE A **WRONG**\n" : "PHASE A CORRECT (PA/PB bit-identical to numpy)\n");
    return rc;
}
