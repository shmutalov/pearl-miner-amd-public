// Host for pearl_coopmat_rR.spv: load PA/PB/key + meta from a job dir, dispatch
// the fused kernel over a range of workgroups (one WG = one (band,block)), write
// all candidate hashes to hashes.bin for the Python checker.
//
// Build: g++ -std=c++17 -O2 -I"$VULKAN_SDK/Include" -I. pearl_host.cpp volk.c -o pearl_host.exe
// Run:   ./pearl_host.exe <jobdir> <spv> [wg_off] [wg_count]
#define VK_NO_PROTOTYPES
#include "volk.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>

#define CHECK(x) do{ VkResult _r=(x); if(_r!=VK_SUCCESS){ printf("FAIL %s = %d\n", #x, _r); return 1; } }while(0)

static uint32_t pickMem(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want){
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
    for(uint32_t i=0;i<mp.memoryTypeCount;++i)
        if((bits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&want)==want) return i;
    return ~0u;
}
struct Buf{ VkBuffer buf; VkDeviceMemory mem; void* map; VkDeviceSize sz; };

static std::vector<char> readFile(const std::string& p){
    FILE* f=fopen(p.c_str(),"rb"); if(!f){printf("cannot open %s\n",p.c_str());exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<char> b(n); size_t rd=fread(b.data(),1,n,f); (void)rd; fclose(f); return b;
}
// crude meta.json parse: pull "key": int
static long metaInt(const std::string& s, const char* key){
    std::string pat=std::string("\"")+key+"\"";
    size_t p=s.find(pat); if(p==std::string::npos){printf("meta missing %s\n",key);exit(1);}
    p=s.find(':',p)+1; return strtol(s.c_str()+p,nullptr,10);
}

int main(int argc, char** argv){
    if(argc<3){ printf("usage: pearl_host <jobdir> <spv> [wg_off] [wg_count]\n"); return 1; }
    std::string dir=argv[1], spv=argv[2];
    auto meta=readFile(dir+"/meta.json"); std::string ms(meta.begin(),meta.end());
    long M=metaInt(ms,"m"), N=metaInt(ms,"n"), K=metaInt(ms,"k");
    long nbands=metaInt(ms,"nbands"), nblocks=metaInt(ms,"nblocks");
    long totalWG=nbands*nblocks;
    long wg_off = argc>3? atol(argv[3]) : 0;
    long wg_cnt = argc>4? atol(argv[4]) : totalWG;
    printf("m=%ld n=%ld k=%ld nbands=%ld nblocks=%ld totalWG=%ld  run [%ld,%ld)\n",
           M,N,K,nbands,nblocks,totalWG,wg_off,wg_off+wg_cnt);

    if(volkInitialize()!=VK_SUCCESS){printf("no vulkan-1\n");return 1;}
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&app;
    VkInstance inst; CHECK(vkCreateInstance(&ici,nullptr,&inst)); volkLoadInstance(inst);
    uint32_t nd=0; vkEnumeratePhysicalDevices(inst,&nd,nullptr);
    std::vector<VkPhysicalDevice> pds(nd); vkEnumeratePhysicalDevices(inst,&nd,pds.data());
    VkPhysicalDevice pd=pds[0]; VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd,&props);
    printf("device: %s\n", props.deviceName);
    uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&qn,nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qn); vkGetPhysicalDeviceQueueFamilyProperties(pd,&qn,qfs.data());
    uint32_t qfi=~0u; for(uint32_t i=0;i<qn;++i) if(qfs[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qfi=i;break;}

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    cmf.cooperativeMatrix=VK_TRUE;
    VkPhysicalDevice8BitStorageFeatures s8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    s8.storageBuffer8BitAccess=VK_TRUE; s8.pNext=&cmf;
    VkPhysicalDeviceShaderFloat16Int8Features fi8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    fi8.shaderInt8=VK_TRUE; fi8.pNext=&s8;
    VkPhysicalDeviceSubgroupSizeControlFeatures sscf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    sscf.subgroupSizeControl=VK_TRUE; sscf.computeFullSubgroups=VK_TRUE; sscf.pNext=&fi8;
    const char* exts[]={VK_KHR_8BIT_STORAGE_EXTENSION_NAME,VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
                        VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME};
    float prio=1.0f; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex=qfi; qci.queueCount=1; qci.pQueuePriorities=&prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext=&sscf; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
    dci.enabledExtensionCount=4; dci.ppEnabledExtensionNames=exts;
    VkDevice dev; CHECK(vkCreateDevice(pd,&dci,nullptr,&dev)); volkLoadDevice(dev);
    VkQueue queue; vkGetDeviceQueue(dev,qfi,0,&queue);

    auto mkBuf=[&](VkDeviceSize sz, VkMemoryPropertyFlags want, Buf& o)->int{
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size=sz; bci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        CHECK(vkCreateBuffer(dev,&bci,nullptr,&o.buf));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,o.buf,&mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize=mr.size;
        mai.memoryTypeIndex=pickMem(pd,mr.memoryTypeBits,want);
        CHECK(vkAllocateMemory(dev,&mai,nullptr,&o.mem)); CHECK(vkBindBufferMemory(dev,o.buf,o.mem,0));
        CHECK(vkMapMemory(dev,o.mem,0,sz,0,&o.map)); o.sz=sz; return 0;
    };
    VkMemoryPropertyFlags HV=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    auto PA=readFile(dir+"/PA.bin"), PB=readFile(dir+"/PB.bin"), KEY=readFile(dir+"/key.bin");
    Buf bPA,bPB,bKEY,bOUT;
    if(mkBuf(PA.size(),HV,bPA))return 1; memcpy(bPA.map,PA.data(),PA.size());
    if(mkBuf(PB.size(),HV,bPB))return 1; memcpy(bPB.map,PB.data(),PB.size());
    if(mkBuf(KEY.size(),HV,bKEY))return 1; memcpy(bKEY.map,KEY.data(),KEY.size());
    VkDeviceSize outSz=(VkDeviceSize)wg_cnt*32*8*sizeof(uint32_t); // dispatch-relative
    if(mkBuf(outSz,HV,bOUT))return 1; memset(bOUT.map,0,outSz);

    VkDescriptorSetLayoutBinding bnd[4]{};
    for(int i=0;i<4;++i){bnd[i].binding=i;bnd[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bnd[i].descriptorCount=1;bnd[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dl.bindingCount=4; dl.pBindings=bnd;
    VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev,&dl,nullptr,&dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,3*sizeof(int32_t)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; plci.setLayoutCount=1; plci.pSetLayouts=&dsl; plci.pushConstantRangeCount=1; plci.pPushConstantRanges=&pcr;
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev,&plci,nullptr,&pl));
    auto code=readFile(spv);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize=code.size(); smci.pCode=(const uint32_t*)code.data();
    VkShaderModule sm; CHECK(vkCreateShaderModule(dev,&smci,nullptr,&sm));
    VkPipelineShaderStageCreateInfo ss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; ss.stage=VK_SHADER_STAGE_COMPUTE_BIT; ss.module=sm; ss.pName="main";
    uint32_t sgSize = argc>5? (uint32_t)atoi(argv[5]) : 0;
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    if(sgSize){ rss.requiredSubgroupSize=sgSize; ss.pNext=&rss; }
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpci.stage=ss; cpci.layout=pl;
    VkPipeline pipe; CHECK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpci,nullptr,&pipe));

    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4};
    VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dp.maxSets=1; dp.poolSizeCount=1; dp.pPoolSizes=&psz;
    VkDescriptorPool dpool; CHECK(vkCreateDescriptorPool(dev,&dp,nullptr,&dpool));
    VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsa.descriptorPool=dpool; dsa.descriptorSetCount=1; dsa.pSetLayouts=&dsl;
    VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(dev,&dsa,&ds));
    VkDescriptorBufferInfo bi[4]={{bPA.buf,0,VK_WHOLE_SIZE},{bPB.buf,0,VK_WHOLE_SIZE},{bKEY.buf,0,VK_WHOLE_SIZE},{bOUT.buf,0,VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wr[4]{};
    for(int i=0;i<4;++i){wr[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;wr[i].dstSet=ds;wr[i].dstBinding=i;wr[i].descriptorCount=1;wr[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;wr[i].pBufferInfo=&bi[i];}
    vkUpdateDescriptorSets(dev,4,wr,0,nullptr);

    VkCommandPoolCreateInfo cpc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpc.queueFamilyIndex=qfi;
    VkCommandPool cpool; CHECK(vkCreateCommandPool(dev,&cpc,nullptr,&cpool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cbai.commandPool=cpool; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    VkCommandBuffer cb; CHECK(vkAllocateCommandBuffers(dev,&cbai,&cb));
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; cbi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK(vkBeginCommandBuffer(cb,&cbi));
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,nullptr);
    int32_t pcv[3]={(int32_t)nblocks,(int32_t)K,(int32_t)wg_off};
    vkCmdPushConstants(cb,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pcv),pcv);
    vkCmdDispatch(cb,(uint32_t)wg_cnt,1,1);
    CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb;
    auto t0=std::chrono::steady_clock::now();
    CHECK(vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE)); CHECK(vkQueueWaitIdle(queue));
    double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    double cand=(double)wg_cnt*32.0;
    printf("dispatch: %ld WGs, %.0f candidates, %.4f s -> %.2f M cand/s "
           "(host-visible PA/PB = lower bound)\n", wg_cnt, cand, dt, cand/dt/1e6);

    // Write only the candidates we ran: [wg_off*32, (wg_off+wg_cnt)*32) hashes.
    FILE* f=fopen((dir+"/hashes.bin").c_str(),"wb");
    uint32_t* o=(uint32_t*)bOUT.map;            // dispatch-relative, starts at 0
    fwrite(o, sizeof(uint32_t), (size_t)wg_cnt*32*8, f);
    fclose(f);
    printf("wrote %s/hashes.bin (%ld candidates)\n", dir.c_str(), wg_cnt*32);
    return 0;
}
