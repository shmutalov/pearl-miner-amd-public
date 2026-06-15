// Host harness for gemm_coopmat.spv: upload random int8 A(MxK), B(NxK), run the
// coopmat GEMM, read back C(MxN int32), compare to a CPU reference C = A * B^T.
// Proves the GLSL cooperative_matrix path is correct on this device's LLPC.
//
// Build: g++ -std=c++17 -O2 -I"$VULKAN_SDK/Include" -I. gemm_host.cpp volk.c -o gemm_host.exe
// Run:   ./gemm_host.exe   (expects gemm_coopmat.spv next to it)
#define VK_NO_PROTOTYPES
#include "volk.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static const int M = 64, N = 64, K = 128;

#define CHECK(x) do{ VkResult _r=(x); if(_r!=VK_SUCCESS){ printf("FAIL %s = %d\n", #x, _r); return 1; } }while(0)

static uint32_t pickMem(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i=0;i<mp.memoryTypeCount;++i)
        if ((bits&(1u<<i)) && (mp.memoryTypes[i].propertyFlags&want)==want) return i;
    return ~0u;
}

struct Buf { VkBuffer buf; VkDeviceMemory mem; void* map; };

int main() {
    if (volkInitialize()!=VK_SUCCESS){ printf("no vulkan-1\n"); return 1; }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&app;
    VkInstance inst; CHECK(vkCreateInstance(&ici,nullptr,&inst)); volkLoadInstance(inst);
    uint32_t nd=0; vkEnumeratePhysicalDevices(inst,&nd,nullptr);
    std::vector<VkPhysicalDevice> pds(nd); vkEnumeratePhysicalDevices(inst,&nd,pds.data());
    VkPhysicalDevice pd=pds[0];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd,&props);
    printf("device: %s\n", props.deviceName);

    uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&qn,nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qn); vkGetPhysicalDeviceQueueFamilyProperties(pd,&qn,qfs.data());
    uint32_t qfi=~0u; for(uint32_t i=0;i<qn;++i) if(qfs[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qfi=i;break;}

    // Feature chain: cooperativeMatrix + shaderInt8 + 8bit storage.
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    cmf.cooperativeMatrix=VK_TRUE;
    VkPhysicalDevice8BitStorageFeatures s8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    s8.storageBuffer8BitAccess=VK_TRUE; s8.pNext=&cmf;
    VkPhysicalDeviceShaderFloat16Int8Features fi8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    fi8.shaderInt8=VK_TRUE; fi8.pNext=&s8;
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    v11.storageBuffer16BitAccess=VK_FALSE; v11.pNext=&fi8;  // just a carrier; nothing required

    const char* exts[]={VK_KHR_8BIT_STORAGE_EXTENSION_NAME, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
                        VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME};
    float prio=1.0f; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex=qfi; qci.queueCount=1; qci.pQueuePriorities=&prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext=&fi8; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
    dci.enabledExtensionCount=3; dci.ppEnabledExtensionNames=exts;
    VkDevice dev; CHECK(vkCreateDevice(pd,&dci,nullptr,&dev)); volkLoadDevice(dev);
    VkQueue queue; vkGetDeviceQueue(dev,qfi,0,&queue);

    auto mkBuf=[&](VkDeviceSize sz, Buf& out)->int{
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size=sz; bci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        CHECK(vkCreateBuffer(dev,&bci,nullptr,&out.buf));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,out.buf,&mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize=mr.size;
        mai.memoryTypeIndex=pickMem(pd,mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        CHECK(vkAllocateMemory(dev,&mai,nullptr,&out.mem));
        CHECK(vkBindBufferMemory(dev,out.buf,out.mem,0));
        CHECK(vkMapMemory(dev,out.mem,0,sz,0,&out.map));
        return 0;
    };
    Buf A,B,C;
    if(mkBuf(M*K, A)) return 1;
    if(mkBuf(N*K, B)) return 1;
    if(mkBuf(M*N*sizeof(int32_t), C)) return 1;

    // Fill A,B with a deterministic pseudo-random int8 pattern in [-64,63].
    int8_t* a=(int8_t*)A.map; int8_t* b=(int8_t*)B.map;
    uint32_t s=0x12345678u;
    auto nxt=[&](){ s^=s<<13; s^=s>>17; s^=s<<5; return s; };
    for(int i=0;i<M*K;++i) a[i]=(int8_t)((int)(nxt()%128)-64);
    for(int i=0;i<N*K;++i) b[i]=(int8_t)((int)(nxt()%128)-64);
    std::memset(C.map,0,M*N*sizeof(int32_t));

    // Descriptor set (3 storage buffers).
    VkDescriptorSetLayoutBinding bnd[3]{};
    for(int i=0;i<3;++i){bnd[i].binding=i;bnd[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bnd[i].descriptorCount=1;bnd[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dl.bindingCount=3; dl.pBindings=bnd;
    VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev,&dl,nullptr,&dsl));
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; plci.setLayoutCount=1; plci.pSetLayouts=&dsl;
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev,&plci,nullptr,&pl));

    FILE* f=fopen("gemm_coopmat.spv","rb"); if(!f){printf("no gemm_coopmat.spv\n");return 1;}
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<char> code(sz); size_t rd=fread(code.data(),1,sz,f); (void)rd; fclose(f);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize=sz; smci.pCode=(const uint32_t*)code.data();
    VkShaderModule sm; CHECK(vkCreateShaderModule(dev,&smci,nullptr,&sm));
    VkPipelineShaderStageCreateInfo ss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ss.stage=VK_SHADER_STAGE_COMPUTE_BIT; ss.module=sm; ss.pName="main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpci.stage=ss; cpci.layout=pl;
    VkPipeline pipe; CHECK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpci,nullptr,&pipe));

    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3};
    VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dp.maxSets=1; dp.poolSizeCount=1; dp.pPoolSizes=&psz;
    VkDescriptorPool dpool; CHECK(vkCreateDescriptorPool(dev,&dp,nullptr,&dpool));
    VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsa.descriptorPool=dpool; dsa.descriptorSetCount=1; dsa.pSetLayouts=&dsl;
    VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(dev,&dsa,&ds));
    VkDescriptorBufferInfo bi[3]={{A.buf,0,VK_WHOLE_SIZE},{B.buf,0,VK_WHOLE_SIZE},{C.buf,0,VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w[3]{};
    for(int i=0;i<3;++i){w[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w[i].dstSet=ds;w[i].dstBinding=i;w[i].descriptorCount=1;w[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w[i].pBufferInfo=&bi[i];}
    vkUpdateDescriptorSets(dev,3,w,0,nullptr);

    VkCommandPoolCreateInfo cpc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpc.queueFamilyIndex=qfi;
    VkCommandPool cpool; CHECK(vkCreateCommandPool(dev,&cpc,nullptr,&cpool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cbai.commandPool=cpool; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    VkCommandBuffer cb; CHECK(vkAllocateCommandBuffers(dev,&cbai,&cb));
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; cbi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK(vkBeginCommandBuffer(cb,&cbi));
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,nullptr);
    vkCmdDispatch(cb,1,1,1);
    CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb;
    CHECK(vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(queue));

    // CPU reference C = A * B^T (int32).
    int32_t* gpu=(int32_t*)C.map;
    int bad=0, first=-1;
    for(int i=0;i<M;++i) for(int j=0;j<N;++j){
        int32_t acc=0;
        for(int l=0;l<K;++l) acc += (int32_t)a[i*K+l]*(int32_t)b[j*K+l];
        if(gpu[i*N+j]!=acc){ if(first<0){first=i*N+j; printf("  first mismatch at (%d,%d): gpu=%d cpu=%d\n",i,j,gpu[i*N+j],acc);} ++bad; }
    }
    printf("M=%d N=%d K=%d : %d / %d cells correct\n", M,N,K, M*N-bad, M*N);
    printf("%s\n", bad? "COOPMAT GEMM **WRONG**" : "COOPMAT GEMM CORRECT");
    return bad?2:0;
}
