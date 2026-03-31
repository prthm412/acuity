#include "Pipeline.h"
#include "VulkanContext.h"
#include "Mesh.h"
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace acuity {
    // Push constant block size:
    // 2 X mat4 (model + view)      +
    // 1 X mat4 (proj)              = 3 X 64 = 192 bytes
    // Vulkan guarantees at least 128 bytes, most GPUs support 256. We use 192.
    struct PushConstantData {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 projection;
    };

    void Pipeline::init(VulkanContext& ctx, VkRenderPass renderPass, const std::string& vertSpvPath, const std::string& fragSpvPath)
    {
        // Load SPIR-V shaders
        auto vertCode = readSpvFile(vertSpvPath);
        auto fragCode = readSpvFile(fragSpvPath);

        VkShaderModule vertModule = createShaderModule(ctx.device, vertCode);
        VkShaderModule fragModule = createShaderModule(ctx.device, fragCode);

        VkPipelineShaderStageCreateInfo vertStage{};
        vertStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
        vertStage.module = vertModule;
        vertStage.pName  = "main"; // entry point function name in shader

        VkPipelineShaderStageCreateInfo fragStage{};
        fragStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragStage.module = fragModule;
        fragStage.pName  = "main";

        VkPipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

        // Vertex input
        // Tells the pipeline how to read vertex data from the VBO
        auto bindingDesc  = Vertex::getBindingDescription();
        auto attributeDescs = Vertex::getAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount   = 1;
        vertexInput.pVertexBindingDescriptions      = &bindingDesc;
        vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescs.size());
        vertexInput.pVertexAttributeDescriptions    = attributeDescs.data();

        // Input assembly
        // How to interpret vertices: TRIANGLE_LIST = every 3 vertices = one triangle
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // Viewport and scissor (dynamic)
        // Dynamic state lets us change viewport/scissor without recreating the pipeline
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount  = 1;

        // Rasterizer
        // Converts triangles to fragments (pixels). We use filled triangles by default.
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable        = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth               = 1.0f;
        rasterizer.cullMode                = VK_CULL_MODE_BACK_BIT;  // cull back faces
        rasterizer.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable         = VK_FALSE;

        // Multisampling
        // No MSAA for now — keeps it simple, can add later
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable  = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Depth stencil
        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable  = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS; // closer = smaller depth = wins
        depthStencil.stencilTestEnable = VK_FALSE;

        // Color blending
        // No transparency for now — opaque rendering
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable    = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable   = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments    = &colorBlendAttachment;

        // Dynamic state
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates    = dynamicStates.data();

        // Push constants
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushRange.offset     = 0;
        pushRange.size       = sizeof(PushConstantData);

        // Pipeline layout
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges    = &pushRange;

        if (vkCreatePipelineLayout(ctx.device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create pipeline layout");

        // Create graphics pipeline
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount          = 2;
        pipelineInfo.pStages             = stages;
        pipelineInfo.pVertexInputState   = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState      = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState   = &multisampling;
        pipelineInfo.pDepthStencilState  = &depthStencil;
        pipelineInfo.pColorBlendState    = &colorBlending;
        pipelineInfo.pDynamicState       = &dynamicState;
        pipelineInfo.layout              = pipelineLayout;
        pipelineInfo.renderPass          = renderPass;
        pipelineInfo.subpass             = 0;

        if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1,
                                    &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create graphics pipeline");

        // Shader modules can be destroyed after pipeline is created
        vkDestroyShaderModule(ctx.device, vertModule, nullptr);
        vkDestroyShaderModule(ctx.device, fragModule, nullptr);

        std::cout << "[Pipeline] Graphics pipeline created" << std::endl;
    }

    void Pipeline::bind(VkCommandBuffer cmd)
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    }

    void Pipeline::destroy(VkDevice device)
    {
        if (pipeline        != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
        if (pipelineLayout  != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    }

    std::vector<char> Pipeline::readSpvFile(const std::string& path)
    {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Failed to open shader file: " + path);

        size_t fileSize = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        return buffer;
    }

    VkShaderModule Pipeline::createShaderModule(VkDevice device, const std::vector<char>& code)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode    = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule module;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &module) != VK_SUCCESS)
            throw std::runtime_error("Failed to create shader module");
        return module;
    }
}