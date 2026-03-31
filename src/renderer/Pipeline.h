#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <string>
#include <vector>

namespace acuity {
    class VulkanContext;

    class Pipeline {
        public:
            VkPipeline          pipeline        = VK_NULL_HANDLE;
            VkPipelineLayout    pipelineLayout  = VK_NULL_HANDLE;

            void init(VulkanContext& ctx, VkRenderPass renderPass, const std::string& vertSpvPath, const std::string& fragSpvPath);
            void destroy(VkDevice device);
            void bind(VkCommandBuffer cmd);

        private:
            std::vector<char> readSpvFile(const std::string& path);
            VkShaderModule    createShaderModule(VkDevice device, const std::vector<char>& code);
    };
}