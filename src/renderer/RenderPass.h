#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>

namespace acuity {
    class VulkanContext;

    class RenderPass {
        public:
            VkRenderPass renderPass = VK_NULL_HANDLE;
            std::vector<VkFramebuffer> framebuffers;

            void init(VulkanContext& ctx);
            void recreate(VulkanContext& ctx);
            void destroy(VkDevice device);

            // Begin/end render pass for a given frame
            void begin(VkCommandBuffer cmd, uint32_t imageIndex, VkExtent2D extent);
            void end(VkCommandBuffer cmd);
        
        private:
            void createRenderPass(VulkanContext& ctx);
            void createFramebuffers(VulkanContext& ctx);
    };
}