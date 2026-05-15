#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
#include <string>

namespace acuity {
    class VulkanContext;

    // Usage:
    //   OffscreenRenderer osr;
    //   osr.init(ctx, renderPass, width, height);
    //   osr.render(cmd, ...);
    //   auto pixels = osr.readPixels();
    //   osr.destroy(device);

    class OffscreenRenderer {
        public:
            uint32_t width  = 512;
            uint32_t height = 512;

            // Vulkan objects for the offscreen framebuffer
            VkImage         colorImage      = VK_NULL_HANDLE;
            VkDeviceMemory  colorMemory     = VK_NULL_HANDLE;
            VkImageView    colorView        = VK_NULL_HANDLE;

            VkImage        depthImage       = VK_NULL_HANDLE;
            VkDeviceMemory depthMemory      = VK_NULL_HANDLE;
            VkImageView    depthView        = VK_NULL_HANDLE;

            // Host-visible staging buffer to copy pixels from GPU to CPU
            VkBuffer       stagingBuffer    = VK_NULL_HANDLE;
            VkDeviceMemory stagingMemory    = VK_NULL_HANDLE;

            VkFramebuffer  framebuffer      = VK_NULL_HANDLE;
            VkRenderPass   renderPass       = VK_NULL_HANDLE;
            VkCommandPool  commandPool      = VK_NULL_HANDLE;
            VkCommandBuffer commandBuffer   = VK_NULL_HANDLE;
            VkFence        fence            = VK_NULL_HANDLE;

            void init(VulkanContext& ctx, uint32_t w, uint32_t h);
            void destroy(VkDevice device);

            // Read pixels from GPU to CPU after rendering
            // Returns RGBA bytes, width * height * 4 bytes total
            std::vector<uint8_t> readPixels(VulkanContext& ctx);

            // Begin recording a render command
            void beginFrame();

            // End recording a render command
            void endFrame(VulkanContext& ctx);
        
        private:
            void createColorResources(VulkanContext& ctx);
            void createDepthResources(VulkanContext& ctx);
            void createRenderPass(VkDevice device);
            void createFramebuffer(VkDevice device);
            void createStagingBuffer(VulkanContext& ctx);
            void createCommandBuffer(VulkanContext& ctx);

            uint32_t findMemoryType(VkPhysicalDevice  physicalDevice, uint32_t typeBits, VkMemoryPropertyFlags props);

            void createImage(VulkanContext& ctx, uint32_t w, uint32_t h,
                             VkFormat format, VkImageUsageFlags usage,
                             VkMemoryPropertyFlags memProps,
                             VkImage& image, VkDeviceMemory& memory);

            void createImageView(VkDevice device, VkImage image, VkFormat format,
                                 VkImageAspectFlags aspect, VkImageView& view);
    };
}