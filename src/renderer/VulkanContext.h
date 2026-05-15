#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
#include <string>
#include <optional>

namespace acuity {
    // graphicsFamily: handles draw calls
    // presentFamily: handles presenting images to the window surface
    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;
        bool isComplete() const {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR        capabilites;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR>   presentModes;
    };

    class VulkanContext {
        public:
            // Core Vulkan objects
            VkInstance              instance        = VK_NULL_HANDLE;
            VkSurfaceKHR            surface         = VK_NULL_HANDLE;
            VkPhysicalDevice        physicalDevice  = VK_NULL_HANDLE;
            VkDevice                device          = VK_NULL_HANDLE;
            VkQueue                 graphicsQueue   = VK_NULL_HANDLE;
            VkQueue                 presentQueue    = VK_NULL_HANDLE;
            VkSwapchainKHR          swapchain       = VK_NULL_HANDLE;
            VkFormat                swapchainFormat = VK_FORMAT_UNDEFINED;
            VkExtent2D              swapchainExtent = {0, 0};
            std::vector<VkImage>    swapchainImages;
            std::vector<VkImageView>swapchainImageViews;
            VkCommandPool           commandPool     = VK_NULL_HANDLE;
            uint32_t                graphicsQueueFamily = 0; // cached for OffscreenRenderer

            // Depth buffer - needed for correct depth ordering of triangles
            VkImage         depthImage          = VK_NULL_HANDLE;
            VkDeviceMemory  depthImageMemory    = VK_NULL_HANDLE;
            VkImageView     depthImageView      = VK_NULL_HANDLE;

            // Initialize everything
            void init(GLFWwindow* window);

            // Recreate swapchain on window resize
            void recreateSwapchain(GLFWwindow* window);

            // Clean up all resources
            void destroy();

            // Query helpers used by other classes
            QueueFamilyIndices      findQueueFamilies(VkPhysicalDevice device) const;
            SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) const;
            uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
            VkFormat findDepthFormat() const;

        private:
            void createInstance();
            void createSurface(GLFWwindow* window);
            void pickPhysicalDevice();
            void createLogicalDevice();
            void createSwapchain(GLFWwindow* window);
            void createImageViews();
            void createCommandPool();
            void createDepthResources();

            bool isDeviceSuitable(VkPhysicalDevice device) const;
            bool checkDeviceExtensionSupport(VkPhysicalDevice device) const;

            VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) const;
            VkPresentModeKHR   chooseSwapPresentMode  (const std::vector<VkPresentModeKHR>& modes) const;
            VkExtent2D         chooseSwapExtent       (const VkSurfaceCapabilitiesKHR& caps, GLFWwindow* window) const;

            void createImage(uint32_t w, uint32_t h, VkFormat format,
                     VkImageTiling tiling, VkImageUsageFlags usage,
                     VkMemoryPropertyFlags properties,
                     VkImage& image, VkDeviceMemory& memory);
                     VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
    };
}