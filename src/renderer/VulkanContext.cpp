#include "VulkanContext.h"
#include <stdexcept>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <limits>

namespace acuity {
    // Required device extensions - swapchain to display images
    const std::vector<const char*> DEVICE_EXTENSIONS = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    void VulkanContext::init(GLFWwindow* window)
    {
        createInstance();
        createSurface(window);
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain(window);
        createImageViews();
        createCommandPool();
        createDepthResources();
        std::cout << "[Vulkan Context] Initialized successfully" << std::endl;
    }

    // Instance
    void VulkanContext::createInstance()
    {
        VkApplicationInfo appInfo{};
        appInfo.sType               = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName    = "Acuity";
        appInfo.applicationVersion  = VK_MAKE_VERSION(0, 1, 0);
        appInfo.pEngineName         = "AcuityEngine";
        appInfo.engineVersion       = VK_MAKE_VERSION(0, 1, 0);
        appInfo.apiVersion          = VK_API_VERSION_1_2;

        uint32_t glfwExtCount = 0;
        const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
        std::vector<const char*> extensions(glfwExts, glfwExts + glfwExtCount);

        VkInstanceCreateInfo createInfo{};
        createInfo.sType                    = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo         = &appInfo;
        createInfo.enabledExtensionCount    = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames  = extensions.data();
        createInfo.enabledLayerCount        = 0;    // No validation layers in release

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan instance.");
    }

    // Surface
    void VulkanContext::createSurface(GLFWwindow* window)
    {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
            throw std::runtime_error("Failed to create window surface");
    }

    // Physical Device
    void VulkanContext::pickPhysicalDevice()
    {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        if (count == 0) throw std::runtime_error("No GPUs with Vulkan support.");

        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());

        // Prefer discrete GPU, fall back to any suitable device
        for (auto& dev : devices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(dev, &props);
            if (isDeviceSuitable(dev) && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physicalDevice = dev;
                std::cout << "[VulkanContext] GPU: " << props.deviceName << std::endl;
                return;
            }
        }
        for (auto& dev : devices) {
            if (isDeviceSuitable(dev)) {
                physicalDevice = dev;
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(dev, &props);
                std::cout << "[VulkanContext] GPU: " << props.deviceName << std::endl;
                return;
            }
        }
        throw std::runtime_error("No suitable GPU found");
    }

    bool VulkanContext::isDeviceSuitable(VkPhysicalDevice dev) const
    {
        QueueFamilyIndices indices = findQueueFamilies(dev);
        bool extsSupported = checkDeviceExtensionSupport(dev);
        bool swapchainOk = false;
        if (extsSupported) {
            SwapChainSupportDetails sc = querySwapChainSupport(dev);
            swapchainOk = !sc.formats.empty() && !sc.presentModes.empty();
        }
        return indices.isComplete() && extsSupported && swapchainOk;
    }

    bool VulkanContext::checkDeviceExtensionSupport(VkPhysicalDevice dev) const
    {
        uint32_t count;
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> available(count);
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, available.data());

        std::set <std::string> required(DEVICE_EXTENSIONS.begin(), DEVICE_EXTENSIONS.end());
        for (auto& ext : available) required.erase(ext.extensionName);
        return required.empty();
    }

    // Queue families
    QueueFamilyIndices VulkanContext::findQueueFamilies(VkPhysicalDevice dev) const
    {
        QueueFamilyIndices indices;
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, families.data());

        for (uint32_t i = 0; i < count; ++i) {
            if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                indices.graphicsFamily = i;

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &presentSupport);
            if (presentSupport) indices.presentFamily = i;

            if (indices.isComplete()) break;
        }
        return indices;
    }

    // Logical device
    void VulkanContext::createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        std::set<uint32_t> uniqueFamilies = {
            indices.graphicsFamily.value(),
            indices.presentFamily.value()
        };

        float queuePriority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queueInfos;
        for (uint32_t family : uniqueFamilies) {
            VkDeviceQueueCreateInfo qi{};
            qi.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qi.queueFamilyIndex = family;
            qi.queueCount       = 1;
            qi.pQueuePriorities = &queuePriority;
            queueInfos.push_back(qi);
        }

        VkPhysicalDeviceFeatures features{};
        features.fillModeNonSolid = VK_TRUE;    // allows wireframe rendering

        VkDeviceCreateInfo createInfo{};
        createInfo.sType                    = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount     = static_cast<uint32_t>(queueInfos.size());
        createInfo.pQueueCreateInfos        = queueInfos.data();
        createInfo.enabledExtensionCount    = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
        createInfo.ppEnabledExtensionNames  = DEVICE_EXTENSIONS.data();
        createInfo.pEnabledFeatures         = &features;

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device))
            throw std::runtime_error("Failed to create logical device");
        
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    // Swapchain
    SwapChainSupportDetails VulkanContext::querySwapChainSupport(VkPhysicalDevice dev) const
    {
        SwapChainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, surface, &details.capabilites);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &formatCount, nullptr);
        if (formatCount) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &formatCount, details.formats.data());
        }

        uint32_t modeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &modeCount, nullptr);
        if (modeCount) {
            details.presentModes.resize(modeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &modeCount, details.presentModes.data());
        }
        return details;
    }

    VkSurfaceFormatKHR VulkanContext::chooseSwapSurfaceFormat(
        const std::vector<VkSurfaceFormatKHR>& formats
    ) const
    {
        // Prefer BGRA8 SRGB - standard format for accurate color reproduction
        for (auto& f :formats)
            if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                return f;
        return formats[0];
    }

    VkPresentModeKHR VulkanContext::chooseSwapPresentMode(
        const std::vector<VkPresentModeKHR>& modes
    ) const
    {
        // Mailbox = trile buffering: low latency, no tearing. 60 FPS
        for (auto& m : modes)
            if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
        // FIFO = vsync, always available
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D VulkanContext::chooseSwapExtent(
        const VkSurfaceCapabilitiesKHR& caps, GLFWwindow* window
    ) const
    {
        if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max())
            return caps.currentExtent;

        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        VkExtent2D extent = { static_cast<uint32_t>(w), static_cast<uint32_t>(h) };
        extent.width    = std::clamp(extent.width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
        extent.height   = std::clamp(extent.height, caps.minImageExtent.height, caps.maxImageExtent.height);
        return extent;
    }

    void VulkanContext::createSwapchain(GLFWwindow* window)
    {
        SwapChainSupportDetails sc  = querySwapChainSupport(physicalDevice);
        VkSurfaceFormatKHR      fmt = chooseSwapSurfaceFormat(sc.formats);
        VkPresentModeKHR       mode = chooseSwapPresentMode(sc.presentModes);
        VkExtent2D              ext = chooseSwapExtent(sc.capabilites, window);

        uint32_t imageCount = sc.capabilites.minImageCount + 1;
        if (sc.capabilites.maxImageCount > 0)
            imageCount = std::min(imageCount, sc.capabilites.maxImageCount);

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface          = surface;
        createInfo.minImageCount    = imageCount;
        createInfo.imageFormat      = fmt.format;
        createInfo.imageColorSpace  = fmt.colorSpace;
        createInfo.imageExtent      = ext;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilies[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain) != VK_SUCCESS)
            throw std::runtime_error("Failed to create swapchain");

        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
        swapchainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());

        swapchainFormat = fmt.format;
        swapchainExtent = ext;
    }

    void VulkanContext::createImageViews()
    {
        swapchainImageViews.resize(swapchainImages.size());
        for (size_t i = 0; i < swapchainImages.size(); ++i)
            swapchainImageViews[i] = createImageView(swapchainImages[i], swapchainFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // Command pool
    void VulkanContext::createCommandPool()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        graphicsQueueFamily       = indices.graphicsFamily.value();
        poolInfo.queueFamilyIndex = graphicsQueueFamily;
        poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create command pool");
    }

    // Depth buffer
    VkFormat VulkanContext::findDepthFormat() const
    {
        for (VkFormat fmt : { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT }) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, fmt, &props);
            if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
                return fmt;
        }
        throw std::runtime_error("Failed to find supported depth format");
    }

    void VulkanContext::createDepthResources()
    {
        VkFormat depthFmt = findDepthFormat();
        createImage(swapchainExtent.width, swapchainExtent.height,
                    depthFmt,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFmt, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // Helpers
    void VulkanContext::createImage(uint32_t w, uint32_t h, VkFormat format,
                                    VkImageTiling tiling, VkImageUsageFlags usage,
                                    VkMemoryPropertyFlags properties,
                                    VkImage& image, VkDeviceMemory& memory)
    {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType     = VK_IMAGE_TYPE_2D;
        imgInfo.extent        = { w, h, 1 };
        imgInfo.mipLevels     = 1;
        imgInfo.arrayLayers   = 1;
        imgInfo.format        = format;
        imgInfo.tiling        = tiling;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imgInfo.usage         = usage;
        imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &imgInfo, nullptr, &image) != VK_SUCCESS)
            throw std::runtime_error("Failed to create image");
        
        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device, image, &memReq);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType             = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize    = memReq.size;
        allocInfo.memoryTypeIndex   = findMemoryType(memReq.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate image memory");

        vkBindImageMemory(device, image, memory, 0);
    }

    VkImageView VulkanContext::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags)
    {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image                           = image;
        viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format                          = format;
        viewInfo.subresourceRange.aspectMask     = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel   = 0;
        viewInfo.subresourceRange.levelCount     = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount     = 1;

        VkImageView view;
        if (vkCreateImageView(device, &viewInfo, nullptr, &view) != VK_SUCCESS)
            throw std::runtime_error("Failed to create image view");
        return view;
    }

    uint32_t VulkanContext::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const
    {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
            if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        throw std::runtime_error("Failed to find suitable memory type");
    }

    void VulkanContext::recreateSwapchain(GLFWwindow* window)
    {
        // Handle minimization
        int w = 0, h = 0;
        while (w == 0 || h == 0) {
            glfwGetFramebufferSize(window, &w, &h);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device);

        // Destroy old swapchain resources
        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);
        for (auto& iv : swapchainImageViews) vkDestroyImageView(device, iv, nullptr);
        vkDestroySwapchainKHR(device, swapchain, nullptr);

        // Recreate
        createSwapchain(window);
        createImageViews();
        createDepthResources();
    }

    void VulkanContext::destroy()
    {
        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
        for (auto& iv : swapchainImageViews) vkDestroyImageView(device, iv, nullptr);
        vkDestroySwapchainKHR(device, swapchain, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
    }
}