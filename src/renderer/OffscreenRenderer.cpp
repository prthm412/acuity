#include "OffscreenRenderer.h"
#include "VulkanContext.h"
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <array>

namespace acuity {
    // Public
    void OffscreenRenderer::init(VulkanContext& ctx, uint32_t w, uint32_t h)
    {
        width = w;
        height = h;

        createColorResources(ctx);
        createDepthResources(ctx);
        createRenderPass(ctx.device);
        createFramebuffer(ctx.device);
        createStagingBuffer(ctx);
        createCommandBuffer(ctx);

        // Fence for CPU-GPU sync
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(ctx.device, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
            throw std::runtime_error("OffscreenRenderer: failed to create fence");

        std::cout << "[OffscreenRenderer] Initialized " << w << "x" << h << std::endl;
    }

    void OffscreenRenderer::destroy(VkDevice device)
    {
        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyFramebuffer(device, framebuffer, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
        vkDestroyImageView(device, colorView, nullptr);
        vkDestroyImage(device, colorImage, nullptr);
        vkFreeMemory(device, colorMemory, nullptr);
        vkDestroyImageView(device, depthView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthMemory, nullptr);
    }

    void OffscreenRenderer::beginFrame()
    {
        vkResetCommandBuffer(commandBuffer, 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
    }

    void OffscreenRenderer::endFrame(VulkanContext& ctx)
    {
        // Transition color image to transfer source layout for readback
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = colorImage;
        barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        // Copy image to staging buffer
        VkBufferImageCopy region{};
        region.bufferOffset      = 0;
        region.bufferRowLength   = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset       = {0, 0, 0};
        region.imageExtent       = {width, height, 1};

        vkCmdCopyImageToBuffer(commandBuffer,
            colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            stagingBuffer, 1, &region);

        vkEndCommandBuffer(commandBuffer);

        // Submit and wait
        VkSubmitInfo submitInfo{};
        submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &commandBuffer;

        vkResetFences(ctx.device, 1, &fence);
        vkQueueSubmit(ctx.graphicsQueue, 1, &submitInfo, fence);
        vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
    }

    std::vector<uint8_t> OffscreenRenderer::readPixels(VulkanContext& ctx)
    {
        size_t imageSize = width * height * 4;
        std::vector<uint8_t> pixels(imageSize);

        void* data;
        vkMapMemory(ctx.device, stagingMemory, 0, imageSize, 0, &data);
        memcpy(pixels.data(), data, imageSize);
        vkUnmapMemory(ctx.device, stagingMemory);

        return pixels;
    }

    // Private
    void OffscreenRenderer::createColorResources(VulkanContext& ctx)
    {
        createImage(ctx, width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    colorImage, colorMemory);

        createImageView(ctx.device, colorImage, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_ASPECT_COLOR_BIT, colorView);
    }

    void OffscreenRenderer::createDepthResources(VulkanContext& ctx)
    {
        createImage(ctx, width, height,
                VK_FORMAT_D32_SFLOAT,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                depthImage, depthMemory);

        createImageView(ctx.device, depthImage,
                        VK_FORMAT_D32_SFLOAT,
                        VK_IMAGE_ASPECT_DEPTH_BIT, depthView);
    }

    void OffscreenRenderer::createRenderPass(VkDevice device)
    {
        // Color attachment
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format         = VK_FORMAT_R8G8B8A8_UNORM;
        colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        // Depth attachment
        VkAttachmentDescription depthAttachment{};
        depthAttachment.format         = VK_FORMAT_D32_SFLOAT;
        depthAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount    = 1;
        subpass.pColorAttachments       = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass    = 0;
        dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 2> attachments = {
            colorAttachment, depthAttachment
        };

        VkRenderPassCreateInfo rpInfo{};
        rpInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        rpInfo.pAttachments    = attachments.data();
        rpInfo.subpassCount    = 1;
        rpInfo.pSubpasses      = &subpass;
        rpInfo.dependencyCount = 1;
        rpInfo.pDependencies   = &dependency;

        if (vkCreateRenderPass(device, &rpInfo, nullptr, &renderPass) != VK_SUCCESS)
            throw std::runtime_error("OffscreenRenderer: failed to create render pass");
    }

    void OffscreenRenderer::createFramebuffer(VkDevice device)
    {
        std::array<VkImageView, 2> views = {colorView, depthView};

        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass      = renderPass;
        fbInfo.attachmentCount = static_cast<uint32_t>(views.size());
        fbInfo.pAttachments    = views.data();
        fbInfo.width           = width;
        fbInfo.height          = height;
        fbInfo.layers          = 1;

        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &framebuffer) != VK_SUCCESS)
            throw std::runtime_error("OffscreenRenderer: failed to create framebuffer");
    }

        void OffscreenRenderer::createStagingBuffer(VulkanContext& ctx)
    {
        VkDeviceSize size = width * height * 4;

        VkBufferCreateInfo bufInfo{};
        bufInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size        = size;
        bufInfo.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(ctx.device, &bufInfo, nullptr, &stagingBuffer) != VK_SUCCESS)
            throw std::runtime_error("OffscreenRenderer: failed to create staging buffer");

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(ctx.device, stagingBuffer, &memReqs);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize  = memReqs.size;
        allocInfo.memoryTypeIndex = findMemoryType(ctx.physicalDevice,
            memReqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(ctx.device, &allocInfo, nullptr, &stagingMemory) != VK_SUCCESS)
            throw std::runtime_error("OffscreenRenderer: failed to allocate staging memory");

        vkBindBufferMemory(ctx.device, stagingBuffer, stagingMemory, 0);
    }

    void OffscreenRenderer::createCommandBuffer(VulkanContext& ctx)
    {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = ctx.graphicsQueueFamily;
        poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
            throw std::runtime_error("OffscreenRenderer: failed to create command pool");

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool        = commandPool;
        allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(ctx.device, &allocInfo, &commandBuffer) != VK_SUCCESS)
            throw std::runtime_error("OffscreenRenderer: failed to allocate command buffer");
    }

    void OffscreenRenderer::createImage(VulkanContext& ctx,
                                        uint32_t w, uint32_t h,
                                        VkFormat format,
                                        VkImageUsageFlags usage,
                                        VkMemoryPropertyFlags memProps,
                                        VkImage& image,
                                        VkDeviceMemory& memory)
    {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType     = VK_IMAGE_TYPE_2D;
        imgInfo.extent        = {w, h, 1};
        imgInfo.mipLevels     = 1;
        imgInfo.arrayLayers   = 1;
        imgInfo.format        = format;
        imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imgInfo.usage         = usage;
        imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(ctx.device, &imgInfo, nullptr, &image) != VK_SUCCESS)
            throw std::runtime_error("OffscreenRenderer: failed to create image");

        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(ctx.device, image, &memReqs);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize  = memReqs.size;
        allocInfo.memoryTypeIndex = findMemoryType(ctx.physicalDevice,
                                                memReqs.memoryTypeBits, memProps);

        if (vkAllocateMemory(ctx.device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
            throw std::runtime_error("OffscreenRenderer: failed to allocate image memory");

        vkBindImageMemory(ctx.device, image, memory, 0);
    }

    void OffscreenRenderer::createImageView(VkDevice device,
                                            VkImage image,
                                            VkFormat format,
                                            VkImageAspectFlags aspect,
                                            VkImageView& view)
    {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image                           = image;
        viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format                          = format;
        viewInfo.subresourceRange.aspectMask     = aspect;
        viewInfo.subresourceRange.baseMipLevel   = 0;
        viewInfo.subresourceRange.levelCount     = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount     = 1;

        if (vkCreateImageView(device, &viewInfo, nullptr, &view) != VK_SUCCESS)
            throw std::runtime_error("OffscreenRenderer: failed to create image view");
    }

    uint32_t OffscreenRenderer::findMemoryType(VkPhysicalDevice physicalDevice,
                                            uint32_t typeBits,
                                            VkMemoryPropertyFlags props)
    {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if ((typeBits & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags & props) == props)
                return i;
        }
        throw std::runtime_error("OffscreenRenderer: failed to find memory type");
    }
}