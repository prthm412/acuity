#include "RenderPass.h"
#include "VulkanContext.h"
#include <stdexcept>
#include <array>

namespace acuity {
    void RenderPass::init(VulkanContext& ctx)
    {
        createRenderPass(ctx);
        createFramebuffers(ctx);
    }

    void RenderPass::recreate(VulkanContext& ctx)
    {
        destroy(ctx.device);
        init(ctx);
    }

    void RenderPass::createRenderPass(VulkanContext& ctx)
    {
        // Color attachment
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format         = ctx.swapchainFormat;
        colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;       // clear to black at start
        colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;      // keep result for display
        colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;   // ready for display

        // Depth attachment
        VkAttachmentDescription depthAttachment{};
        depthAttachment.format         = ctx.findDepthFormat();
        depthAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;  // don't need depth after frame
        depthAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorRef{};
        colorRef.attachment = 0;
        colorRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthRef{};
        depthRef.attachment = 1;
        depthRef.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        // Subpass
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount    = 1;
        subpass.pColorAttachments       = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef;

        // Subpass dependency
        VkSubpassDependency dependency{};
        dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass    = 0;
        dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };
        VkRenderPassCreateInfo rpInfo{};
        rpInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        rpInfo.pAttachments    = attachments.data();
        rpInfo.subpassCount    = 1;
        rpInfo.pSubpasses      = &subpass;
        rpInfo.dependencyCount = 1;
        rpInfo.pDependencies   = &dependency;

        if (vkCreateRenderPass(ctx.device, &rpInfo, nullptr, &renderPass) != VK_SUCCESS)
            throw std::runtime_error("Failed to create render pass");
    }

    void RenderPass::createFramebuffers(VulkanContext& ctx)
    {
        // One framebuffer per swapchain image
        // Each framebuffer binds a swapchain image view + depth image view
        framebuffers.resize(ctx.swapchainImageViews.size());
        for (size_t i = 0; i < ctx.swapchainImageViews.size(); ++i) {
            std::array<VkImageView, 2> attachments = {
                ctx.swapchainImageViews[i],
                ctx.depthImageView
            };

            VkFramebufferCreateInfo fbInfo{};
            fbInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fbInfo.renderPass      = renderPass;
            fbInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            fbInfo.pAttachments    = attachments.data();
            fbInfo.width           = ctx.swapchainExtent.width;
            fbInfo.height          = ctx.swapchainExtent.height;
            fbInfo.layers          = 1;

            if (vkCreateFramebuffer(ctx.device, &fbInfo, nullptr, &framebuffers[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create framebuffer");
        }
    }

    void RenderPass::begin(VkCommandBuffer cmd, uint32_t imageIndex, VkExtent2D extent)
    {
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color        = {{ 0.1f, 0.1f, 0.1f, 1.0f }}; // dark grey background
        clearValues[1].depthStencil = { 1.0f, 0 };                  // max depth = 1.0

        VkRenderPassBeginInfo rpBegin{};
        rpBegin.sType               = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBegin.renderPass          = renderPass;
        rpBegin.framebuffer         = framebuffers[imageIndex];
        rpBegin.renderArea.offset   = { 0, 0 };
        rpBegin.renderArea.extent   = extent;
        rpBegin.clearValueCount     = static_cast<uint32_t>(clearValues.size());
        rpBegin.pClearValues        = clearValues.data();

        vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
    }

    void RenderPass::end(VkCommandBuffer cmd)
    {
        vkCmdEndRenderPass(cmd);
    }

    void RenderPass::destroy(VkDevice device)
    {
        for (auto& fb : framebuffers) vkDestroyFramebuffer(device, fb, nullptr);
        framebuffers.clear();
        if (renderPass != VK_NULL_HANDLE) vkDestroyRenderPass(device, renderPass, nullptr);
    }
}