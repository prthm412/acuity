// ============================================================
// Acuity - Renderer
// This file verifies all dependencies are linked correctly.
// ============================================================

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include "renderer/Camera.h"
#include "renderer/Mesh.h"
#include "renderer/Pipeline.h"
#include "renderer/RenderPass.h"
#include "renderer/VulkanContext.h"

#include <iostream>
#include <stdexcept>
#include <chrono>
#include <string>

// Constants
constexpr uint32_t WINDOW_WIDTH  = 1280;
constexpr uint32_t WINDOW_HEIGHT = 720;
constexpr int      MAX_FRAMES_IN_FLIGHT = 2;    // double buffering on CPU side

// Push constant layout (must match shader)
struct PushConstantData {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 projection;
};

// Global state (accessed in GLFW callbacks)
acuity::OrbitalCamera g_camera;
bool    g_mouseLMB      = false;
bool    g_mouseMMB      = false;
double  g_lastMouseX    = 0.0;
double  g_lastMouseY    = 0.0;
bool    g_framebufferResized = false;

// GLFW callbacks
void framebufferResizeCallback(GLFWwindow*, int, int) {
    g_framebufferResized = true;
}

void mouseButtonCallback(GLFWwindow*, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT)   g_mouseLMB = (action == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_MIDDLE) g_mouseMMB = (action == GLFW_PRESS);
}

void cursorPosCallback(GLFWwindow*, double xpos, double ypos) {
    float dx = static_cast<float>(xpos - g_lastMouseX);
    float dy = static_cast<float>(ypos - g_lastMouseY);
    g_lastMouseX = xpos;
    g_lastMouseY = ypos;
    g_camera.onMouseMove(dx, dy, g_mouseLMB, g_mouseMMB);
}

void scrollCallback(GLFWwindow*, double, double yoffset) {
    g_camera.onMouseScroll(static_cast<float>(yoffset));
}

void keyCallback(GLFWwindow* window, int key, int, int action, int) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, GLFW_TRUE);
        g_camera.onKeyPress(key);
    }
}

// Main application class
class AcuityApp {
    public:
        void run(const std::string& meshPath) {
            initWindow();
            initVulkan(meshPath);
            mainLoop();
            cleanup();
        }
    
    private:
        GLFWwindow*             window = nullptr;
        acuity::VulkanContext   ctx;
        acuity::RenderPass      renderPass;
        acuity::Pipeline        pipeline;
        acuity::Mesh            mesh;

        // Per-frame synchronization objects
        std::vector<VkSemaphore> imageAvailableSemaphores;
        std::vector<VkSemaphore> renderFinishedSemaphores;
        std::vector<VkFence>     inFlightFences;
        std::vector<VkCommandBuffer> commandBuffers;
        uint32_t currentFrame = 0;

        // ImGui descriptor pool
        VkDescriptorPool imguiPool = VK_NULL_HANDLE;

        // FPS tracking
        uint32_t frameCount = 0;
        float    fps        = 0.0f;
        std::chrono::steady_clock::time_point lastFpsTime;

        void initWindow() {
            glfwInit();
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // No OpenGL context
            window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Acuity", nullptr, nullptr);
            glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
            glfwSetMouseButtonCallback(window, mouseButtonCallback);
            glfwSetCursorPosCallback(window, cursorPosCallback);
            glfwSetScrollCallback(window, scrollCallback);
            glfwSetKeyCallback(window, keyCallback);
            glfwGetCursorPos(window, &g_lastMouseX, &g_lastMouseY);
        }

        void initVulkan(const std::string& meshPath) {
            ctx.init(window);
            renderPass.init(ctx);
            pipeline.init(ctx, renderPass.renderPass, "shaders/mesh.vert.spv", "shaders/mesh.frag.spv");

            // Load mesh (or generate procedural sphere if no path given)
            if (!meshPath.empty()) {
                if (!mesh.loadFromFile(meshPath)) {
                    std::cerr << "[App] Mesh load failed, using procedural sphere" << std::endl;
                    generateSphere(1.0f, 32, 32);
                }
            } else {
                generateSphere(1.0f, 32, 32);
            }
            mesh.uploadToGPU(ctx.device, ctx.physicalDevice, ctx.commandPool, ctx.graphicsQueue);

            createCommandBuffers();
            createSyncObjects();
            initImGui();

            lastFpsTime = std::chrono::steady_clock::now();
            std::cout << "[App] Initialized. Controls: LMB=orbit, MMB=pan, scroll=zoom, R=reset" << std::endl;
        }

        void generateSphere(float radius, int stacks, int slices) {
            mesh.vertices.clear();
            mesh.indices.clear();

            for (int i = 0; i <= stacks; ++i) {
                float phi = glm::pi<float>() * i / stacks;
                for (int j = 0; j <= slices; ++j) {
                    float theta = 2.0f * glm::pi<float>() * j / slices;
                    acuity::Vertex v{};
                    v.position = {
                        radius * sin(phi) * cos(theta),
                        radius * cos(phi),
                        radius * sin(phi) * sin(theta)
                    };
                    v.normal = glm::normalize(v.position);
                    v.color  = { 0.7f, 0.7f, 0.9f };
                    mesh.vertices.push_back(v);
                }
            }
            for (int i = 0; i < stacks; ++i) {
                for (int j = 0; j < slices; ++j) {
                    int a = i * (slices + 1) + j;
                    int b = a + slices + 1;
                    mesh.indices.push_back(a);
                    mesh.indices.push_back(b);
                    mesh.indices.push_back(a + 1);
                    mesh.indices.push_back(b);
                    mesh.indices.push_back(b + 1);
                    mesh.indices.push_back(a + 1);
                }
            }
            std::cout << "[App] Procedural sphere: "
                      << mesh.vertices.size() << " vertices, "
                      << mesh.indices.size() / 3 << " triangles" << std::endl;
        }

        void createCommandBuffers() {
            commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool        = ctx.commandPool;
            allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
            if (vkAllocateCommandBuffers(ctx.device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
                throw std::runtime_error("Failed to allocate command buffers");
        }

        void createSyncObjects() {
            imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
            renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
            inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

            VkSemaphoreCreateInfo semInfo{};
            semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

            VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;     // start signaled so first frame does not hang

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
                if (vkCreateSemaphore(ctx.device, &semInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                    vkCreateSemaphore(ctx.device, &semInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                    vkCreateFence    (ctx.device, &fenceInfo, nullptr, &inFlightFences[i])         != VK_SUCCESS)
                    throw std::runtime_error("Failed to create sync objects");
            }
        }

        void initImGui() {
            // ImGui descriptor pool (needs a large pool for its own textures)
            VkDescriptorPoolSize poolSizes[] = {
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 }
            };
            VkDescriptorPoolCreateInfo poolInfo{};
            poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
            poolInfo.maxSets       = 1000;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes    = poolSizes;
            vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &imguiPool);

            IMGUI_CHECKVERSION();
            ImGui::CreateContext();
            ImGui::StyleColorsDark();

            ImGui_ImplGlfw_InitForVulkan(window, true);

            ImGui_ImplVulkan_InitInfo initInfo{};
            initInfo.ApiVersion         = VK_API_VERSION_1_2;
            initInfo.Instance           = ctx.instance;
            initInfo.PhysicalDevice     = ctx.physicalDevice;
            initInfo.Device             = ctx.device;
            initInfo.QueueFamily        = ctx.findQueueFamilies(ctx.physicalDevice).graphicsFamily.value();
            initInfo.Queue              = ctx.graphicsQueue;
            initInfo.DescriptorPool     = imguiPool;
            initInfo.MinImageCount      = MAX_FRAMES_IN_FLIGHT;
            initInfo.ImageCount         = static_cast<uint32_t>(ctx.swapchainImages.size());
            initInfo.PipelineInfoMain.RenderPass = renderPass.renderPass;
            ImGui_ImplVulkan_Init(&initInfo);
        }

        // Main loop
        void mainLoop() {
            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();
                updateFPS();
                drawFrame();
            }
            vkDeviceWaitIdle(ctx.device);
        }

        void updateFPS() {
            ++frameCount;
            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - lastFpsTime).count();
            if (elapsed >= 0.5f) {
                fps = frameCount / elapsed;
                frameCount = 0;
                lastFpsTime = now;
                std::string title = "Acuity | FPS: " + std::to_string(static_cast<int>(fps));
                glfwSetWindowTitle(window, title.c_str());
            }
        }

        void drawFrame() {
            // Wait for previous frame using this slot to finish
            vkWaitForFences(ctx.device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

            // Acquire next swapchain image
            uint32_t imageIndex;
            VkResult result = vkAcquireNextImageKHR(ctx.device, ctx.swapchain, UINT64_MAX,
                                                    imageAvailableSemaphores[currentFrame],
                                                    VK_NULL_HANDLE, &imageIndex);
            if (result == VK_ERROR_OUT_OF_DATE_KHR) {
                recreateSwapchain(); return;
            }

            vkResetFences(ctx.device, 1, &inFlightFences[currentFrame]);

            // Record commands for this frame
            VkCommandBuffer cmd = commandBuffers[currentFrame];
            vkResetCommandBuffer(cmd, 0);
            recordCommandBuffer(cmd, imageIndex);

            // submit to GPU
            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkSubmitInfo submitInfo{};
            submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.waitSemaphoreCount   = 1;
            submitInfo.pWaitSemaphores      = &imageAvailableSemaphores[currentFrame];
            submitInfo.pWaitDstStageMask    = &waitStage;
            submitInfo.commandBufferCount   = 1;
            submitInfo.pCommandBuffers      = &cmd;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores    = &renderFinishedSemaphores[currentFrame];

            if (vkQueueSubmit(ctx.graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
                throw std::runtime_error("Failed to submit draw command buffer");
            
            // Present rendered image to screen
            VkPresentInfoKHR presentInfo{};
            presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores    = &renderFinishedSemaphores[currentFrame];
            presentInfo.swapchainCount     = 1;
            presentInfo.pSwapchains        = &ctx.swapchain;
            presentInfo.pImageIndices      = &imageIndex;

            result = vkQueuePresentKHR(ctx.presentQueue, &presentInfo);
            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || g_framebufferResized) {
                g_framebufferResized = false;
                recreateSwapchain();
            }

            currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        }

        void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            vkBeginCommandBuffer(cmd, &beginInfo);

            renderPass.begin(cmd, imageIndex, ctx.swapchainExtent);

            pipeline.bind(cmd);

            // Dynamic viewport and scissor
            VkViewport viewport{};
            viewport.x        = 0.0f; viewport.y      = 0.0f;
            viewport.width    = static_cast<float>(ctx.swapchainExtent.width);
            viewport.height   = static_cast<float>(ctx.swapchainExtent.height);
            viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;
            vkCmdSetViewport(cmd, 0, 1, &viewport);

            VkRect2D scissor{ {0,0}, ctx.swapchainExtent };
            vkCmdSetScissor(cmd, 0, 1, &scissor);

            // Push MVP matrices as push constants
            float aspect = static_cast<float>(ctx.swapchainExtent.width) /
                        static_cast<float>(ctx.swapchainExtent.height);
            PushConstantData pc{};
            pc.model      = glm::mat4(1.0f);
            pc.view       = g_camera.getViewMatrix();
            pc.projection = g_camera.getProjectionMatrix(aspect);
            vkCmdPushConstants(cmd, pipeline.pipelineLayout,
                            VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantData), &pc);

            // Bind vertex and index buffers, draw
            VkBuffer     vbufs[]  = { mesh.vertexBuffer };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(cmd, 0, 1, vbufs, offsets);
            vkCmdBindIndexBuffer  (cmd, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed      (cmd, mesh.getIndexCount(), 1, 0, 0, 0);

            // ImGui overlay
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Always);
            ImGui::SetNextWindowSize({220, 120}, ImGuiCond_Always);
            ImGui::Begin("Acuity Debug", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
            ImGui::Text("FPS: %.1f", fps);
            ImGui::Text("Vertices:  %u", mesh.getVertexCount());
            ImGui::Text("Triangles: %u", mesh.getIndexCount() / 3);
            ImGui::Text("R = reset camera");
            ImGui::End();

            ImGui::Render();
            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

            renderPass.end(cmd);
            if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
                throw std::runtime_error("Failed to record command buffer");
        }

        void recreateSwapchain() {
            vkDeviceWaitIdle(ctx.device);
            renderPass.recreate(ctx);
            pipeline.destroy(ctx.device);
            pipeline.init(ctx, renderPass.renderPass,
                          "shaders/mesh.vert.spv", "shaders/mesh.frag.spv");
        }

        void cleanup() {
            ImGui_ImplVulkan_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
            vkDestroyDescriptorPool(ctx.device, imguiPool, nullptr);

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
                vkDestroySemaphore(ctx.device, imageAvailableSemaphores[i], nullptr);
                vkDestroySemaphore(ctx.device, renderFinishedSemaphores[i], nullptr);
                vkDestroyFence    (ctx.device, inFlightFences[i],           nullptr);
            }
            mesh.destroy(ctx.device);
            pipeline.destroy(ctx.device);
            renderPass.destroy(ctx.device);
            ctx.destroy();
            glfwDestroyWindow(window);
            glfwTerminate();
        }
};

int main(int argc, char* argv[])
{
    std::string meshPath = (argc > 1) ? argv[1] : "";
    AcuityApp app;
    try {
        app.run(meshPath);
    } catch (const std::exception& e) {
        std::cerr << "[Fatal] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}