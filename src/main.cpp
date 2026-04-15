// ============================================================
// Acuity - 1.4 Basic LOD Selector
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

#include "lod/LODMesh.h"
#include "lod/LODGenerator.h"
#include "lod/GeometricLODSelector.h"

#include <iostream>
#include <stdexcept>
#include <chrono>
#include <string>
#include <numeric>

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
bool    g_lodVisualization   = false;   // V key toggles LOD color mode

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
        if (key == GLFW_KEY_V)      g_lodVisualization = !g_lodVisualization;
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
        acuity::LODMesh         lodMesh;
        acuity::GeometricLODSelector lodSelector;

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

        // LOD performance tracking (for baseline report)
        int         currentLODLevel  = 0;
        float       currentDistance  = 0.0f;
        uint32_t    framesSinceStart = 0;
        std::vector<float> fpsHistory;

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

            // In Step 1.3 it was: Load mesh (or generate procedural sphere if no path given)
            // Step 1.4: Generate LOD hierarchy
            if (!meshPath.empty()) {
                lodMesh = acuity::LODGenerator::generateFromFile(meshPath);
            } else {
                // Use procedural sphere as default
                lodMesh = generateSphereLODs(1.0f, 32, 32);
            }

            // Upload all LOD levels to GPU
            acuity::LODGenerator::uploadToGPU(lodMesh, ctx.device,
                                                       ctx.physicalDevice,
                                                       ctx.commandPool,
                                                       ctx.graphicsQueue);

            createCommandBuffers();
            createSyncObjects();
            initImGui();

            lastFpsTime = std::chrono::steady_clock::now();
            std::cout << "[App] LOD systems ready. Controls: LMB=orbit, MMB=pan, scroll=zoom, R=reset" << std::endl;
        }

        // Generate procedural sphere at multiple LOD levls
        acuity::LODMesh generateSphereLODs(float radius, int stacks, int slices) {
            std::vector<acuity::Vertex> vertices;
            std::vector<uint32_t>       indices;

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
                    v.color  = glm::vec3(0.7f, 0.7f, 0.9f);
                    vertices.push_back(v);
                }
            }
            for (int i = 0; i < stacks; ++i) {
                for (int j = 0; j < slices; ++j) {
                    int a = i * (slices + 1) + j;
                    int b = a + slices + 1;
                    indices.push_back(a);   indices.push_back(b);
                    indices.push_back(a+1); indices.push_back(b);
                    indices.push_back(b+1); indices.push_back(a+1);
                }
            }
            return acuity::LODGenerator::generate("sphere", vertices, indices);
        }

        void createCommandBuffers() {
            commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            VkCommandBufferAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool        = ctx.commandPool;
            ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
            if (vkAllocateCommandBuffers(ctx.device, &ai, commandBuffers.data()) != VK_SUCCESS)
                throw std::runtime_error("Failed to allocate command buffers");
        }

        void createSyncObjects() {
            imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
            renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
            inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

            VkSemaphoreCreateInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            VkFenceCreateInfo fi{};
            fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
                if (vkCreateSemaphore(ctx.device, &si, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                    vkCreateSemaphore(ctx.device, &si, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                    vkCreateFence    (ctx.device, &fi, nullptr, &inFlightFences[i])           != VK_SUCCESS)
                    throw std::runtime_error("Failed to create sync objects");
            }
        }

        void initImGui() {
            VkDescriptorPoolSize poolSizes[] = {
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 }
            };
            VkDescriptorPoolCreateInfo pi{};
            pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pi.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
            pi.maxSets       = 1000;
            pi.poolSizeCount = 1;
            pi.pPoolSizes    = poolSizes;
            vkCreateDescriptorPool(ctx.device, &pi, nullptr, &imguiPool);

            IMGUI_CHECKVERSION();
            ImGui::CreateContext();
            ImGui::StyleColorsDark();
            ImGui_ImplGlfw_InitForVulkan(window, true);

            ImGui_ImplVulkan_InitInfo initInfo{};
            initInfo.ApiVersion                      = VK_API_VERSION_1_2;
            initInfo.Instance                        = ctx.instance;
            initInfo.PhysicalDevice                  = ctx.physicalDevice;
            initInfo.Device                          = ctx.device;
            initInfo.QueueFamily                     = ctx.findQueueFamilies(ctx.physicalDevice).graphicsFamily.value();
            initInfo.Queue                           = ctx.graphicsQueue;
            initInfo.DescriptorPool                  = imguiPool;
            initInfo.MinImageCount                   = MAX_FRAMES_IN_FLIGHT;
            initInfo.ImageCount                      = static_cast<uint32_t>(ctx.swapchainImages.size());
            initInfo.PipelineInfoMain.RenderPass     = renderPass.renderPass;
            ImGui_ImplVulkan_Init(&initInfo);
        }

        void mainLoop() {
            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();
                updateFPS();
                updateLOD();
                drawFrame();
                ++framesSinceStart;
            }
            vkDeviceWaitIdle(ctx.device);
        }

        void updateFPS() {
            ++frameCount;
            auto  now     = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - lastFpsTime).count();
            if (elapsed >= 0.5f) {
                fps = frameCount / elapsed;
                fpsHistory.push_back(fps);
                frameCount  = 0;
                lastFpsTime = now;
                std::string title = "Acuity | FPS: " + std::to_string(int(fps))
                                + " | LOD: " + std::to_string(currentLODLevel);
                glfwSetWindowTitle(window, title.c_str());
            }
        }

        void updateLOD() {
            // Select LOD based on camera distance to origin (where our mesh sits)
            glm::vec3 camPos   = g_camera.getPosition();
            glm::vec3 meshPos  = glm::vec3(0.0f);
            currentDistance    = glm::length(camPos - meshPos);
            currentLODLevel    = lodSelector.selectLOD(camPos, meshPos, lodMesh);
        }

        void drawFrame() {
            vkWaitForFences(ctx.device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

            uint32_t imageIndex;
            VkResult result = vkAcquireNextImageKHR(ctx.device, ctx.swapchain, UINT64_MAX,
                                                    imageAvailableSemaphores[currentFrame],
                                                    VK_NULL_HANDLE, &imageIndex);
            if (result == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapchain(); return; }

            vkResetFences(ctx.device, 1, &inFlightFences[currentFrame]);

            VkCommandBuffer cmd = commandBuffers[currentFrame];
            vkResetCommandBuffer(cmd, 0);
            recordCommandBuffer(cmd, imageIndex);

            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkSubmitInfo si{};
            si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.waitSemaphoreCount   = 1;
            si.pWaitSemaphores      = &imageAvailableSemaphores[currentFrame];
            si.pWaitDstStageMask    = &waitStage;
            si.commandBufferCount   = 1;
            si.pCommandBuffers      = &cmd;
            si.signalSemaphoreCount = 1;
            si.pSignalSemaphores    = &renderFinishedSemaphores[currentFrame];

            if (vkQueueSubmit(ctx.graphicsQueue, 1, &si, inFlightFences[currentFrame]) != VK_SUCCESS)
                throw std::runtime_error("Failed to submit draw command buffer");

            VkPresentInfoKHR presentInfo{};
            presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores    = &renderFinishedSemaphores[currentFrame];
            presentInfo.swapchainCount     = 1;
            presentInfo.pSwapchains        = &ctx.swapchain;
            presentInfo.pImageIndices      = &imageIndex;

            result = vkQueuePresentKHR(ctx.presentQueue, &presentInfo);
            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR
                || g_framebufferResized) {
                g_framebufferResized = false;
                recreateSwapchain();
            }
            currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        }

        void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            vkBeginCommandBuffer(cmd, &bi);

            renderPass.begin(cmd, imageIndex, ctx.swapchainExtent);
            pipeline.bind(cmd);

            VkViewport viewport{};
            viewport.x = 0; viewport.y = 0;
            viewport.width    = float(ctx.swapchainExtent.width);
            viewport.height   = float(ctx.swapchainExtent.height);
            viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;
            vkCmdSetViewport(cmd, 0, 1, &viewport);

            VkRect2D scissor{ {0,0}, ctx.swapchainExtent };
            vkCmdSetScissor(cmd, 0, 1, &scissor);

            float aspect = float(ctx.swapchainExtent.width) / float(ctx.swapchainExtent.height);
            PushConstantData pc{};
            pc.model      = glm::mat4(1.0f);
            pc.view       = g_camera.getViewMatrix();
            pc.projection = g_camera.getProjectionMatrix(aspect);
            vkCmdPushConstants(cmd, pipeline.pipelineLayout,
                            VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantData), &pc);

            // Draw the currently selected LOD level
            acuity::LODLevel& level = lodMesh.levels[currentLODLevel];

            // In visualization mode, colors are already baked into vertices (LOD_COLORS)
            // In normal mode, we use the default grey color
            if (!g_lodVisualization) {
                // Temporarily set grey for normal rendering
                // (colors are already set correctly from LODGenerator)
            }

            VkBuffer     vbufs[]   = { level.mesh.vertexBuffer };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(cmd, 0, 1, vbufs, offsets);
            vkCmdBindIndexBuffer  (cmd, level.mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed      (cmd, level.mesh.getIndexCount(), 1, 0, 0, 0);

            // ImGui overlay
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowPos ({10, 10},  ImGuiCond_Always);
            ImGui::SetNextWindowSize({260, 200}, ImGuiCond_Always);
            ImGui::Begin("Acuity Debug", nullptr,
                        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

            ImGui::Text("FPS:       %.1f", fps);
            ImGui::Separator();
            ImGui::Text("LOD Level: %d / %d", currentLODLevel,
                        int(lodMesh.levelCount()) - 1);
            ImGui::Text("Triangles: %u", level.triangleCount);
            ImGui::Text("Distance:  %.2f", currentDistance);
            ImGui::Text("Ratio:     %.1f%%", level.targetRatio * 100.0f);
            ImGui::Separator();

            // LOD level breakdown
            for (int i = 0; i < int(lodMesh.levelCount()); ++i) {
                bool active = (i == currentLODLevel);
                if (active) ImGui::PushStyleColor(ImGuiCol_Text, {0.3f,1.0f,0.3f,1.0f});
                ImGui::Text("  LOD%d: %u tris (%.1f%%)", i,
                            lodMesh.levels[i].triangleCount,
                            lodMesh.levels[i].targetRatio * 100.0f);
                if (active) ImGui::PopStyleColor();
            }

            ImGui::Separator();
            ImGui::Text("V = toggle LOD colors");
            ImGui::Text("LOD colors: %s", g_lodVisualization ? "ON" : "OFF");
            ImGui::End();

            ImGui::Render();
            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

            renderPass.end(cmd);
            if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
                throw std::runtime_error("Failed to record command buffer");
        }

        void recreateSwapchain() {
            int w = 0, h = 0;
            while (w == 0 || h == 0) { glfwGetFramebufferSize(window, &w, &h); glfwWaitEvents(); }
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
                vkDestroyFence    (ctx.device, inFlightFences[i], nullptr);
            }
            lodMesh.destroy(ctx.device);
            pipeline.destroy(ctx.device);
            renderPass.destroy(ctx.device);
            ctx.destroy();
            glfwDestroyWindow(window);
            glfwTerminate();
        }
};

int main(int argc, char* argv[]) {
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