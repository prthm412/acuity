#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <array>

namespace acuity {
    // Vertex: one point in our mesh with position, normal, and color.
    // Layout here must exactly match what the vertex shader expects
    // at location = 0, 1, 2
    struct Vertex {
        glm::vec3 position; // location 0
        glm::vec3 normal;   // location 1
        glm::vec3 color;    // location 2

        // Binding description: one buffer, one vertex at a time (per-vertex rate).
        static VkVertexInputBindingDescription getBindingDescription();

        // Attribute descriptions: where each field is within the vertex struct.
        static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions();
    };

    class Mesh {
        public:
            // CPU-side data
            std::vector<Vertex>     vertices;
            std::vector<uint32_t>   indices;

            // GPU-side buffers (created by uploadToGPU)
            VkBuffer        vertexBuffer =          VK_NULL_HANDLE;
            VkDeviceMemory  vertexBufferMemory =    VK_NULL_HANDLE;
            VkBuffer        indexBuffer =           VK_NULL_HANDLE;
            VkDeviceMemory  indexBufferMemory =     VK_NULL_HANDLE;

            // Load mesh from file using Assimp
            bool loadFromFile(const std::string& path);

            // Upload CPU data to GPU buffers
            void uploadToGPU(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue);

            // Free GPU resources
            void destroy(VkDevice device);

            // Set all vertices to a given color (used for LOD color coding)
            void setColor(glm::vec3 color);

            uint32_t getIndexCount()  const { return static_cast<uint32_t>(indices.size()); }
            uint32_t getVertexCount() const { return static_cast<uint32_t>(vertices.size()); }

        private:
            void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);

            void copyBuffer(VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

            uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);
    };
}