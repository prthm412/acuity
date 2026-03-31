#include "Mesh.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace acuity {
    // Vertex layout descriptions
    VkVertexInputBindingDescription Vertex::getBindingDescription()
    {
        VkVertexInputBindingDescription desc{};
        desc.binding    = 0;
        desc.stride     = sizeof(Vertex);               // bytes between vertices
        desc.inputRate  = VK_VERTEX_INPUT_RATE_VERTEX;
        return desc;
    }

    std::array<VkVertexInputAttributeDescription, 3> Vertex::getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 3> attrs{};

        // position at 0, format = R32G32B32 = vec3 of floats
        attrs[0].binding = 0;
        attrs[0].location = 0;
        attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset = offsetof(Vertex, position);

        // normal at location 1
        attrs[1].binding = 0;
        attrs[1].location = 1;
        attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[1].offset = offsetof(Vertex, normal);

        // color at location 2
        attrs[2].binding = 0;
        attrs[2].location = 2;
        attrs[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[2].offset = offsetof(Vertex, color);
        
        return attrs;
    }

    // Mesh loading
    bool Mesh::loadFromFile(const std::string& path)
    {
        Assimp::Importer importer;

        // Triangulate: convert quads/polygons to triangles
        // GenNormals:  generate normals if mesh doesn't have them
        // FlipUVs:     flip texture coordinates for Vulkan's top-left origin
        // JoinIdenticalVertices: merge duplicate vertices to reduce index count
        const aiScene* scene = importer.ReadFile(path,
            aiProcess_Triangulate | 
            aiProcess_GenSmoothNormals | 
            aiProcess_FlipUVs | 
            aiProcess_JoinIdenticalVertices);
        
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            std::cerr << "[Mesh] Failed to load: " << importer.GetErrorString() << std::endl;
            return false;
        }

        vertices.clear();
        indices.clear();

        // Process all meshes in the scene (merge them into one)
        for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
            aiMesh* mesh = scene->mMeshes[m];
            uint32_t vertexOffset = static_cast<uint32_t>(vertices.size());

            for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
                Vertex v{};
                v.position = { mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z };
                v.normal   = mesh->HasNormals()
                                ? glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z)
                                : glm::vec3(0.0f, 1.0f, 0.0f);
                v.color    = glm::vec3(0.8f, 0.8f, 0.8f); // default light grey
                vertices.push_back(v);
            }
            
            for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
                aiFace& face = mesh->mFaces[i];
                for (unsigned int j = 0; j < face.mNumIndices; ++j) {
                    indices.push_back(vertexOffset + face.mIndices[j]);
                }
            }
        }

        std::cout << "[Mesh] Loaded: " << path
                  << " | Vertices: " << vertices.size()
                  << " | Triangles: " << indices.size() / 3 << std::endl;
        return true;
    }

    // GPU upload
    void Mesh::uploadToGPU(VkDevice device, VkPhysicalDevice physicalDevice,
                        VkCommandPool commandPool, VkQueue graphicsQueue)
    {
        // Vertex buffer
        VkDeviceSize vbSize = sizeof(Vertex) * vertices.size();

        // Staging buffer: CPU-visible, used as transfer source
        VkBuffer stagingVB; VkDeviceMemory stagingVBMem;
        createBuffer(device, physicalDevice, vbSize,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    stagingVB, stagingVBMem);

        // Copy vertex data into staging buffer
        void* data;
        vkMapMemory(device, stagingVBMem, 0, vbSize, 0, &data);
        memcpy(data, vertices.data(), vbSize);
        vkUnmapMemory(device, stagingVBMem);

        // Device-local buffer: GPU-fast, used as vertex input
        createBuffer(device, physicalDevice, vbSize,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    vertexBuffer, vertexBufferMemory);

        copyBuffer(device, commandPool, graphicsQueue, stagingVB, vertexBuffer, vbSize);

        vkDestroyBuffer(device, stagingVB, nullptr);
        vkFreeMemory(device, stagingVBMem, nullptr);

        // Index buffer
        VkDeviceSize ibSize = sizeof(uint32_t) * indices.size();

        VkBuffer stagingIB; VkDeviceMemory stagingIBMem;
        createBuffer(device, physicalDevice, ibSize,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    stagingIB, stagingIBMem);

        vkMapMemory(device, stagingIBMem, 0, ibSize, 0, &data);
        memcpy(data, indices.data(), ibSize);
        vkUnmapMemory(device, stagingIBMem);

        createBuffer(device, physicalDevice, ibSize,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    indexBuffer, indexBufferMemory);

        copyBuffer(device, commandPool, graphicsQueue, stagingIB, indexBuffer, ibSize);

        vkDestroyBuffer(device, stagingIB, nullptr);
        vkFreeMemory(device, stagingIBMem, nullptr);
    }

    void Mesh::setColor(glm::vec3 color)
    {
        for (auto& v : vertices) v.color = color;
    }

    void Mesh::destroy(VkDevice device)
    {
        if (indexBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, indexBuffer, nullptr);
        if (indexBufferMemory != VK_NULL_HANDLE) vkFreeMemory(device, indexBufferMemory, nullptr);
        if (vertexBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, vertexBuffer, nullptr);
        if (vertexBufferMemory != VK_NULL_HANDLE) vkFreeMemory(device, vertexBufferMemory, nullptr);
    }

    // Helper: create buffer
    void Mesh::createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, 
                         VkDeviceSize size, VkBufferUsageFlags usage,
                         VkMemoryPropertyFlags properties,
                         VkBuffer& buffer, VkDeviceMemory& bufferMemory)
    {
        VkBufferCreateInfo bufInfo{};
        bufInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size        = size;
        bufInfo.usage       = usage;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufInfo, nullptr, &buffer) != VK_SUCCESS)
            throw std::runtime_error("Failed to create buffer");
        
        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(device, buffer, &memReq);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType             = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize    = memReq.size;
        allocInfo.memoryTypeIndex   = findMemoryType(physicalDevice, memReq.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate buffer memory");

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    // Helper: copy buffer via command buffer
    void Mesh::copyBuffer(VkDevice device, VkCommandPool commandPool,
                       VkQueue graphicsQueue,
                       VkBuffer src, VkBuffer dst, VkDeviceSize size)
    {
        // Allocate a temporary command buffer for the copy operation
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool        = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer cmdBuf;
        vkAllocateCommandBuffers(device, &allocInfo, &cmdBuf);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(cmdBuf, &beginInfo);
        VkBufferCopy copyRegion{ 0, 0, size };
        vkCmdCopyBuffer(cmdBuf, src, dst, 1, &copyRegion);
        vkEndCommandBuffer(cmdBuf);

        VkSubmitInfo submitInfo{};
        submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &cmdBuf;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);
        vkFreeCommandBuffers(device, commandPool, 1, &cmdBuf);
    }

    // Helper: find suitable GPU memory type
    uint32_t Mesh::findMemoryType(VkPhysicalDevice physicalDevice,
                               uint32_t typeFilter,
                               VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((typeFilter & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        }
        throw std::runtime_error("Failed to find suitable memory type");
    }
}