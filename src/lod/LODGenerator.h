#pragma once
#include "LODMesh.h"
#include "renderer/Mesh.h"
#include <vector>
#include <string>

namespace acuity {
    class LODGenerator {
        public:
            // LOD level definitions
            // Ratios: 100%, 50%, 25%, 12.5%, 6.25%
            static constexpr float LOD_RATIOS[5] = {
                1.0f, 0.5f, 0.25f, 0.125f, 0.0625f
            };

            // Debug colors for LOD visualization (green=high quality, red=low quality)
            static constexpr glm::vec3 LOD_COLORS[5] = {
                {0.2f, 0.8f, 0.2f},     // LOD 0 green
                {0.8f, 0.8f, 0.2f},     // LOD 1 yellow
                {0.8f, 0.5f, 0.1f},     // LOD 2 orange
                {0.8f, 0.2f, 0.1f},     // LOD 3 red-orange
                {0.6f, 0.1f, 0.1f},     // LOD 4 dark-red
            };

            // Generate 5 LOD Levels
            static LODMesh generate(
                const std::string&           meshName,
                const std::vector<Vertex>&   vertices,
                const std::vector<uint32_t>& indices
            );

            // Convenience: load from file and generate LODs
            static LODMesh generateFromFile(const std::string& filePath);

            // Upload all LOD levels to GPU
            static void uploadToGPU(
                LODMesh&            lodMesh,
                VkDevice            device,
                VkPhysicalDevice    physicalDevice,
                VkCommandPool       commandPool,
                VkQueue             graphicsQueue
            );
    };
}