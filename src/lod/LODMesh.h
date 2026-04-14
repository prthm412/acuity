#pragma once
#include "renderer/Mesh.h"
#include <vector>
#include <string>

namespace acuity {
    // One LOD level: a simplified mesh + its metadata
    struct LODLevel {
        Mesh        mesh;
        float       targetRatio;
        uint32_t    triangleCount;
        glm::vec3   debugColor;
    };

    // All LOD levels for one 3D asset
    struct LODMesh {
        std::string           name;
        std::vector<LODLevel> levels;    // levels[0] = highest quality, levels[N-1] = lowest quality

        size_t levelCount() const { return levels.size(); }

        // Free all GPU resources for every level
        void destroy(VkDevice device) {
            for (auto& l : levels) l.mesh.destroy(device);
        }
    };
}