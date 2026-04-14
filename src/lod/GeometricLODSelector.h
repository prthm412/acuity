#pragma once
#include "LODMesh.h"
#include <glm/glm.hpp>

namespace acuity {
    class GeometricLODSelector {
        public:
            // Distance thresholds for switching LOD levels.
            // Below threshold[0] = LOD 0
            // Between [0] and [1] = LOD 1, etc.
            // Beyond threshold[4] = LOD 4 (lowest detail)
            float distanceThresholds[5] = {
                5.0f,   // LOD 0: < 5 units
                15.0f,  // LOD 1: 5 - 15 units
                30.0f,  // LOD 2: 15 - 30 units
                60.0f,  // LOD 3: 30 - 60 units
                1e9f    // LOD 4: > 60 units
            };
            
            // Select appropriate LOD level index given camera and object positions
            int selectLOD(const glm::vec3& cameraPos,
                          const glm::vec3& objectPos,
                          const LODMesh&   lodMesh) const;
            
            // Get distance between camera and object
            float getDistance(const glm::vec3& cameraPos,
                              const glm::vec3& objectPos) const;
    };
}