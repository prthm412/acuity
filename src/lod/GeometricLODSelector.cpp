#include "GeometricLODSelector.h"
#include <algorithm>

namespace acuity {
    float GeometricLODSelector::getDistance(const glm::vec3& cameraPos,
                                            const glm::vec3& objectPos) const
    {
        return glm::length(cameraPos - objectPos);
    }

    int GeometricLODSelector::selectLOD(const glm::vec3& cameraPos,
                                        const glm::vec3& objectPos,
                                        const LODMesh&   lodMesh) const
    {
        float dist      = getDistance(cameraPos, objectPos);
        int   numLevels = static_cast<int>(lodMesh.levelCount());

        // Walk through thresholds and return first level whose threshold exceeds distance
        for (int i = 0; i < numLevels - 1; ++i) {
            if (dist < distanceThresholds[i])
                return i;
        }
        // Beyond all thresholds: use lowest detail level
        return numLevels - 1;
    }
}