#include "FrustumCuller.h"
#include <cmath>

namespace acuity {
    // Extracts 6 frustum planes using Gribb/Hartmann method
    // Each plane is extracted from a row combination of the VP matrix
    void FrustumCuller::update(const glm::mat4& vp) {
        // Left plane: col3 + col0
        m_planes[0] = normalizePlane({
            glm::vec3(vp[0][3] + vp[0][0],
                    vp[1][3] + vp[1][0],
                    vp[2][3] + vp[2][0]),
            vp[3][3] + vp[3][0]
        });

        // Right plane:  col3 - col0
        m_planes[1] = normalizePlane({
            glm::vec3(vp[0][3] - vp[0][0],
                    vp[1][3] - vp[1][0],
                    vp[2][3] - vp[2][0]),
            vp[3][3] - vp[3][0]
        });

        // Bottom plane: col3 + col1
        m_planes[2] = normalizePlane({
            glm::vec3(vp[0][3] + vp[0][1],
                    vp[1][3] + vp[1][1],
                    vp[2][3] + vp[2][1]),
            vp[3][3] + vp[3][1]
        });

        // Top plane:    col3 - col1
        m_planes[3] = normalizePlane({
            glm::vec3(vp[0][3] - vp[0][1],
                    vp[1][3] - vp[1][1],
                    vp[2][3] - vp[2][1]),
            vp[3][3] - vp[3][1]
        });

        // Near plane:   col3 + col2
        m_planes[4] = normalizePlane({
            glm::vec3(vp[0][3] + vp[0][2],
                    vp[1][3] + vp[1][2],
                    vp[2][3] + vp[2][2]),
            vp[3][3] + vp[3][2]
        });

        // Far plane:    col3 - col2
        m_planes[5] = normalizePlane({
            glm::vec3(vp[0][3] - vp[0][2],
                    vp[1][3] - vp[1][2],
                    vp[2][3] - vp[2][2]),
            vp[3][3] - vp[3][2]
        });
    }

    // isSphereVisible
    // Tests sphere against all 6 planes
    // If signed distance to any plane < -radius, sphere is outside
    bool FrustumCuller::isSphereVisible(const glm::vec3& center, float radius) const
    {
        for (const auto& plane : m_planes) {
            float dist = glm::dot(plane.normal, center) + plane.distance;
            if (dist < -radius)
                return false;
        }
        return true;
    }

    // isPointVisible
    bool FrustumCuller::isPointVisible(const glm::vec3& point) const
    {
        return isSphereVisible(point, 0.0f);
    }

    // normalizePlane
    Plane FrustumCuller::normalizePlane(const Plane& p)
    {
        float len = glm::length(p.normal);
        if (len < 1e-8f) return p;
        return { p.normal / len, p.distance / len };
    }
}