#pragma once

#include <glm/glm.hpp>
#include <array>

namespace acuity {
    struct Plane {
        glm::vec3 normal;
        float     distance;
    };

    class FrustumCuller {
        public:
            FrustumCuller() = default;

            // Extract frustum planes from combined view-projection matrix
            // Call once per frame before any visibility tests
            void update(const glm::mat4& viewProjection);

            // Returns true if the sphere (center, radius) intersects or
            // is inside the frustum. Returns false if fully outside.
            bool isSphereVisible(const glm::vec3& center, float radius) const;

            // Returns true if a point is inside the frustum
            bool isPointVisible(const glm::vec3& point) const;

        private:
            // 6 planes; left, right, bottom, top, near, far
            std::array<Plane, 6> m_planes;

            // Normalize a plane so the normal has unit length
            static Plane normalizePlane(const Plane& p);
    };
}