#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace acuity {
    // Mouse drag = rotate, scroll wheel = zoom, middle drag = pan
    class OrbitalCamera {
        public:
            OrbitalCamera();

            // Call every fram with the window dimension to get matrices
            glm::mat4 getViewMatrix()   const;
            glm::mat4 getProjectionMatrix(float aspectRatio) const;
            glm::vec3 getPosition()     const;

            // Input handlers - called from the GLFW callbacks in main.cpp
            void onMouseMove(float deltaX, float deltaY, bool leftBtn, bool middleBtn);
            void onMouseScroll(float yOffset);
            void onKeyPress(int key);

            // Reset to default view
            void reset();

            // Camera parameters (public for ImGui debug panel later)
            float theta;        // horizontal angle in radians
            float phi;          // vertical angle in radians
            float radius;       // distance from target
            glm::vec3 target;   // point the camera orbits around
            float fov;          // field of view in degrees
            float nearPlane;
            float farPlane;
        
        private:
            glm::vec3 computePosition() const;
    };
}