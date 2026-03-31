#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>

// GLFW key codes
#include <GLFW/glfw3.h>

namespace acuity {
    OrbitalCamera::OrbitalCamera()
    {
        reset();
    }

    void OrbitalCamera::reset()
    {
        theta   = 0.0f;
        phi     = glm::radians(30.0f); // slightly above equator
        radius  = 3.0f;
        target  = glm::vec3(0.0f);
        fov     = 45.0f;
        nearPlane   = 0.01f;
        farPlane    = 100.0f;
    }

    glm::vec3 OrbitalCamera::computePosition() const
    {
        // Convert spherical (theta, phi, radius) to Cartesian (x, y, z)
        float x = radius * cos(phi) * sin(theta);
        float y = radius * sin(phi);
        float z = radius * cos(phi) * cos(theta);
        return target + glm::vec3(x, y, z);
    }

    glm::mat4 OrbitalCamera::getViewMatrix() const
    {
        // glm::lookAt builds a view matrix given camera position, target, and up vector
        return glm::lookAt(computePosition(), target, glm::vec3(0.0f, 1.0f, 0.0f));
    }

    glm::mat4 OrbitalCamera::getProjectionMatrix(float aspectRatio) const
    {
        // Perspective projection matrix
        // GLM_FORCE_DEPTH_ZERO_TO_ONE (set in CMake)
        // Vulkan-compatible depth range changes from [-1,1] to [0,1]
        glm::mat4 proj = glm::perspective(glm::radians(fov), aspectRatio, nearPlane, farPlane);
        // Vulkan's Y axis is flipped compared to OpenGL - flip it back
        proj[1][1] *= -1.0f;
        return proj;
    }

    glm::vec3 OrbitalCamera::getPosition() const
    {
        return computePosition();
    }
    
    void OrbitalCamera::onMouseMove(float deltaX, float deltaY, bool leftBtn, bool middleBtn)
    {
        if (leftBtn) {
            // Left drag: orbit around target
            theta -= deltaX * 0.005f;
            phi += deltaY * 0.005f;

            // Clamp phi to avoid gimbal lock at poles
            phi = std::clamp(phi, glm::radians(-89.0f), glm::radians(89.0f));
        }
        if (middleBtn) {
            // Middle drag: pan the target point
            glm::vec3 right = glm::normalize(glm::cross(target - computePosition(), glm::vec3(0, 1, 0)));
            glm::vec3 up    = glm::vec3(0.0f, 1.0f, 0.0f);
            target         -= right * deltaX * radius * 0.001f;
            target         += up    * deltaY * radius * 0.001f;
        }
    }

    void OrbitalCamera::onMouseScroll(float yOffset)
    {
        // Scroll: zoom in/out by adjusting radius
        radius -= yOffset * radius * 0.1f;
        radius  = std::clamp(radius, 0.1f, 500.0f);
    }

    void OrbitalCamera::onKeyPress(int key)
    {
        if (key == GLFW_KEY_R) reset();
    }
}