// ============================================================
// Acuity - Build Verification (Step 1.2)
// This file verifies all dependencies are linked correctly.
// Will be replaced in Step 1.3 with the Vulkan renderer.
// ============================================================

// GLFW + Vulkan: GLFW is our window/input library.
// GLFW_INCLUDE_VULKAN tells GLFW to include Vulkan headers automatically
// instead of OpenGL, since we are building a Vulkan renderer.
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// GLM: OpenGL Mathematics library. Used throughout the project for
// vectors (glm::vec3), matrices (glm::mat4), transforms, etc.
// It is a header-only library, so including it is enough to use it.
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Assimp: The Open Asset Import Library. Used in Step 1.3 onward
// to load 3D mesh files (.obj, .ply, .fbx, etc.) into our renderer.
#include <assimp/Importer.hpp>
#include <assimp/scene.h>

// ImGui: Immediate Mode GUI library. Used for the debug UI panel
// in Step 4.4 — switching LOD methods, viewing stats, etc.
#include <imgui.h>

// ONNX Runtime: The inference engine that runs our trained
// perception model (exported as .onnx) inside the C++ renderer.
// This is the bridge between the Python training and C++ runtime.
#include <onnxruntime_cxx_api.h>

// Standard library
#include <iostream>
#include <string>
#include <vector>

int main()
{
    std::cout << "=== Acuity Build Verification ===" << std::endl;

    // ── Test 1: GLFW (window system) ──────────────────────────
    if (glfwInit()) {
        std::cout << "[PASS] GLFW initialized" << std::endl;
        glfwTerminate();
    } else {
        std::cerr << "[FAIL] GLFW failed to initialize" << std::endl;
        return 1;
    }

    // ── Test 2: Vulkan availability ───────────────────────────
    // glfwVulkanSupported() checks if the Vulkan loader is present
    // on the system. Requires GLFW to be initialized first.
    glfwInit();
    if (glfwVulkanSupported()) {
        std::cout << "[PASS] Vulkan is supported on this system" << std::endl;
    } else {
        std::cerr << "[FAIL] Vulkan not supported" << std::endl;
        return 1;
    }
    glfwTerminate();

    // ── Test 3: GLM (math library) ────────────────────────────
    // A simple sanity check: create a 3D vector and transform it.
    // If GLM headers are broken, this won't compile at all.
    glm::vec3 testVec(1.0f, 2.0f, 3.0f);
    glm::mat4 identity = glm::mat4(1.0f);
    glm::vec4 result   = identity * glm::vec4(testVec, 1.0f);
    std::cout << "[PASS] GLM working - vector: ("
              << result.x << ", " << result.y << ", " << result.z << ")"
              << std::endl;

    // ── Test 4: Assimp (mesh loader) ──────────────────────────
    // Just instantiate the importer. If Assimp linked correctly,
    // this will succeed without loading any actual file.
    Assimp::Importer importer;
    std::cout << "[PASS] Assimp importer created" << std::endl;

    // ── Test 5: ONNX Runtime (ML inference engine) ────────────
    // Create an ONNX Runtime environment object.
    // ORT_LOGGING_LEVEL_WARNING suppresses verbose startup logs.
    Ort::Env ortEnv(ORT_LOGGING_LEVEL_WARNING, "AcuityTest");
    std::cout << "[PASS] ONNX Runtime environment created" << std::endl;

    // ── All tests passed ──────────────────────────────────────
    std::cout << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "  All systems operational!" << std::endl;
    std::cout << "  Environment setup complete." << std::endl;
    std::cout << "  Ready for Step 1.3 (Vulkan Renderer)." << std::endl;
    std::cout << "==========================================" << std::endl;

    return 0;
}