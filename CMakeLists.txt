cmake_minimum_required(VERSION 3.20)
project(Acuity VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "vcpkg toolchain file")
endif()

# ── Find packages ──────────────────────────────────────────────
find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

# ── Source files ───────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/renderer/VulkanContext.cpp
    src/renderer/RenderPass.cpp
    src/renderer/Pipeline.cpp
    src/renderer/Mesh.cpp
    src/renderer/Camera.cpp
    src/renderer/OffscreenRenderer.cpp
    src/renderer/BatchRenderer.cpp
    src/lod/QuadricSimplifier.cpp
    src/lod/LODGenerator.cpp
    src/lod/GeometricLODSelector.cpp
    src/ml/ONNXInference.cpp
    src/lod/FeatureExtractor.cpp
)

add_executable(Acuity ${SOURCES})

target_include_directories(Acuity PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
    $<TARGET_PROPERTY:onnxruntime::onnxruntime,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(Acuity PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
    assimp::assimp
    imgui::imgui
    onnxruntime::onnxruntime
)

target_compile_definitions(Acuity PRIVATE
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)

# ── Shader compilation ─────────────────────────────────────────
# glslc compiles GLSL source to SPIR-V binary (.spv)
# The compiled shaders go into the build/shaders/ directory
# so the executable finds them at runtime with path "shaders/mesh.vert.spv"
find_program(GLSLC glslc HINTS
    "$ENV{VULKAN_SDK}/Bin"
    "C:/VulkanSDK/1.4.341.1/Bin")

if(GLSLC)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)

    add_custom_command(
        OUTPUT  ${CMAKE_BINARY_DIR}/shaders/mesh.vert.spv
        COMMAND ${GLSLC} ${CMAKE_SOURCE_DIR}/assets/shaders/mesh.vert
                         -o ${CMAKE_BINARY_DIR}/shaders/mesh.vert.spv
        DEPENDS ${CMAKE_SOURCE_DIR}/assets/shaders/mesh.vert
        COMMENT "Compiling vertex shader")

    add_custom_command(
        OUTPUT  ${CMAKE_BINARY_DIR}/shaders/mesh.frag.spv
        COMMAND ${GLSLC} ${CMAKE_SOURCE_DIR}/assets/shaders/mesh.frag
                         -o ${CMAKE_BINARY_DIR}/shaders/mesh.frag.spv
        DEPENDS ${CMAKE_SOURCE_DIR}/assets/shaders/mesh.frag
        COMMENT "Compiling fragment shader")

    add_custom_target(Shaders ALL DEPENDS
        ${CMAKE_BINARY_DIR}/shaders/mesh.vert.spv
        ${CMAKE_BINARY_DIR}/shaders/mesh.frag.spv)

    add_dependencies(Acuity Shaders)
    message(STATUS "glslc found: ${GLSLC} - shaders will auto-compile")
else()
    message(WARNING "glslc not found - compile shaders manually")
endif()

message(STATUS "Vulkan: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Acuity configured successfully")