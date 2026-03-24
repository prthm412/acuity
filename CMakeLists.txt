cmake_minimum_required(VERSION 3.20)
project(Acuity VERSION 0.1.0 LANGUAGES CXX)

# ─────────────────────────────────────────────
# C++ Standard
# ─────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─────────────────────────────────────────────
# vcpkg toolchain (must be set before find_package calls)
# ─────────────────────────────────────────────
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "vcpkg toolchain file")
endif()

# ─────────────────────────────────────────────
# Find packages
# ─────────────────────────────────────────────
find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

# ─────────────────────────────────────────────
# Source files
# ─────────────────────────────────────────────
set(SOURCES
    src/main.cpp
)

# ─────────────────────────────────────────────
# Executable
# ─────────────────────────────────────────────
add_executable(Acuity ${SOURCES})

# ─────────────────────────────────────────────
# Include directories
# ─────────────────────────────────────────────
target_include_directories(Acuity PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
)

# ─────────────────────────────────────────────
# Link libraries
# ─────────────────────────────────────────────
target_link_libraries(Acuity PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
    assimp::assimp
    imgui::imgui
    onnxruntime::onnxruntime
)

# ─────────────────────────────────────────────
# Compile definitions
# ─────────────────────────────────────────────
target_compile_definitions(Acuity PRIVATE
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)

# ─────────────────────────────────────────────
# Status messages
# ─────────────────────────────────────────────
message(STATUS "Vulkan found: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Acuity configured successfully")