#pragma once

#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <glm/glm.hpp>

namespace acuity {
    // Enumeration
    enum class LODMethod {
        Geometric,      // distance based QEM selector
        Perceptual,     // ONNX perception model selector
        Oracle          // always LOD 0 : maximum quality reference
    };

    // Data structures
    struct CameraKeyframe {
        glm::vec3 position;
        glm::vec3 target;
        float     timestamp;    // seconds from start of path
    };

    struct BenchmarkConfig {
        std::string         sceneName;
        std::string         meshPath;
        LODMethod           method       = LODMethod::Geometric;
        float               duration     = 60.0f;            // seconds
        int                 repetition   = 0;                // 0-based rep index
        std::vector<CameraKeyframe> cameraPath;               // interpolated each frame
        std::string         outputDir    = "results/benchmarks";
    };

    struct FrameMetrics {
        float     frameTimeMs;       // wall-clock ms for this frame
        float     fps;               // 1000 / frameTimeMs
        int       selectedLOD;       // 0-4
        int       triangleCount;     // triangles rendered this frame
        float     lodSelectionMs;    // time spent in LOD selection
        float     renderMs;          // time spent in Vulkan draw
    };

    struct BenchmarkResult {
        // Identity
        std::string   sceneName;
        std::string   methodName;
        int           repetition;
        float         durationSeconds;

        // Aggregated performance
        float         avgFPS;
        float         minFPS;
        float         maxFPS;
        float         avgFrameTimeMs;
        float         p95FrameTimeMs;         // 95th percentile

        // Memory / geometry
        float         avgTriangles;
        float         minTriangles;
        float         maxTriangles;
        uint64_t      gpuMemoryBytes;         // VkDeviceMemory allocated

        // LOD distribution (how many frames spent at each LOD)
        int           lodDistribution[5];    // [lod0_frames .. lod4_frames]

        // LOD selection overhead
        float         avgLODSelectionMs;
        float         maxLODSelectionMs;

        // Raw per-frame data (for statistical analysis in Step 5.3)
        std::vector<FrameMetrics> frames;
    };

    // BenchmarkRunner
    class BenchmarkRunner {
        public:
            // Callbacks the app must wir up so BenchmarkRunner can drive the frame loop
            struct AppInterface {
                // Returns current frame metrics from the running renderer
                std::function<FrameMetrics()>          getFrameMetrics;
                // Sets the active LOD method
                std::function<void(LODMethod)>         setLODMethod;
                // Moves the camera to the given position/target
                std::function<void(glm::vec3, glm::vec3)> setCamera;
                // Returns GPU memory currently allocated (bytes)
                std::function<uint64_t()>              getGPUMemory;
                // Captures a screenshot to the given path (PNG)
                std::function<void(const std::string&)> captureScreenshot;
                // Advance one rendered frame and return false when the window is closed
                std::function<bool()>                  tickFrame;
            };

            explicit BenchmarkRunner(AppInterface iface);

            // Run a single benchmark configuration and return its result
            BenchmarkResult run(const BenchmarkConfig& cfg);

            // Run the full benchmark matrix (all scenes × methods × repetitions)
            // Results are written to cfg.outputDir as JSON files
            void runAll(const std::vector<BenchmarkConfig>& configs);

            // Write a BenchmarkResult to JSON
            static void writeJSON(const BenchmarkResult& result, const std::string& path);

            // Write all results to a single summary CSV
            static void writeCSV(const std::vector<BenchmarkResult>& results,
                                const std::string& path);

            // Build the predefined camera paths for all 4 scenes
            static std::vector<CameraKeyframe> buildCameraPath(const std::string& sceneName,
                                                                float duration);
                                                            
            // Convert LODMethod enum to string for JSON/CSV output
            static std::string methodName(LODMethod m);
            
        private:
            AppInterface m_iface;

            // Interpolate camera position along the path at time t
            static void interpolateCamera(const std::vector<CameraKeyframe>& path, float t,
                                          glm::vec3& outPos, glm::vec3& outTarget);

            // Compute 95th percentile of a sorted float vector
            static float percentile95(std::vector<float> v);
    };
}