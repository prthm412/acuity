#include "BenchmarkRunner.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace acuity {
    // Constructor
    BenchmarkRunner::BenchmarkRunner(AppInterface iface) : m_iface(std::move(iface)) {}

    // methodName
    std::string BenchmarkRunner::methodName(LODMethod m) {
        switch (m)
        {
        case LODMethod::Geometric:  return "Geometric";
        case LODMethod::Perceptual: return "Perceptual";
        case LODMethod::Oracle:     return "Oracle";
        }
        return "Unknown";
    }

    // buildCameraPath
    std::vector<CameraKeyframe> BenchmarkRunner::buildCameraPath(
        const std::string& sceneName, float duration
    )
    {
        // Each scene gets a circular orbit path with 8 keyframes.
        // The path starts and ends at the same point (closed loop).
        // Distances and heights are tuned per scene so the mesh fills
        // a reasonable portion of the screen throughout the orbit.

        struct SceneParams {
            glm::vec3 centre;
            float     orbitRadius;
            float     height;
        };

        SceneParams p;
        if      (sceneName == "Bunny")      p = { {0,0,0},  3.0f,  1.0f };
        else if (sceneName == "Dragon")     p = { {0,0,0},  4.0f,  1.5f };
        else if (sceneName == "Sponza")     p = { {0,5,0},  12.0f, 4.0f };
        else if (sceneName == "SanMiguel")  p = { {0,3,0},  10.0f, 3.0f };
        else                                p = { {0,0,0},  5.0f,  2.0f };

        const int N = 8;
        std::vector<CameraKeyframe> path;
        path.reserve(N + 1);

        for (int i = 0; i <= N; ++i) {
            float angle = static_cast<float>(i) / N * 2.0f * static_cast<float>(M_PI);
            CameraKeyframe kf;
            kf.position  = glm::vec3(
                p.centre.x + p.orbitRadius * std::cos(angle),
                p.centre.y + p.height,
                p.centre.z + p.orbitRadius * std::sin(angle));
            kf.target    = p.centre;
            kf.timestamp = duration * static_cast<float>(i) / N;
            path.push_back(kf);
        }
        return path;
    }

    // interpolateCamera
    void BenchmarkRunner::interpolateCamera(
        const std::vector<CameraKeyframe>& path,
        float t,
        glm::vec3& outPos,
        glm::vec3& outTarget)
    {
        if (path.empty()) return;
        if (t <= path.front().timestamp) {
            outPos    = path.front().position;
            outTarget = path.front().target;
            return;
        }
        if (t >= path.back().timestamp) {
            outPos    = path.back().position;
            outTarget = path.back().target;
            return;
        }

        // Find surrounding keyframes
        for (size_t i = 0; i + 1 < path.size(); ++i) {
            if (t >= path[i].timestamp && t < path[i + 1].timestamp) {
                float span  = path[i + 1].timestamp - path[i].timestamp;
                float alpha = (t - path[i].timestamp) / span;
                outPos    = glm::mix(path[i].position,  path[i + 1].position,  alpha);
                outTarget = glm::mix(path[i].target,    path[i + 1].target,    alpha);
                return;
            }
        }
    }

    // percentile95
    float BenchmarkRunner::percentile95(std::vector<float> v) {
        if (v.empty()) return 0.0f;
        std::sort(v.begin(), v.end());
        size_t idx = static_cast<size_t>(0.95f * (v.size() - 1));
        return v[idx];
    }

    // run
    BenchmarkResult BenchmarkRunner::run(const BenchmarkConfig& cfg) {
        using Clock = std::chrono::high_resolution_clock;

        std::cout << "[Benchmark] Starting: " << cfg.sceneName
                << " | " << methodName(cfg.method)
                << " | rep " << cfg.repetition << "\n";

        m_iface.setLODMethod(cfg.method);

        BenchmarkResult result;
        result.sceneName       = cfg.sceneName;
        result.methodName      = methodName(cfg.method);
        result.repetition      = cfg.repetition;
        result.durationSeconds = cfg.duration;
        result.lodDistribution[0] = result.lodDistribution[1] =
        result.lodDistribution[2] = result.lodDistribution[3] =
        result.lodDistribution[4] = 0;

        // Warm-up phase (2 seconds)
        auto warmStart = Clock::now();
        while (std::chrono::duration<float>(Clock::now() - warmStart).count() < 2.0f) {
            glm::vec3 pos, tgt;
            interpolateCamera(cfg.cameraPath, 0.0f, pos, tgt);
            m_iface.setCamera(pos, tgt);
            if (!m_iface.tickFrame()) return result;
        }

        // Measurement phase
        auto measStart = Clock::now();
        while (true) {
            float elapsed = std::chrono::duration<float>(Clock::now() - measStart).count();
            if (elapsed >= cfg.duration) break;

            glm::vec3 pos, tgt;
            interpolateCamera(cfg.cameraPath, elapsed, pos, tgt);
            m_iface.setCamera(pos, tgt);

            if (!m_iface.tickFrame()) break;

            FrameMetrics fm = m_iface.getFrameMetrics();
            result.frames.push_back(fm);

            if (fm.selectedLOD >= 0 && fm.selectedLOD < 5)
                result.lodDistribution[fm.selectedLOD]++;
        }

        // Screenshot
        std::ostringstream ss;
        ss << cfg.outputDir << "/screenshot_"
        << cfg.sceneName << "_"
        << methodName(cfg.method) << "_rep"
        << cfg.repetition << ".png";
        m_iface.captureScreenshot(ss.str());

        // Aggregate
        if (!result.frames.empty()) {
            std::vector<float> fpsList, ftList, triList, lodMs;
            for (const auto& f : result.frames) {
                fpsList.push_back(f.fps);
                ftList.push_back(f.frameTimeMs);
                triList.push_back(static_cast<float>(f.triangleCount));
                lodMs.push_back(f.lodSelectionMs);
            }
            result.avgFPS          = std::accumulate(fpsList.begin(), fpsList.end(), 0.0f) / fpsList.size();
            result.minFPS          = *std::min_element(fpsList.begin(), fpsList.end());
            result.maxFPS          = *std::max_element(fpsList.begin(), fpsList.end());
            result.avgFrameTimeMs  = std::accumulate(ftList.begin(), ftList.end(), 0.0f) / ftList.size();
            result.p95FrameTimeMs  = percentile95(ftList);
            result.avgTriangles    = std::accumulate(triList.begin(), triList.end(), 0.0f) / triList.size();
            result.minTriangles    = *std::min_element(triList.begin(), triList.end());
            result.maxTriangles    = *std::max_element(triList.begin(), triList.end());
            result.gpuMemoryBytes  = m_iface.getGPUMemory();
            result.avgLODSelectionMs = std::accumulate(lodMs.begin(), lodMs.end(), 0.0f) / lodMs.size();
            result.maxLODSelectionMs = *std::max_element(lodMs.begin(), lodMs.end());
        }

        std::cout << "[Benchmark] Done: avgFPS=" << result.avgFPS
                << " avgTri=" << result.avgTriangles
                << " avgLOD_ms=" << result.avgLODSelectionMs << "\n";

        return result;
    }

    // runAll
    void BenchmarkRunner::runAll(const std::vector<BenchmarkConfig>& configs) {
        std::vector<BenchmarkResult> allResults;

        for (const auto& cfg : configs) {
            BenchmarkResult r = run(cfg);

            // Write per-run JSON
            std::ostringstream path;
            path << cfg.outputDir << "/"
                << cfg.sceneName << "_"
                << methodName(cfg.method) << "_rep"
                << cfg.repetition << ".json";
            writeJSON(r, path.str());

            allResults.push_back(std::move(r));
        }

        // Write summary CSV
        writeCSV(allResults, configs.empty() ? "results/benchmarks/summary.csv"
                                            : configs[0].outputDir + "/summary.csv");

        std::cout << "[Benchmark] All runs complete. "
                << allResults.size() << " results written.\n";
    }

    // writeJSON
    void BenchmarkRunner::writeJSON(const BenchmarkResult& r, const std::string& path) {
        std::ofstream f(path);
        if (!f) {
            std::cerr << "[Benchmark] Cannot write JSON: " << path << "\n";
            return;
        }

        auto q = [](const std::string& s) -> std::string {
            return "\"" + s + "\"";
        };

        f << std::fixed << std::setprecision(4);
        f << "{\n";
        f << "  " << q("scene")          << ": " << q(r.sceneName)         << ",\n";
        f << "  " << q("method")         << ": " << q(r.methodName)        << ",\n";
        f << "  " << q("repetition")     << ": " << r.repetition            << ",\n";
        f << "  " << q("duration_s")     << ": " << r.durationSeconds       << ",\n";
        f << "  " << q("avg_fps")        << ": " << r.avgFPS                << ",\n";
        f << "  " << q("min_fps")        << ": " << r.minFPS                << ",\n";
        f << "  " << q("max_fps")        << ": " << r.maxFPS                << ",\n";
        f << "  " << q("avg_frame_ms")   << ": " << r.avgFrameTimeMs        << ",\n";
        f << "  " << q("p95_frame_ms")   << ": " << r.p95FrameTimeMs        << ",\n";
        f << "  " << q("avg_triangles")  << ": " << r.avgTriangles          << ",\n";
        f << "  " << q("min_triangles")  << ": " << r.minTriangles          << ",\n";
        f << "  " << q("max_triangles")  << ": " << r.maxTriangles          << ",\n";
        f << "  " << q("gpu_memory_bytes")<< ": " << r.gpuMemoryBytes       << ",\n";
        f << "  " << q("avg_lod_sel_ms") << ": " << r.avgLODSelectionMs     << ",\n";
        f << "  " << q("max_lod_sel_ms") << ": " << r.maxLODSelectionMs     << ",\n";
        f << "  " << q("lod_distribution") << ": ["
        << r.lodDistribution[0] << ","
        << r.lodDistribution[1] << ","
        << r.lodDistribution[2] << ","
        << r.lodDistribution[3] << ","
        << r.lodDistribution[4] << "],\n";

        // Per-frame data
        f << "  " << q("frames") << ": [\n";
        for (size_t i = 0; i < r.frames.size(); ++i) {
            const auto& fm = r.frames[i];
            f << "    {";
            f << q("ft_ms")  << ":" << fm.frameTimeMs    << ",";
            f << q("fps")    << ":" << fm.fps             << ",";
            f << q("lod")    << ":" << fm.selectedLOD     << ",";
            f << q("tris")   << ":" << fm.triangleCount   << ",";
            f << q("lod_ms") << ":" << fm.lodSelectionMs  << ",";
            f << q("rnd_ms") << ":" << fm.renderMs;
            f << "}";
            if (i + 1 < r.frames.size()) f << ",";
            f << "\n";
        }
        f << "  ]\n";
        f << "}\n";

        std::cout << "[Benchmark] JSON written: " << path << "\n";
    }

    // writeCSV
    void BenchmarkRunner::writeCSV(const std::vector<BenchmarkResult>& results,
                                   const std::string& path)
    {
        std::ofstream f(path);
        if (!f) {
            std::cerr << "[Benchmark] Cannot write CSV: " << path << "\n";
            return;
        }

        f << "scene,method,repetition,avg_fps,min_fps,max_fps,"
        << "avg_frame_ms,p95_frame_ms,avg_triangles,min_triangles,max_triangles,"
        << "gpu_memory_bytes,avg_lod_sel_ms,max_lod_sel_ms,"
        << "lod0_frames,lod1_frames,lod2_frames,lod3_frames,lod4_frames\n";

        f << std::fixed << std::setprecision(4);
        for (const auto& r : results) {
            f << r.sceneName       << ","
            << r.methodName      << ","
            << r.repetition      << ","
            << r.avgFPS          << ","
            << r.minFPS          << ","
            << r.maxFPS          << ","
            << r.avgFrameTimeMs  << ","
            << r.p95FrameTimeMs  << ","
            << r.avgTriangles    << ","
            << r.minTriangles    << ","
            << r.maxTriangles    << ","
            << r.gpuMemoryBytes  << ","
            << r.avgLODSelectionMs << ","
            << r.maxLODSelectionMs << ","
            << r.lodDistribution[0] << ","
            << r.lodDistribution[1] << ","
            << r.lodDistribution[2] << ","
            << r.lodDistribution[3] << ","
            << r.lodDistribution[4] << "\n";
        }

        std::cout << "[Benchmark] CSV written: " << path << "\n";
    }
}