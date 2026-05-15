#pragma once

#include <array>
#include <vector>
#include <string>
#include <glm/glm.hpp>

namespace Acuity {
    static constexpr int TOTAL_FEATURES      = 38;
    static constexpr int GEOMETRIC_FEATURES  = 10;
    static constexpr int PERCEPTUAL_FEATURES = 15;
    static constexpr int VIEW_FEATURES       = 13;

    struct MeshData {
        std::vector<glm::vec3> vertices;
        std::vector<glm::vec3> normals;
        std::vector<uint32_t>  indices;
        int                    lodLevel;      // 0-4
        int                    triangleCount;
        int                    vertexCount;
    };

    struct ViewData {
        float distance;
        float azimuth;
        float elevation;
        float screenCoverage;
        glm::vec3 cameraPos;
        glm::vec3 meshCenter;
    };

    class FeatureExtractor {
        public:
            FeatureExtractor();

            // Load StandardScaler stats from a txt file
            // (exported alongside feature_scaler.pkl)
            bool loadScaler(const std::string& scalerPath);

            // Extract and normalize all 38 features
            // Returns array ready to pass directly to ONNXInference::predict()
            std::array<float, TOTAL_FEATURES> extract(const MeshData& mesh, const ViewData& view) const;

            // Extract raw (unnormalized features; useful for debugging)
            std::array<float, TOTAL_FEATURES> extractRaw(const MeshData& mesh, const ViewData& view) const;

            bool isScalerLoaded() const { return m_scalerLoaded; }

        private:
            // Scaler parameters
            std::array<float, TOTAL_FEATURES> m_mean{};
            std::array<float, TOTAL_FEATURES> m_std{};
            bool m_scalerLoaded = false;

            // Feature group extractors
            void extractGeometric(const MeshData& mesh, float* out) const; // writes [0-9]
            void extractPerceptual(const MeshData& mesh, float* out) const; // writes [10-24]
            void extractView(const MeshData& mesh, const ViewData& view, float* out) const; // writes [25-37]

            // Geometry helpers
            float computeSurfaceArea(const MeshData& mesh) const;
            float computeVolume(const MeshData& mesh) const;
            float computeMeanEdgeLength(const MeshData& mesh) const;
            float computeStdEdgeLength(const MeshData& mesh) const;
            float computeMeanCurvature(const MeshData& mesh) const;
            float computeNormalVariation(const MeshData& mesh) const;
            glm::vec3 computeCenter(const MeshData& mesh) const;
            float computeBoundingRadius(const MeshData& mesh) const;
    };
}