#pragma once

#include "FeatureExtractor.h"
#include "../ml/ONNXInference.h"

#include <array>
#include <vector>
#include <string>
#include <unordered_map>

namespace acuity {
    static constexpr int NUM_LOD_LEVELS = 5;
    static constexpr float DEFAULT_THRESHOLD = 0.5f;

    struct LODCandidate {
        int lodLevel;
        float qualityScore;
        int triangleCount;
    };

    class PerceptualLODSelector {
        public:
            PerceptualLODSelector();

            // Initialize - load ONNX model and scaler
            bool init(const std::string& modelPath, const std::string& scalerPath);

            // Select best LOD level for a mesh given current view
            // meshId: unique identifier for caching
            // lodMeshes: one MeshData per LOD level (index = LOD level)
            // view: current camera/view state
            // Returns selected LOD index (0 = highest detail)
            int selectLOD(const std::string& meshId, const std::vector<MeshData>& lodMeshes, const ViewData& view);

            // Get all candidate scores from last selectLOD call
            const std::vector<LODCandidate>& getLastCandidates() const {
                return m_lastCandidates;
            }

            // Quality threshold: LOD selected if score >= threshold
            void setThreshold(float t) { m_threshold = t; }
            float getThreshold() const { return m_threshold; }

            // Clear spatial cache (call when camera moves significantly)
            void clearCache() { m_cache.clear(); }
            bool isReady() const { return m_ready; }

        private:
            FeatureExtractor    m_extractor;
            std::unique_ptr<ONNXInference> m_model;
            float               m_threshold = DEFAULT_THRESHOLD;
            bool                m_ready     = false;

            // Cache: meshId + quantized view key -> selected LOD
            std::unordered_map<std::string, int> m_cache;

            // Last frame candidates for debug UI
            std::vector<LODCandidate> m_lastCandidates;

            // Build cache key from mesh id and quantized view distance
            std::string makeCacheKey(const std::string& meshId, const ViewData& view) const;
    };
}