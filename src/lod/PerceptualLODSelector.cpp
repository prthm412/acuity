#define _USE_MATH_DEFINES
#include <cmath>

#include "PerceptualLODSelector.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace Acuity {
    // Constructor
    PerceptualLODSelector::PerceptualLODSelector() : m_model(nullptr) // model path is set in init()
    {

    }

    // init
    bool PerceptualLODSelector::init(const std::string& modelPath, const std::string& scalerPath)
    {
        // Load scaler first
        if (!m_extractor.loadScaler(scalerPath)) {
            std::cerr << "[PerceptualLODSelector] Failed to load scaler\n";
            return false;
        }

        // Model is already constructed with empty path in member init
        // Reconstruct with actual path using placement approach:
        // reload by constructing a new ONNXInference in place
        try {
            m_model = std::make_unique<ONNXInference>(modelPath);
            new (&m_model) ONNXInference(modelPath);
        } catch (const std::exception& e) {
            std::cerr << "[PerceptualLODSelector] Failed to load model: " << e.what() << "\n";
            return false;
        }

        m_ready = true;
        std::cout << "[PerceptualLODSelector] Ready\n";
        return true;
    }

    // selectLOD
    int PerceptualLODSelector::selectLOD(
        const std::string& meshId, const std::vector<MeshData>& lodMeshes, const ViewData& view
    )
    {
        if (!m_ready || lodMeshes.empty()) return 0;

        // Check cache
        std::string key = makeCacheKey(meshId, view);
        auto it = m_cache.find(key);
        if (it != m_cache.end())
            return it->second;

        // Score all LOD levels
        m_lastCandidates.clear();
        for (int lod = 0; lod < static_cast<int>(lodMeshes.size()); ++lod) {
            auto features = m_extractor.extract(lodMeshes[lod], view);
            float score   = m_model->predict(features);

            LODCandidate c;
            c.lodLevel      = lod;
            c.qualityScore  = score;
            c.triangleCount = lodMeshes[lod].triangleCount;
            m_lastCandidates.push_back(c);
        }

        // Select: highest LOD index (lowest detail) with score >= threshold
        // means: use the most simplified mesh that still looks good enough
        int selected = 0;
        for (int lod = static_cast<int>(m_lastCandidates.size()) -1; lod >= 0; --lod) {
            if (m_lastCandidates[lod].qualityScore >= m_threshold) {
                selected = lod;
                break;
            }
        }

        // Store in cache
        m_cache[key] = selected;
        return selected;
    }

    // makeCacheKey
    // Quantizes view distance to nearest 0.5 units so small camera
    // movements don't invalidate the cache every frame
    std::string PerceptualLODSelector::makeCacheKey(
        const std::string& meshId, const ViewData& view
    ) const {
        // Quantize distance to 0.5 unit buckets
        int distBucket = static_cast<int>(view.distance / 0.5f);

        std::ostringstream ss;
        ss << meshId << "_" << distBucket;
        return ss.str();
    }
}