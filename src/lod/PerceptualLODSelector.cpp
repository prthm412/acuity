#define _USE_MATH_DEFINES
#include <cmath>

#include "PerceptualLODSelector.h"
#include <iostream>
#include <algorithm>
#include <sstream>

namespace acuity {
    // Constructor 

    PerceptualLODSelector::PerceptualLODSelector()
        : m_model(nullptr)
    {
    }

    // init 

    bool PerceptualLODSelector::init(const std::string& modelPath,
                                    const std::string& scalerPath)
    {
        if (!m_extractor.loadScaler(scalerPath)) {
            std::cerr << "[PerceptualLODSelector] Failed to load scaler\n";
            return false;
        }

        try {
            m_model = std::make_unique<ONNXInference>(modelPath);
        } catch (const std::exception& e) {
            std::cerr << "[PerceptualLODSelector] Failed to load model: "
                    << e.what() << "\n";
            return false;
        }

        m_ready = true;
        std::cout << "[PerceptualLODSelector] Ready\n";
        return true;
    }

    // selectLOD (single mesh)

    int PerceptualLODSelector::selectLOD(
        const std::string&           meshId,
        const std::vector<MeshData>& lodMeshes,
        const ViewData&              view)
    {
        if (!m_ready || lodMeshes.empty()) return 0;

        // Check cache
        std::string key = makeCacheKey(meshId, view);
        auto it = m_cache.find(key);
        if (it != m_cache.end()) {
            // Still apply temporal stability even on cache hit
            return applyTemporalStability(meshId, it->second);
        }

        // Score all LOD levels
        m_lastCandidates = scoreLODs(lodMeshes, view);

        // Select best LOD respecting threshold and memory budget
        int selected = selectFromCandidates(m_lastCandidates);

        // Apply temporal stability to prevent popping
        selected = applyTemporalStability(meshId, selected);

        // Store in cache
        m_cache[key] = selected;
        return selected;
    }

    // selectLODBatch
    // Batch inference: extracts all features first, then runs one
    // batched ONNX inference call instead of N individual calls.
    // This is more efficient when many meshes need LOD selection.

    std::vector<int> PerceptualLODSelector::selectLODBatch(
        const std::vector<std::string>&           meshIds,
        const std::vector<std::vector<MeshData>>& lodMeshesPerMesh,
        const std::vector<ViewData>&              views)
    {
        if (!m_ready) return std::vector<int>(meshIds.size(), 0);

        int numMeshes = static_cast<int>(meshIds.size());
        std::vector<int> results(numMeshes, 0);

        // Build flat feature batch: [numMeshes * NUM_LOD_LEVELS, FEATURE_DIM]
        std::vector<float> featureBatch;
        featureBatch.reserve(
            numMeshes * NUM_LOD_LEVELS * TOTAL_FEATURES);

        // Track which meshes need inference vs cache hits
        std::vector<bool>  needsInference(numMeshes, false);
        std::vector<int>   cachedResults(numMeshes, 0);

        for (int m = 0; m < numMeshes; ++m) {
            std::string key = makeCacheKey(meshIds[m], views[m]);
            auto it = m_cache.find(key);
            if (it != m_cache.end()) {
                cachedResults[m] = applyTemporalStability(
                    meshIds[m], it->second);
                results[m] = cachedResults[m];
                needsInference[m] = false;
            } else {
                needsInference[m] = true;
                // Extract features for all LOD levels of this mesh
                const auto& lodMeshes = lodMeshesPerMesh[m];
                for (int lod = 0; lod < (int)lodMeshes.size(); ++lod) {
                    auto features = m_extractor.extract(
                        lodMeshes[lod], views[m]);
                    for (float f : features)
                        featureBatch.push_back(f);
                    // Pad with zeros if fewer than NUM_LOD_LEVELS
                }
                // Pad if mesh has fewer LOD levels than NUM_LOD_LEVELS
                int numLODs = static_cast<int>(lodMeshes.size());
                for (int pad = numLODs; pad < NUM_LOD_LEVELS; ++pad) {
                    for (int f = 0; f < TOTAL_FEATURES; ++f)
                        featureBatch.push_back(0.0f);
                }
            }
        }

        // Count meshes needing inference
        int inferenceCount = 0;
        for (bool b : needsInference) if (b) ++inferenceCount;

        if (inferenceCount > 0) {
            // Run batch inference
            int batchSize = inferenceCount * NUM_LOD_LEVELS;
            auto scores = m_model->predictBatch(featureBatch, batchSize);

            // Map scores back to meshes
            int scoreIdx = 0;
            for (int m = 0; m < numMeshes; ++m) {
                if (!needsInference[m]) continue;

                const auto& lodMeshes = lodMeshesPerMesh[m];
                std::vector<LODCandidate> candidates;
                for (int lod = 0; lod < NUM_LOD_LEVELS; ++lod) {
                    if (lod < (int)lodMeshes.size()) {
                        LODCandidate c;
                        c.lodLevel      = lod;
                        c.qualityScore  = scores[scoreIdx];
                        c.triangleCount = lodMeshes[lod].triangleCount;
                        candidates.push_back(c);
                    }
                    ++scoreIdx;
                }

                int selected = selectFromCandidates(candidates);
                selected = applyTemporalStability(meshIds[m], selected);

                std::string key = makeCacheKey(meshIds[m], views[m]);
                m_cache[key] = selected;
                results[m]   = selected;
            }
        }

        return results;
    }

    // scoreLODs

    std::vector<LODCandidate> PerceptualLODSelector::scoreLODs(
        const std::vector<MeshData>& lodMeshes,
        const ViewData&              view)
    {
        std::vector<LODCandidate> candidates;
        for (int lod = 0; lod < static_cast<int>(lodMeshes.size()); ++lod) {
            auto  features = m_extractor.extract(lodMeshes[lod], view);
            float score    = m_model->predict(features);

            LODCandidate c;
            c.lodLevel      = lod;
            c.qualityScore  = score;
            c.triangleCount = lodMeshes[lod].triangleCount;
            candidates.push_back(c);
        }
        return candidates;
    }

    // selectFromCandidates
    // Walks from lowest detail (LOD4) to highest (LOD0).
    // Selects the lowest detail LOD whose quality score >= threshold
    // AND whose triangle count is within memory budget.

    int PerceptualLODSelector::selectFromCandidates(
        const std::vector<LODCandidate>& candidates)
    {
        int selected = 0; // fallback: highest detail

        for (int lod = static_cast<int>(candidates.size()) - 1;
            lod >= 0; --lod)
        {
            const auto& c = candidates[lod];

            // Check quality threshold
            if (c.qualityScore < m_threshold) continue;

            // Check memory budget
            if (m_memoryBudget != NO_MEMORY_BUDGET &&
                c.triangleCount > m_memoryBudget) continue;

            selected = lod;
            break;
        }

        return selected;
    }

    // applyTemporalStability
    // Prevents rapid LOD switching (visual popping) by requiring the
    // system to commit to a new LOD for at least m_stabilityFrames
    // frames before switching again.
    //
    // When a new target LOD is computed:
    // - If it matches current LOD: increment frames counter, stay
    // - If it differs: only switch if we've been stable long enough

    int PerceptualLODSelector::applyTemporalStability(
        const std::string& meshId, int targetLOD)
    {
        auto& state = m_temporalStates[meshId];

        if (state.framesAtLOD == 0) {
            // First time seeing this mesh — initialize
            state.currentLOD      = targetLOD;
            state.targetLOD       = targetLOD;
            state.framesAtLOD     = 1;
            state.stabilityFrames = m_stabilityFrames;
            return targetLOD;
        }

        state.targetLOD = targetLOD;

        if (targetLOD == state.currentLOD) {
            // Stable — increment counter
            ++state.framesAtLOD;
        } else {
            // Target changed — only switch if stable long enough
            if (state.framesAtLOD >= state.stabilityFrames) {
                state.currentLOD  = targetLOD;
                state.framesAtLOD = 1;
            }
            // Otherwise stay on current LOD
        }

        return state.currentLOD;
    }

    // makeCacheKey 

    std::string PerceptualLODSelector::makeCacheKey(
        const std::string& meshId,
        const ViewData&    view) const
    {
        // Quantize distance to 0.5 unit buckets
        int distBucket = static_cast<int>(view.distance / 0.5f);
        std::ostringstream ss;
        ss << meshId << "_" << distBucket;
        return ss.str();
    }

}