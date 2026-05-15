#pragma once

#include "FeatureExtractor.h"
#include "../ml/ONNXInference.h"

#include <array>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

namespace acuity {
    //  Uses the trained ONNX model to select the best LOD level for
    //  a mesh given the current view. Owns a FeatureExtractor and
    //  ONNXInference instance. Called once per mesh per frame.
    //
    //  Selection logic:
    //  - Extract 38 features for each LOD candidate (0-4)
    //  - Run ONNX inference to get quality score for each
    //  - Select highest LOD (lowest detail) whose quality score
    //    is above the quality threshold
    //  - Respect memory budget: skip LODs that exceed budget
    //  - Apply temporal stability: prevent rapid LOD switching
    //    (hysteresis) to avoid visual popping

    static constexpr int   NUM_LOD_LEVELS    = 5;
    static constexpr float DEFAULT_THRESHOLD = 0.5f;

    // Memory budget: maximum triangle count allowed
    // -1 means no budget constraint
    static constexpr int   NO_MEMORY_BUDGET  = -1;

    // Temporal stability: minimum frames before LOD can change
    static constexpr int   DEFAULT_STABILITY_FRAMES = 10;

    struct LODCandidate {
        int   lodLevel;
        float qualityScore;
        int   triangleCount;
    };

    // Per-mesh temporal state
    struct TemporalState {
        int      currentLOD     = 0;
        int      targetLOD      = 0;
        int      framesAtLOD    = 0;
        int      stabilityFrames = DEFAULT_STABILITY_FRAMES;
    };

    class PerceptualLODSelector {
    public:
        PerceptualLODSelector();

        // Initialize — loads ONNX model and scaler
        bool init(const std::string& modelPath,
                const std::string& scalerPath);

        // Select best LOD level for a single mesh given current view.
        // meshId:    unique identifier for caching and temporal state
        // lodMeshes: one MeshData per LOD level (index = LOD level)
        // view:      current camera/view state
        // Returns selected LOD index (0 = highest detail)
        int selectLOD(const std::string&            meshId,
                    const std::vector<MeshData>&  lodMeshes,
                    const ViewData&               view);

        // Batch selection: select LODs for multiple meshes at once.
        // More efficient than calling selectLOD() in a loop because
        // features are extracted and inference is run in one batch.
        std::vector<int> selectLODBatch(
            const std::vector<std::string>&           meshIds,
            const std::vector<std::vector<MeshData>>& lodMeshesPerMesh,
            const std::vector<ViewData>&              views);

        // Get all candidate scores from last selectLOD call
        const std::vector<LODCandidate>& getLastCandidates() const {
            return m_lastCandidates;
        }

        // Quality threshold: LOD selected if score >= threshold
        void  setThreshold(float t) { m_threshold = t; }
        float getThreshold()  const { return m_threshold; }

        // Memory budget: maximum triangles allowed per mesh
        // Set to NO_MEMORY_BUDGET (-1) to disable
        void setMemoryBudget(int maxTriangles) { m_memoryBudget = maxTriangles; }
        int  getMemoryBudget() const           { return m_memoryBudget; }

        // Temporal stability: frames before LOD level can change
        void setStabilityFrames(int frames) { m_stabilityFrames = frames; }
        int  getStabilityFrames() const     { return m_stabilityFrames; }

        // Clear spatial cache (call when scene changes significantly)
        void clearCache() { m_cache.clear(); }

        // Clear temporal state for all meshes
        void clearTemporalState() { m_temporalStates.clear(); }

        bool isReady() const { return m_ready; }

    private:
        FeatureExtractor               m_extractor;
        std::unique_ptr<ONNXInference> m_model;
        float                          m_threshold      = DEFAULT_THRESHOLD;
        int                            m_memoryBudget   = NO_MEMORY_BUDGET;
        int                            m_stabilityFrames = DEFAULT_STABILITY_FRAMES;
        bool                           m_ready          = false;

        // Cache: meshId + quantized view key → selected LOD
        std::unordered_map<std::string, int> m_cache;

        // Temporal state per mesh
        std::unordered_map<std::string, TemporalState> m_temporalStates;

        // Last frame candidates for debug UI
        std::vector<LODCandidate> m_lastCandidates;

        // Score all LOD levels for one mesh, returns candidates
        std::vector<LODCandidate> scoreLODs(
            const std::vector<MeshData>& lodMeshes,
            const ViewData&              view);

        // Select best LOD from candidates respecting threshold
        // and memory budget
        int selectFromCandidates(
            const std::vector<LODCandidate>& candidates);

        // Apply temporal stability (hysteresis) to prevent popping
        int applyTemporalStability(const std::string& meshId,
                                    int                targetLOD);

        // Build cache key from mesh id and quantized view distance
        std::string makeCacheKey(const std::string& meshId,
                                const ViewData&    view) const;
    };

}