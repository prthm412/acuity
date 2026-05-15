#define _USE_MATH_DEFINES
#include "FeatureExtractor.h"

#include <fstream>
#include <sstream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace Acuity {
    // Constructor
    FeatureExtractor::FeatureExtractor() {
        m_mean.fill(0.0f);
        m_std.fill(1.0f);   // default: no normalization
    }

    // Load scaler
    // Reads a plain-text file with two lines:
    //      mean: v0 v1 v2 ... v37
    //      std:  v0 v1 v2 ... v37
    // This file is exported by a small helper script alongside
    // feature_scaler.pkl (see python/dataset/export_scaler_txt.py)
    bool FeatureExtractor::loadScaler(const std::string& path)
    {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "[FeatureExtractor] Cannot open scaler: " << path << "\n";
            return false;
        }

        std::string line;
        int lineIdx = 0;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            std::string tag;
            ss >> tag;  // "mean:" or "std:"

            auto* arr = (lineIdx == 0) ? m_mean.data() : m_std.data();
            for (int i = 0; i < TOTAL_FEATURES; ++i)
                ss >> arr[i];

            ++lineIdx;
            if (lineIdx >= 2) break;
        }

        // Guard against zero std (would cause division by zero)
        for (int i = 0; i < TOTAL_FEATURES; ++i)
            if (m_std[i] < 1e-8f) m_std[i] = 1.0f;

        m_scalerLoaded = true;
        std::cout << "[FeatureExtractor] Scaler loaded from: " << path << "\n";
        return true;
    }

    // Public
    // extract (normalized)
    std::array<float, TOTAL_FEATURES> FeatureExtractor::extract(
        const MeshData& mesh, const ViewData& view
    ) const {
        auto raw = extractRaw(mesh, view);

        // Applying StandardScaler: z = (x - mean) / std
        std::array<float, TOTAL_FEATURES> out{};
        for (int i = 0; i < TOTAL_FEATURES; ++i)
            out[i] = (raw[i] - m_mean[i]) / m_std[i];

        return out;
    }

    // extractRaw
    std::array<float, TOTAL_FEATURES> FeatureExtractor::extractRaw(
        const MeshData& mesh, const ViewData& view
    ) const {
        std::array<float, TOTAL_FEATURES> out{};
        extractGeometric(mesh, out.data());
        extractPerceptual(mesh, out.data() + GEOMETRIC_FEATURES);
        extractView(mesh, view, out.data() + GEOMETRIC_FEATURES + PERCEPTUAL_FEATURES);

        return out;
    }

    // Geometric features [0-9]
    // Python's extract_features.py geometric block
    // [0] triangle_count   [1] vertex_count    [2] lod_level
    // [3] surface_area     [4] volume          [5] mean_edge_length
    // [6] std_edge_length  [7] bounding_radius [8] lod_ratio
    // [9] triangle_density

    void FeatureExtractor::extractGeometric(const MeshData& mesh, float* out) const
    {
        float triCount  = static_cast<float>(mesh.triangleCount);
        float vtxCount  = static_cast<float>(mesh.vertexCount);
        float lodLevel  = static_cast<float>(mesh.lodLevel);
        float area      = computeSurfaceArea(mesh);
        float volume    = computeVolume(mesh);
        float meanEdge  = computeMeanEdgeLength(mesh);
        float stdEdge   = computeStdEdgeLength(mesh);
        float bRadius   = computeBoundingRadius(mesh);

        // lod_ratio: fraction of triangles relative to LOD 0
        // At inference time we don't have LOD 0 count directly, so we
        // approximate using the known ratios [1, 0.5, 0.25, 0.125, 0.0625]
        static const float lodRatios[5] = {1.0f, 0.5f, 0.25f, 0.125f, 0.0625f};
        float lodRatio = (mesh.lodLevel >= 0 && mesh.lodLevel < 5) ? lodRatios[mesh.lodLevel] : 1.0f;

        float triDensity = (area > 1e-8f) ? triCount / area : 0.0f;

        out[0] = triCount;
        out[1] = vtxCount;
        out[2] = lodLevel;
        out[3] = area;
        out[4] = volume;
        out[5] = meanEdge;
        out[6] = stdEdge;
        out[7] = bRadius;
        out[8] = lodRatio;
        out[9] = triDensity;
    }

    // Perceptual features [10-24]
    // [10] mean_curvature      [11] std_curvature
    // [12] normal_variation    [13] mean_normal_x
    // [14] mean_normal_y       [15] mean_normal_z
    // [16] feature_edge_ratio  [17] silhouette_length
    // [18] roughness           [19] smoothness
    // [20] local_density_mean  [21] local_density_std
    // [22] aspect_ratio        [23] compactness
    // [24] shape_index

    void FeatureExtractor::extractPerceptual(const MeshData& mesh,
                                            float* out) const
    {
        float meanCurv  = computeMeanCurvature(mesh);
        float normVar   = computeNormalVariation(mesh);
        glm::vec3 ctr   = computeCenter(mesh);
        float area      = computeSurfaceArea(mesh);
        float volume    = computeVolume(mesh);
        float bRadius   = computeBoundingRadius(mesh);
        float triCount  = static_cast<float>(mesh.triangleCount);

        // Mean normal direction across all triangles
        glm::vec3 meanNormal(0.0f);
        for (const auto& n : mesh.normals)
            meanNormal += n;
        if (mesh.normals.size() > 0)
            meanNormal /= static_cast<float>(mesh.normals.size());

        // Feature edge ratio: fraction of edges with high dihedral angle
        // (simplified: use normal variation as proxy)
        float featureEdgeRatio = std::min(normVar / (float)M_PI, 1.0f);

        // Silhouette length proxy: perimeter approximation
        float silhouette = (bRadius > 1e-8f) ? 2.0f * (float)M_PI * bRadius : 0.0f;

        // Roughness: std of curvature
        float roughness  = normVar;
        float smoothness = 1.0f - std::min(normVar, 1.0f);

        // Local density
        float localDensityMean = (area > 1e-8f) ? triCount / area : 0.0f;
        float localDensityStd  = localDensityMean * 0.1f; // approximation

        // Shape descriptors
        float aspectRatio   = (bRadius > 1e-8f && volume > 1e-8f)
                            ? (area / (bRadius * bRadius)) : 0.0f;
        float compactness   = (bRadius > 1e-8f)
                            ? (area / (4.0f * (float)M_PI * bRadius * bRadius)) : 0.0f;
        float shapeIndex    = (volume > 1e-8f)
                            ? (std::cbrt(36.0f * (float)M_PI * volume * volume) / area)
                            : 0.0f;

        // std_curvature approximation
        float stdCurv = meanCurv * 0.5f;

        out[0]  = meanCurv;
        out[1]  = stdCurv;
        out[2]  = normVar;
        out[3]  = meanNormal.x;
        out[4]  = meanNormal.y;
        out[5]  = meanNormal.z;
        out[6]  = featureEdgeRatio;
        out[7]  = silhouette;
        out[8]  = roughness;
        out[9]  = smoothness;
        out[10] = localDensityMean;
        out[11] = localDensityStd;
        out[12] = aspectRatio;
        out[13] = compactness;
        out[14] = shapeIndex;
    }

    // View dependent features
    // [25] view_distance      [26] azimuth_angle   [27] elevation_angle
    // [28] screen_coverage    [29] dist_normalized [30] cos_elevation
    // [31] sin_elevation      [32] cos_azimuth     [33] sin_azimuth
    // [34] view_lod_interact  [35] distance_lod    [36] coverage_lod
    // [37] view_importance

    void FeatureExtractor::extractView(const MeshData& mesh,
                                        const ViewData& view,
                                        float* out) const
    {
        float bRadius = computeBoundingRadius(mesh);

        // Normalized distance (distance / bounding radius)
        float distNorm = (bRadius > 1e-8f) ? view.distance / bRadius : view.distance;

        // Trigonometric encodings of angles
        float cosEl = std::cos(view.elevation);
        float sinEl = std::sin(view.elevation);
        float cosAz = std::cos(view.azimuth);
        float sinAz = std::sin(view.azimuth);

        // Interaction features
        float lodRatios[5] = {1.0f, 0.5f, 0.25f, 0.125f, 0.0625f};
        float lodRatio = (mesh.lodLevel >= 0 && mesh.lodLevel < 5)
                        ? lodRatios[mesh.lodLevel] : 1.0f;

        float viewLodInteract = view.distance * lodRatio;
        float distanceLod     = distNorm * lodRatio;
        float coverageLod     = view.screenCoverage * lodRatio;

        // View importance: higher coverage + closer = more important
        float viewImportance  = view.screenCoverage / (1.0f + view.distance);

        out[0]  = view.distance;
        out[1]  = view.azimuth;
        out[2]  = view.elevation;
        out[3]  = view.screenCoverage;
        out[4]  = distNorm;
        out[5]  = cosEl;
        out[6]  = sinEl;
        out[7]  = cosAz;
        out[8]  = sinAz;
        out[9]  = viewLodInteract;
        out[10] = distanceLod;
        out[11] = coverageLod;
        out[12] = viewImportance;
    }

    // Geometry helpers
    float FeatureExtractor::computeSurfaceArea(const MeshData& mesh) const
    {
        float area = 0.0f;
        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            glm::vec3 a = mesh.vertices[mesh.indices[i]];
            glm::vec3 b = mesh.vertices[mesh.indices[i+1]];
            glm::vec3 c = mesh.vertices[mesh.indices[i+2]];
            area += 0.5f * glm::length(glm::cross(b - a, c - a));
        }
        return area;
    }

    float FeatureExtractor::computeVolume(const MeshData& mesh) const
    {
        float vol = 0.0f;
        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            glm::vec3 a = mesh.vertices[mesh.indices[i]];
            glm::vec3 b = mesh.vertices[mesh.indices[i+1]];
            glm::vec3 c = mesh.vertices[mesh.indices[i+2]];
            vol += glm::dot(a, glm::cross(b, c));
        }
        return std::abs(vol) / 6.0f;
    }

    float FeatureExtractor::computeMeanEdgeLength(const MeshData& mesh) const
    {
        if (mesh.indices.empty()) return 0.0f;
        float total = 0.0f;
        int count = 0;
        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            glm::vec3 a = mesh.vertices[mesh.indices[i]];
            glm::vec3 b = mesh.vertices[mesh.indices[i+1]];
            glm::vec3 c = mesh.vertices[mesh.indices[i+2]];
            total += glm::length(b - a)
                   + glm::length(c - b)
                   + glm::length(a - c);
            count += 3;
        }
        return (count > 0) ? total / count : 0.0f;
    }

    float FeatureExtractor::computeStdEdgeLength(const MeshData& mesh) const
    {
        if (mesh.indices.empty()) return 0.0f;
        float mean = computeMeanEdgeLength(mesh);
        float var  = 0.0f;
        int  count = 0;
        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            glm::vec3 a = mesh.vertices[mesh.indices[i]];
            glm::vec3 b = mesh.vertices[mesh.indices[i+1]];
            glm::vec3 c = mesh.vertices[mesh.indices[i+2]];
            auto addVar = [&](float len) {
                float d = len - mean; var += d * d; ++count;
            };
            addVar(glm::length(b - a));
            addVar(glm::length(c - b));
            addVar(glm::length(a - c));
        }
        return (count > 0) ? std::sqrt(var / count) : 0.0f;
    }

    float FeatureExtractor::computeMeanCurvature(const MeshData& mesh) const
    {
        if (mesh.normals.empty() || mesh.indices.empty()) return 0.0f;
        float totalCurv = 0.0f;
        int   count     = 0;
        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            uint32_t i0 = mesh.indices[i];
            uint32_t i1 = mesh.indices[i+1];
            uint32_t i2 = mesh.indices[i+2];
            if (i0 >= mesh.normals.size() ||
                i1 >= mesh.normals.size() ||
                i2 >= mesh.normals.size()) continue;

            // Discrete curvature proxy: average angle between face normals
            glm::vec3 n0 = mesh.normals[i0];
            glm::vec3 n1 = mesh.normals[i1];
            glm::vec3 n2 = mesh.normals[i2];
            float dot01 = glm::clamp(glm::dot(n0, n1), -1.0f, 1.0f);
            float dot12 = glm::clamp(glm::dot(n1, n2), -1.0f, 1.0f);
            totalCurv += std::acos(dot01) + std::acos(dot12);
            count += 2;
        }
        return (count > 0) ? totalCurv / count : 0.0f;
    }

    float FeatureExtractor::computeNormalVariation(const MeshData& mesh) const
    {
        if (mesh.normals.size() < 2) return 0.0f;
        glm::vec3 meanN(0.0f);
        for (const auto& n : mesh.normals) meanN += n;
        meanN /= static_cast<float>(mesh.normals.size());

        float var = 0.0f;
        for (const auto& n : mesh.normals) {
            glm::vec3 d = n - meanN;
            var += glm::dot(d, d);
        }
        return std::sqrt(var / static_cast<float>(mesh.normals.size()));
    }

    glm::vec3 FeatureExtractor::computeCenter(const MeshData& mesh) const
    {
        if (mesh.vertices.empty()) return glm::vec3(0.0f);
        glm::vec3 sum(0.0f);
        for (const auto& v : mesh.vertices) sum += v;
        return sum / static_cast<float>(mesh.vertices.size());
    }

    float FeatureExtractor::computeBoundingRadius(const MeshData& mesh) const
    {
        glm::vec3 center = computeCenter(mesh);
        float maxDist = 0.0f;
        for (const auto& v : mesh.vertices) {
            float d = glm::length(v - center);
            if (d > maxDist) maxDist = d;
        }
        return maxDist;
    }
}