#include "LODGenerator.h"
#include "QuadricSimplifier.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace acuity {
    // constexpr arrays for older compilers
    constexpr float     LODGenerator::LOD_RATIOS[5];
    constexpr glm::vec3 LODGenerator::LOD_COLORS[5];

    LODMesh LODGenerator::generate(
        const std::string&          meshName,
        const std::vector<Vertex>&  vertices,
        const std::vector<uint32_t>& indices
    )
    {
        LODMesh lodMesh;
        lodMesh.name = meshName;
        lodMesh.levels.resize(5);

        std::cout << "[LODGenerator] Generating LODs for: " << meshName << std::endl;
        std::cout << "[LODGenerator] Source: "
                  << vertices.size() << " vertices, "
                  << indices.size() / 3 << " triangles" << std::endl;

        auto totalStart = std::chrono::steady_clock::now();

        // Adaptive LOD ratios: prevent over-simplification on high-poly meshes
        // Standard ratios work for meshes under 100k triangles.
        // For larger meshes, cap minimum ratio to avoid QEM instability.
        float effectiveRatios[5];
        size_t triCount = indices.size() / 3;
        if (triCount > 100000) {
            effectiveRatios[0] = 1.0f;
            effectiveRatios[1] = 0.5f;
            effectiveRatios[2] = 0.25f;
            effectiveRatios[3] = 0.15f;   // instead of 0.125
            effectiveRatios[4] = 0.10f;   // instead of 0.0625
        } else {
            effectiveRatios[0] = LOD_RATIOS[0];
            effectiveRatios[1] = LOD_RATIOS[1];
            effectiveRatios[2] = LOD_RATIOS[2];
            effectiveRatios[3] = LOD_RATIOS[3];
            effectiveRatios[4] = LOD_RATIOS[4];
        }

        for (int i = 0; i < 5; ++i) {
            LODLevel& level = lodMesh.levels[i];
            level.targetRatio = effectiveRatios[i];
            level.debugColor  = LOD_COLORS[i];

            std::vector<Vertex>     outVerts;
            std::vector<uint32_t>   outIdx;

            // LOD 0 - No simplification needed (full res)
            QuadricSimplifier::simplify(vertices, indices, effectiveRatios[i], outVerts, outIdx);

            level.mesh.vertices = std::move(outVerts);
            level.mesh.indices  = std::move(outIdx);
            level.triangleCount = level.mesh.getIndexCount() / 3;

            // Apply debug color to all vertices
            level.mesh.setColor(level.debugColor);

            std::cout << "[LODGenerator] LOD " << i
                      << " (" << int(effectiveRatios[i] * 100) << "%): "
                      << level.triangleCount << " triangles" << std::endl;
        }

        auto totalEnd = std::chrono::steady_clock::now();
        float totalMs = std::chrono::duration<float, std::milli>(totalEnd - totalStart).count();
        std::cout << "[LOD Generator] Total generation time: " << totalMs << "ms" << std::endl;

        return lodMesh;
    }

    LODMesh LODGenerator::generateFromFile(const std::string& filePath)
    {
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(filePath,
            aiProcess_Triangulate |
            aiProcess_GenSmoothNormals |
            aiProcess_JoinIdenticalVertices);
        
        if (!scene || !scene->mRootNode) {
            std::cerr << "[LODGenerator] Failed to load: "
                      << importer.GetErrorString() << std::endl;
            return{};
        }

        std::vector<Vertex>     vertices;
        std::vector<uint32_t>   indices;

        for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
            aiMesh* mesh = scene->mMeshes[m];
            uint32_t offset = static_cast<uint32_t>(vertices.size());

            for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
                Vertex v{};
                v.position = { mesh->mVertices[i].x,
                               mesh->mVertices[i].y,
                               mesh->mVertices[i].z };
                v.normal   = mesh->HasNormals()
                         ? glm::vec3(mesh->mNormals[i].x,
                                     mesh->mNormals[i].y,
                                     mesh->mNormals[i].z)
                         : glm::vec3(0.f, 1.f, 0.f);
                v.color    = glm::vec3(0.8f);
                vertices.push_back(v);
            }
            for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
                aiFace& face = mesh->mFaces[i];
                for (unsigned int j = 0; j < face.mNumIndices; ++j)
                    indices.push_back(offset + face.mIndices[j]);
            }
        }

        // Extract mesh name from file path
        std::string name = filePath;
        size_t slash = name.find_last_of("/\\");
        if (slash != std::string::npos) name = name.substr(slash + 1);
        size_t dot = name.find_last_of('.');
        if (dot != std::string::npos) name = name.substr(0, dot);

        return generate(name, vertices, indices);
    }

    void LODGenerator::uploadToGPU(
        LODMesh&         lodMesh,
        VkDevice         device,
        VkPhysicalDevice physicalDevice,
        VkCommandPool    commandPool,
        VkQueue          graphicsQueue
    )
    {
        for (auto& level : lodMesh.levels) {
            // Apply debug color before upload so GPU sees the colored vertices
            level.mesh.setColor(level.debugColor);
            level.mesh.uploadToGPU(device, physicalDevice, commandPool, graphicsQueue);
        }
        std::cout << "[LODGenerator] All LOD levels uploaded to GPU for: "
                  << lodMesh.name << std::endl;
    }

    void LODGenerator::saveToCache(const LODMesh& lodMesh, const std::string& cacheDir)
    {
        // Create cache directory if it doesn't exist
        std::filesystem::create_directories(cacheDir);

        for (int i = 0; i < (int)lodMesh.levels.size(); ++i) {
            std::string path = cacheDir + "/" + lodMesh.name + "_lod" + std::to_string(i) + ".obj";

            std::ofstream f(path);
            if (!f.is_open()) {
                std::cerr << "[LODGenerator] Cache write failed: " << path << std::endl;
                continue;
            }

            const auto& verts   = lodMesh.levels[i].mesh.vertices;
            const auto& indices = lodMesh.levels[i].mesh.indices;

            f << "# Acuity LOD cache: " << lodMesh.name << " level " << i << "\n";
            f << "# Triangles: " << indices.size() / 3 << "\n";

            for (const auto& v : verts)
                f << "v " << v.position.x << " "
                          << v.position.y << " "
                          << v.position.z << "\n";

            for (size_t j = 0; j < indices.size(); j+=3)
                f << "f " << indices[j]+1 << " "
                          << indices[j+1]+1 << " "
                          << indices[j+2]+1 << "\n";

            f.close();
        }
        std::cout << "[LODGenerator] Cache saved: " << lodMesh.name << std::endl;
    }

    bool LODGenerator::loadFromCache(const std::string& meshName, const std::string& cacheDir, LODMesh& lodMesh)
    {
        // Check all 5 cache files exist before attempting load
        for (int i = 0; i < 5; ++i) {
            std::string path = cacheDir + "/" + meshName + "_lod" + std::to_string(i) + ".obj";
            if (!std::filesystem::exists(path))
                return false;
        }

        lodMesh.name = meshName;
        lodMesh.levels.resize(5);

        for (int i = 0; i < 5; ++i) {
            std::string path = cacheDir + "/" + meshName + "_lod" + std::to_string(i) + ".obj";

            std::ifstream f(path);
            if (!f.is_open()) return false;

            std::vector<Vertex>   verts;
            std::vector<uint32_t> indices;

            std::string line;
            while (std::getline(f, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream ss(line);
                std::string token;
                ss >> token;
                if (token == "v") {
                    Vertex v{};
                    ss >> v.position.x >> v.position.y >> v.position.z;
                    verts.push_back(v);
                } else if (token == "f") {
                    uint32_t a, b, c;
                    ss >> a >> b >> c;
                    indices.push_back(a - 1);
                    indices.push_back(b - 1);
                    indices.push_back(c - 1);
                }
            }

            lodMesh.levels[i].mesh.vertices     = std::move(verts);
            lodMesh.levels[i].mesh.indices      = std::move(indices);
            lodMesh.levels[i].targetRatio       = LOD_RATIOS[i];
            lodMesh.levels[i].debugColor        = LOD_COLORS[i];
            lodMesh.levels[i].triangleCount     = lodMesh.levels[i].mesh.getIndexCount() / 3;
            lodMesh.levels[i].mesh.setColor(LOD_COLORS[i]);
        }

        std::cout << "[LODGenerator] Cache loaded: " << meshName << std::endl;
        return true;
    }
}