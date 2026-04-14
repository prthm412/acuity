#include "LODGenerator.h"
#include "QuadricSimplifier.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>
#include <chrono>

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

        for (int i = 0; i < 5; ++i) {
            LODLevel& level = lodMesh.levels[i];
            level.targetRatio = LOD_RATIOS[i];
            level.debugColor  = LOD_COLORS[i];

            std::vector<Vertex>     outVerts;
            std::vector<uint32_t>   outIdx;

            // LOD 0 - No simplification needed (full res)
            QuadricSimplifier::simplify(vertices, indices, LOD_RATIOS[i], outVerts, outIdx);

            level.mesh.vertices = std::move(outVerts);
            level.mesh.indices  = std::move(outIdx);
            level.triangleCount = level.mesh.getIndexCount() / 3;

            // Apply debug color to all vertices
            level.mesh.setColor(level.debugColor);

            std::cout << "[LODGenerator] LOD " << i
                      << " (" << int(LOD_RATIOS[i] * 100) << "%): "
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
}