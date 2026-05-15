#include "BatchRenderer.h"
#include "VulkanContext.h"
#include "Pipeline.h"
#include "OffscreenRenderer.h"
#include "../lod/LODGenerator.h"
#include "../lod/LODMesh.h"
#include "Mesh.h"
#include "Camera.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <array>

#include <regex>

namespace fs = std::filesystem;

namespace acuity {
    // JSON Helpers
    static std::string extractString(const std::string& json, const std::string& key)
    {
        std::string pattern = "\"" + key + "\"\\s*:\\s*\"([^\"]+)\"";
        std::regex  re(pattern);
        std::smatch m;
        if (std::regex_search(json, m, re)) return m[1].str();
        return "";
    }

    static int extractInt(const std::string& json, const std::string& key)
    {
        std::string pattern = "\"" + key + "\"\\s*:\\s*(-?\\d+)";
        std::regex  re(pattern);
        std::smatch m;
        if (std::regex_search(json, m, re)) return std::stoi(m[1].str());
        return 0;
    }

    static float extractFloat(const std::string& json, const std::string& key)
    {
        std::string pattern = "\"" + key + "\"\\s*:\\s*(-?[\\d.]+)";
        std::regex  re(pattern);
        std::smatch m;
        if (std::regex_search(json, m, re)) return std::stof(m[1].str());
        return 0.0f;
    }

    // Extract array of objects between matching braces for a key
    static std::vector<std::string> extractObjectArray(const std::string& json, const std::string& key)
    {
        std::vector<std::string> objects;
        std::string searchKey = "\"" + key + "\"";
        size_t keyPos = json.find(searchKey);
        if (keyPos == std::string::npos) return objects;

        size_t arrStart = json.find('[', keyPos);
        if (arrStart == std::string::npos) return objects;

        int depth = 0;
        size_t objStart = std::string::npos;
        for (size_t i = arrStart; i < json.size(); ++i) {
            if (json[i] == '{') {
                if (depth == 0) objStart = i;
                ++depth;
            } else if (json[i] == '}') {
                --depth;
                if (depth == 0 && objStart != std::string::npos) {
                    objects.push_back(json.substr(objStart, i - objStart + 1));
                    objStart = std::string::npos;
                }
            } else if (json[i] == ']' && depth == 0) {
                break;
            }
        }
        return objects;
    }

    // Public
    BatchRenderer::RenderStats BatchRenderer::run(
        VulkanContext&      ctx,
        Pipeline&           pipeline,
        const std::string&  configPath,
        const std::string&  outputDir
    )
    {
        RenderStats stats;
        std::cout << "[BatchRenderer] Starting batch render" << std::endl;
        std::cout << "[BatchRenderer] Config: " << configPath << std::endl;
        std::cout << "[BatchRenderer] Output: " << outputDir << std::endl;

        // Load configuration
        RenderConfig config = loadConfig(configPath);
        if (config.meshes.empty()) {
            std::cerr << "[BatchRenderer] No meshes in config. Aborting." << std::endl;
            return stats;
        }

        std::cout << "[BatchRenderer] Meshes: " << config.meshes.size()
                  << ", LOD: " << config.lodLevels.size()
                  << ", Poses: " << config.poses.size()
                  << ", Total renders: "
                  << config.meshes.size() * config.lodLevels.size() * config.poses.size()
                  << std::endl;

        // Create output directory
        fs::create_directories(outputDir);

        // Open log file
        std::string logPath = outputDir + "/render_log.csv";
        std::ofstream logFile(logPath);
        logFile << "mesh_name,lod_level,lod_ratio,pose_id,azimuth_deg,"
                   "elevation_deg,distance_mult,distance_label,"
                   "output_path,render_time_ms\n";
        
        auto totalStart = std::chrono::steady_clock::now();

        // Render each mesh
        for (const auto& mesh : config.meshes) {
            renderMesh(ctx, pipeline, mesh, config, outputDir, stats, logFile);
        }

        auto totalEnd = std::chrono::steady_clock::now();
        stats.totalTimeMs = std::chrono::duration<float, std::milli>(
            totalEnd - totalStart).count();
        
        logFile.close();

        std::cout << "[BatchRenderer] Complete." << std::endl;
        std::cout << "[BatchRenderer] Renderer: " << stats.totalRendered << std::endl;
        std::cout << "[BatchRenderer] Skipped: " << stats.totalSkipped << std::endl;
        std::cout << "[BatchRenderer] Total time: "
                  << stats.totalTimeMs / 1000.0f << "s" << std::endl;

        return stats;
    }

    // Private
    BatchRenderer::RenderConfig BatchRenderer::loadConfig(const std::string& configPath)
    {
        RenderConfig config;

        std::ifstream f(configPath);
        if (!f.is_open()) {
            std::cerr << "[BatchRenderer] Cannot open config: " << configPath << std::endl;
            return config;
        }

        std::string json((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());

        config.imageWidth  = extractInt(json, "image_width");
        config.imageHeight = extractInt(json, "image_height");

        // Parse poses
        for (const auto& obj : extractObjectArray(json, "poses")) {
            PoseConfig pose;
            pose.poseId         = extractInt(obj,   "pose_id");
            pose.azimuthDeg     = extractFloat(obj, "azimuth_deg");
            pose.elevationDeg   = extractFloat(obj, "elevation_deg");
            pose.distanceMult   = extractFloat(obj, "distance_mult");
            pose.distanceLabel  = extractString(obj, "distance_label");
            pose.angleLabel     = extractString(obj, "angle_label");
            config.poses.push_back(pose);
        }

        // Parse meshes
        for (const auto& obj : extractObjectArray(json, "meshes")) {
            MeshConfig mesh;
            mesh.name = extractString(obj, "name");
            mesh.path = extractString(obj, "path");
            config.meshes.push_back(mesh);
        }

        // LOD levels and ratios
        config.lodLevels = {0, 1, 2, 3, 4};
        config.lodRatios = {1.0f, 0.5f, 0.25f, 0.125f, 0.0625f};

        std::cout << "[BatchRenderer] Config loaded: "
                  << config.meshes.size() << " meshes, "
                  << config.poses.size()  << " poses, "
                  << config.imageWidth    << "x" << config.imageHeight << std::endl;

        return config;
    }

    void BatchRenderer::renderMesh(
        VulkanContext&      ctx,
        Pipeline&           pipeline,
        const MeshConfig&   meshCfg,
        const RenderConfig& config,
        const std::string&  outputDir,
        RenderStats&        stats,
        std::ofstream&      logFile
    )
    {
        std::cout << "[BatchRenderer] Processing: " << meshCfg.name << std::endl;

        // Load mesh
        Mesh baseMesh;
        std::string resolvedPath = meshCfg.path;
        if (!baseMesh.loadFromFile(resolvedPath)) {
            std::cerr << "[BatchRenderer] Failed to load: " << resolvedPath << std::endl;
            stats.totalSkipped += (int)(config.lodLevels.size() * config.poses.size());
            return;
        }

        std::cout << "[BatchRenderer] Loaded: " << baseMesh.getVertexCount()
                  << " vertices, " << baseMesh.getIndexCount() / 3
                  << " triangles" << std::endl;

        // Compute bounding sphere radius for camera distance scaling
        float maxDist = 0.0f;
        for (const auto& v : baseMesh.vertices) {
            float d = std::sqrt(v.position.x * v.position.x + v.position.y * v.position.y + v.position.z * v.position.z);
            if (d > maxDist) maxDist = d;
        }
        float boundingRadius = (maxDist > 0.0f) ? maxDist : 1.0f;

        // Generate LOD hierarchy - use cache if available to skip slow QEM
        std::string cacheDir = "data/processed/lod_cache";
        LODMesh lodMesh;
        if (!LODGenerator::loadFromCache(meshCfg.name, cacheDir, lodMesh)) {
            std::cerr << "[BatchRenderer] Cache miss for: " << meshCfg.name
                    << " — run python/dataset/generate_lod_cache.py first"
                    << std::endl;
            stats.totalSkipped += (int)(config.lodLevels.size() * config.poses.size());
            return;
        }

        // Safety net: if any LOD level has exploded geometry (vertices outside
        // 3x bounding radius), replace it with the previous level's geometry.
        for (int i = 1; i < (int)lodMesh.levels.size(); ++i) {
            bool exploded = false;
            for (const auto& v : lodMesh.levels[i].mesh.vertices) {
                float d = std::sqrt(v.position.x * v.position.x +
                                    v.position.y * v.position.y +
                                    v.position.z * v.position.z);
                if (d > boundingRadius * 3.0f) {
                    exploded = true;
                    break;
                }
            }
            if (exploded) {
                std::cout << "[BatchRenderer] LOD " << i
                        << " exploded for " << meshCfg.name
                        << " — replacing with LOD " << (i-1) << std::endl;
                lodMesh.levels[i].mesh.vertices = lodMesh.levels[i-1].mesh.vertices;
                lodMesh.levels[i].mesh.indices  = lodMesh.levels[i-1].mesh.indices;
                lodMesh.levels[i].triangleCount = lodMesh.levels[i-1].triangleCount;
            }
        }

        // Upload all LOD levels to GPU
        LODGenerator::uploadToGPU(lodMesh, ctx.device, ctx.physicalDevice, ctx.commandPool, ctx.graphicsQueue);

        // Initialize offscreen renderer
        OffscreenRenderer osr;
        osr.init(ctx, config.imageWidth, config.imageHeight);

        // Render each LOD x pose combination
        for (int lodIdx = 0; lodIdx < (int)config.lodLevels.size(); ++lodIdx) {
            int   lodLevel = config.lodLevels[lodIdx];
            float lodRatio = config.lodRatios[lodIdx];
            const Mesh& lodMeshLevel = lodMesh.levels[lodIdx].mesh;

            if (lodMeshLevel.getIndexCount() == 0) {
                std::cout << "[BatchRenderer] LOD " << lodLevel
                          << " has no geometry, skipping" << std::endl;
                stats.totalSkipped += (int)config.poses.size();
                continue;
            }

            for (const auto& pose : config.poses) {
                // Build output filename
                std::ostringstream fname;
                fname << meshCfg.name
                      << "_lod" << lodLevel
                      << "_"    << pose.angleLabel
                      << "_"    << pose.distanceLabel
                      << ".png";

                std::string outPath = outputDir + "/" + fname.str();

                // Skip if already rendered (resume support)
                if (fs::exists(outPath)) {
                    ++stats.totalSkipped;
                    continue;
                }

                auto frameStart = std::chrono::steady_clock::now();

                // Compute camera matrices
                float viewMat[16], projMat[16];
                computeCameraTransform(pose.azimuthDeg, pose.elevationDeg,
                                       pose.distanceMult, boundingRadius,
                                       viewMat, projMat,
                                       config.imageWidth, config.imageHeight);

                // Record render commands
                osr.beginFrame();

                // Begin render pass
                std::array<VkClearValue, 2> clearValues{};
                clearValues[0].color        = {{0.15f, 0.15f, 0.15f, 1.0f}};
                clearValues[1].depthStencil = {1.0f, 0};

                VkRenderPassBeginInfo rpBegin{};
                rpBegin.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                rpBegin.renderPass        = osr.renderPass;
                rpBegin.framebuffer       = osr.framebuffer;
                rpBegin.renderArea.offset = {0, 0};
                rpBegin.renderArea.extent = {osr.width, osr.height};
                rpBegin.clearValueCount   = 2;
                rpBegin.pClearValues      = clearValues.data();

                vkCmdBeginRenderPass(osr.commandBuffer, &rpBegin,
                                    VK_SUBPASS_CONTENTS_INLINE);

                // Set viewport and scissor
                VkViewport viewport{};
                viewport.x        = 0.0f;
                viewport.y        = 0.0f;
                viewport.width    = static_cast<float>(osr.width);
                viewport.height   = static_cast<float>(osr.height);
                viewport.minDepth = 0.0f;
                viewport.maxDepth = 1.0f;
                vkCmdSetViewport(osr.commandBuffer, 0, 1, &viewport);

                VkRect2D scissor{{0, 0}, {osr.width, osr.height}};
                vkCmdSetScissor(osr.commandBuffer, 0, 1, &scissor);

                // Bind pipeline
                pipeline.bind(osr.commandBuffer);

                // Push MVP matrices as push constants
                // Layout: model(64) + view(64) + proj(64) = 192 bytes
                float modelMat[16] = {
                    1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1  // identity
                };
                vkCmdPushConstants(osr.commandBuffer,
                                pipeline.pipelineLayout,
                                VK_SHADER_STAGE_VERTEX_BIT,
                                0,   64, modelMat);
                vkCmdPushConstants(osr.commandBuffer,
                                pipeline.pipelineLayout,
                                VK_SHADER_STAGE_VERTEX_BIT,
                                64,  64, viewMat);
                vkCmdPushConstants(osr.commandBuffer,
                                pipeline.pipelineLayout,
                                VK_SHADER_STAGE_VERTEX_BIT,
                                128, 64, projMat);

                // Draw the LOD mesh
                VkBuffer     vBuf = lodMeshLevel.vertexBuffer;
                VkDeviceSize off  = 0;
                vkCmdBindVertexBuffers(osr.commandBuffer, 0, 1, &vBuf, &off);
                vkCmdBindIndexBuffer(osr.commandBuffer,
                                    lodMeshLevel.indexBuffer, 0,
                                    VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(osr.commandBuffer,
                                lodMeshLevel.getIndexCount(), 1, 0, 0, 0);

                vkCmdEndRenderPass(osr.commandBuffer);

                // Submit and wait
                osr.endFrame(ctx);

                // Read pixels and save PNG
                auto pixels  = osr.readPixels(ctx);
                bool saved   = savePNG(outPath, pixels,
                                       config.imageWidth, config.imageHeight);

                auto frameEnd = std::chrono::steady_clock::now();
                float frameMs = std::chrono::duration<float, std::milli>(
                    frameEnd - frameStart
                ).count();

                if (saved) {
                    ++stats.totalRendered;
                    stats.totalTimeMs += frameMs;

                    // Log entry
                    logFile << meshCfg.name         << ","
                            << lodLevel             << ","
                            << lodRatio             << ","
                            << pose.poseId          << ","
                            << pose.azimuthDeg      << ","
                            << pose.elevationDeg    << ","
                            << pose.distanceMult    << ","
                            << pose.distanceLabel   << ","
                            << outPath              << ","
                            << frameMs              << "\n";
                } else {
                    ++stats.totalSkipped;
                }
            }
        }

        osr.destroy(ctx.device);
        lodMesh.destroy(ctx.device);

        std::cout << "[BatchRenderer] Done: " << meshCfg.name
                  << " (" << stats.totalRendered << " rendered so far)" << std::endl;
    }

    void BatchRenderer::computeCameraTransform(
        float azimuthDeg, float elevationDeg, float distanceMult, float boundingRadius,
        float outView[16], float outProj[16],
        int width, int height
    )
    {
        const float PI = 3.14159265358979323846f;
        float az   = azimuthDeg   * PI / 180.0f;
        float el   = elevationDeg * PI / 180.0f;
        float dist = distanceMult * boundingRadius;

        // Camera position in spherical coordinates
        float cx = dist * std::cos(el) * std::sin(az);
        float cy = dist * std::sin(el);
        float cz = dist * std::cos(el) * std::cos(az);

        // Look-at: eye = (cx, cy, cz), center = (0, 0, 0), up = (0, 1, 0)
        float fx = -cx, fy = -cy, fz = -cz;
        float flen = std::sqrt(fx*fx + fy*fy + fz*fz);
        fx /= flen; fy /= flen; fz /= flen;

        float ux = 0.0f, uy = 1.0f, uz = 0.0f;

        // right = forward x up
        float rx = fy*uz - fz*uy;
        float ry = fz*ux - fx*uz;
        float rz = fx*uy - fy*ux;
        float rlen = std::sqrt(rx*rx + ry*ry + rz*rz);
        rx /= rlen; ry /= rlen; rz /= rlen;

        // up = right x forward
        ux = ry*fz - rz*fy;
        uy = rz*fx - rx*fz;
        uz = rx*fy - ry*fx;

        // View matrix (row-major, Vulkan column-major convention)
        outView[0]  =  rx; outView[1]  =  ux; outView[2]  = -fx; outView[3]  = 0;
        outView[4]  =  ry; outView[5]  =  uy; outView[6]  = -fy; outView[7]  = 0;
        outView[8]  =  rz; outView[9]  =  uz; outView[10] = -fz; outView[11] = 0;
        outView[12] = -(rx*cx + ry*cy + rz*cz);
        outView[13] = -(ux*cx + uy*cy + uz*cz);
        outView[14] =  (fx*cx + fy*cy + fz*cz);
        outView[15] = 1.0f;

        // Perspective projection
        float fovY    = 45.0f * PI / 180.0f;
        float aspect  = static_cast<float>(width) / static_cast<float>(height);
        float near    = 0.01f;
        float far     = dist * 10.0f;
        float tanHalf = std::tan(fovY * 0.5f);

        outProj[0]  = 1.0f / (aspect * tanHalf);
        outProj[1]  = 0; outProj[2]  = 0; outProj[3]  = 0;
        outProj[4]  = 0;
        outProj[5]  = -1.0f / tanHalf; // Vulkan Y-flip
        outProj[6]  = 0; outProj[7]  = 0;
        outProj[8]  = 0; outProj[9]  = 0;
        outProj[10] = far / (near - far);
        outProj[11] = -1.0f;
        outProj[12] = 0; outProj[13] = 0;
        outProj[14] = -(far * near) / (far - near);
        outProj[15] = 0;
    }

    bool BatchRenderer::savePNG(const std::string& path, const std::vector<uint8_t>& pixels, int width, int height)
    {
        int result = stbi_write_png(path.c_str(), width, height, 4, pixels.data(), width * 4);
        return result != 0;
    }
}