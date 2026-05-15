#pragma once
#include <string>
#include <vector>

namespace acuity {
    class VulkanContext;
    class Pipeline;

    // meshes x LOD levels x camera poses
    // output images:
    //      {mesh_name}_lod{0-4}_az{angle}_dist{distance_label}.png

    class BatchRenderer {
        public:
            struct RenderStats {
                int     totalRendered = 0;
                int     totalSkipped  = 0;
                float   totalTimeMs   = 0.0f;
            };

            // Run full batch render pipeline
            // configPath: path to data/processed/render_configs.json
            // outputDir:  path to data/processed/rendered_images/
            RenderStats run(VulkanContext& ctx, Pipeline& pipeline,
                            const std::string& configPath, const std::string& outputDir);
        
        private:
            struct PoseConfig {
                int         poseId;
                float       azimuthDeg;
                float       elevationDeg;
                float       distanceMult;
                std::string distanceLabel;
                std::string angleLabel;
            };

            struct MeshConfig {
                std::string name;
                std::string path;
            };

            struct RenderConfig {
                int                     imageWidth;
                int                     imageHeight;
                std::vector<int>        lodLevels;
                std::vector<float>      lodRatios;
                std::vector<PoseConfig> poses;
                std::vector<MeshConfig> meshes;
            };

            RenderConfig loadConfig(const std::string& configPath);

            void renderMesh(VulkanContext&     ctx,
                        Pipeline&          pipeline,
                        const MeshConfig&  mesh,
                        const RenderConfig& config,
                        const std::string& outputDir,
                        RenderStats&       stats,
                        std::ofstream&     logFile);
            
            // Compute camera position from spherical coordinates and mesh bounds
            void computeCameraTransform(float azimuthDeg, float elevationDeg, float distanceMult, float boundingRadius,
                                        float outView[16], float outProj[16], int width, int height);

            // Write RGBA pixel buffer to PNG using stb_image_write
            bool savePNG(const std::string& path,
                         const std::vector<uint8_t>& pixels,
                         int width, int height);
    };
}