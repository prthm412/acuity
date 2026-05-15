#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <array>

namespace Acuity {
    class ONNXInference {
        public:
            static constexpr int  FEATURE_DIM = 38;
            static constexpr char INPUT_NAME[] = "features";
            static constexpr char OUTPUT_NAME[] = "quality_score";

            // Load model from disk. Throws runtime_error on failure
            explicit ONNXInference(const std::string& modelPath);
            ~ONNXInference() = default;

            // Non-copyable (owns Ort resources)
            ONNXInference(const ONNXInference&)            = delete;
            ONNXInference& operator=(const ONNXInference&) = delete;

            // Single-sample inference
            // features: exactly FEATURE_DIM floats
            // Returns predicted quality score in [0, 1]
            float predict(const std::array<float, FEATURE_DIM>& features) const;

            // Batch inference
            // features: flat float array of shape [batchSize * FEATURE_DIM]
            // Returns vector of batchSize scores
            std::vector<float> predictBatch(const std::vector<float>& features, int batchSize) const;

            // Benchmark: runs n inferences and returns mean latency in ms
            double benchmark(int n = 1000) const;

            bool isLoaded() const { return m_loaded; }
        
        private:
            Ort::Env                          m_env;
            Ort::SessionOptions               m_sessionOptions;
            std::unique_ptr<Ort::Session>     m_session;
            Ort::AllocatorWithDefaultOptions  m_allocator;
            bool                              m_loaded = false;

            // Helper: run inference on a pre-built input tensor
            std::vector<float> runSession(const float* inputData, int batchSize) const;
    };
}