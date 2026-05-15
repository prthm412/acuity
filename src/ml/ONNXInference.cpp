#include "ONNXInference.h"

#include <stdexcept>
#include <numeric>
#include <chrono>
#include <cstring>
#include <iostream>
#include <cstdio>
#ifdef _WIN32
#include <io.h>
#endif

namespace acuity {
    // Constructor
    ONNXInference::ONNXInference(const std::string& modelPath) : m_env(ORT_LOGGING_LEVEL_ERROR, "AcuityONNX") {
        m_sessionOptions.SetIntraOpNumThreads(1);
        m_sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Suppress ONNX schema registration warnings printed to stderr
        FILE* devNull = nullptr;
        int savedStderr = -1;

    #ifdef _WIN32
        savedStderr = _dup(_fileno(stderr));
        freopen_s(&devNull, "NUL", "w", stderr);
    #endif

        // Convert path to wide string on Windows (ONNX Runtime requires wchar_t*)
    #ifdef _WIN32
        std::wstring widePath(modelPath.begin(), modelPath.end());
        m_session = std::make_unique<Ort::Session>(m_env, widePath.c_str(), m_sessionOptions);
    #else
        m_session = std::make_unique<Ort::Session>(m_env, modelPath.c_str(), m_sessionOptions);
    #endif

        m_loaded = true;
        std::cout << "[ONNXInference] Model loaded: " << modelPath << "\n";
    }

    // Single sample
    float ONNXInference::predict(const std::array<float, FEATURE_DIM>& features) const {
        auto scores = runSession(features.data(), 1);
        return scores[0];
    }

    // Batch
    std::vector<float> ONNXInference::predictBatch(
        const std::vector<float>& features, int batchSize
    ) const {
        if (static_cast<int>(features.size()) != batchSize * FEATURE_DIM)
            throw std::runtime_error("[ONNXInference] Feature vector size mismatch");

        return runSession(features.data(), batchSize);
    }

    // Benchmark
    double ONNXInference::benchmark(int n) const
    {
        std::array<float, FEATURE_DIM> dummy{};
        dummy.fill(0.5f);

        // Warmup
        for (int i = 0; i < 50; ++i)
            predict(dummy);

        // Timed runs
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n; ++i)
            predict(dummy);
        auto t1 = std::chrono::high_resolution_clock::now();

        double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return totalMs / n;
    }

    // Private
    // runSession
    std::vector<float> ONNXInference::runSession(
        const float* inputData,
        int           batchSize) const
    {
        // Build input tensor shape: [batchSize, FEATURE_DIM]
        std::array<int64_t, 2> inputShape = {
            static_cast<int64_t>(batchSize),
            static_cast<int64_t>(FEATURE_DIM)
        };

        auto memInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo,
            const_cast<float*>(inputData),
            static_cast<size_t>(batchSize) * FEATURE_DIM,
            inputShape.data(),
            inputShape.size());

        // I/O name pointers (must match ONNX export names exactly)
        const char* inputNames[]  = { INPUT_NAME  };
        const char* outputNames[] = { OUTPUT_NAME };

        auto outputTensors = m_session->Run(
            Ort::RunOptions{nullptr},
            inputNames,  &inputTensor, 1,
            outputNames,               1);

        // Extract output
        const float* outputData = outputTensors[0].GetTensorData<float>();
        int          outputSize = static_cast<int>(
            outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount());

        return std::vector<float>(outputData, outputData + outputSize);
    }
}