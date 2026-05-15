# ONNXInference API Documentation

## Overview

`ONNXInference` (`src/ml/ONNXInference.h`) is a C++ wrapper around
ONNX Runtime that loads and runs the trained `lod_perception.onnx`
model for real-time perceptual quality prediction.

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `FEATURE_DIM` | 38 | Input feature vector dimension |
| `INPUT_NAME` | `"features"` | ONNX model input node name |
| `OUTPUT_NAME` | `"quality_score"` | ONNX model output node name |

## Constructor

```cpp
explicit ONNXInference(const std::string& modelPath);
```

Loads the ONNX model from disk. Initializes ONNX Runtime environment
with warning-level logging suppressed, single-threaded inference,
and full graph optimization enabled.

**Parameters:**
- `modelPath` — path to `lod_perception.onnx`

**Throws:** `std::runtime_error` if model cannot be loaded.

---

## Methods

### predict()

```cpp
float predict(const std::array<float, FEATURE_DIM>& features) const;
```

Runs single-sample inference. Returns predicted perceptual quality
score in [0, 1] where 1.0 = highest quality (LOD0 equivalent).

**Parameters:**
- `features` — normalized 38-dimensional feature vector produced
  by `FeatureExtractor::extract()`

**Returns:** Quality score in [0, 1]

**Latency:** ~0.006ms on CPU (RTX 4060 Laptop)

---

### predictBatch()

```cpp
std::vector<float> predictBatch(const std::vector<float>& features,
                                 int batchSize) const;
```

Runs batch inference on multiple samples in one ONNX Runtime call.
More efficient than calling `predict()` in a loop when scoring
multiple LOD levels or multiple meshes simultaneously.

**Parameters:**
- `features` — flat float array of shape `[batchSize * FEATURE_DIM]`
- `batchSize` — number of samples in the batch

**Returns:** Vector of `batchSize` quality scores

**Throws:** `std::runtime_error` if feature vector size does not
match `batchSize * FEATURE_DIM`

---

### benchmark()

```cpp
double benchmark(int n = 1000) const;
```

Measures mean single-sample inference latency in milliseconds.
Runs 50 warmup iterations then `n` timed iterations.

**Parameters:**
- `n` — number of timed runs (default: 1000)

**Returns:** Mean latency in milliseconds

---

### isLoaded()

```cpp
bool isLoaded() const;
```

Returns `true` if the model was successfully loaded.

---

## Usage Example

```cpp
#include "ml/ONNXInference.h"

// Load model
acuity::ONNXInference model("data/models/lod_perception.onnx");

// Single inference
std::array<float, 38> features{};
// ... fill features from FeatureExtractor ...
float score = model.predict(features);

// Batch inference (5 LOD levels at once)
std::vector<float> batchFeatures(5 * 38);
// ... fill batch features ...
auto scores = model.predictBatch(batchFeatures, 5);

// Benchmark
double latencyMs = model.benchmark(1000);
std::cout << "Latency: " << latencyMs << " ms\n";
```

---

## Notes

- Non-copyable — owns ONNX Runtime session resources
- Thread safety: do not call from multiple threads simultaneously
- ONNX schema warnings are suppressed during session construction
  by temporarily redirecting stderr to NUL (Windows only)
- On Windows, model path is converted to `wchar_t*` as required
  by the ONNX Runtime Win32 API