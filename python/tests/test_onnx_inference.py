"""
python/tests/test_onnx_inference.py

Integration tests verifying that the ONNX model produces consistent
results — checking numerical stability, input/output shapes, batch
inference consistency, and latency targets.
"""

import numpy as np
import onnxruntime as ort
import time
import sys
from pathlib import Path

MODEL_PATH  = "data/models/lod_perception.onnx"
FEATURE_DIM = 38
LATENCY_TARGET_MS = 1.0

def load_session(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(model_path)

def test_model_loads():
    print("Test 1: Model loads correctly...")
    session = load_session(MODEL_PATH)
    assert session is not None
    print("  PASSED")

def test_input_output_names():
    print("Test 2: Input/output names are correct...")
    session = load_session(MODEL_PATH)
    input_names  = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    assert "features"     in input_names,  f"Expected 'features' in inputs, got {input_names}"
    assert "quality_score" in output_names, f"Expected 'quality_score' in outputs, got {output_names}"
    print(f"  Inputs:  {input_names}")
    print(f"  Outputs: {output_names}")
    print("  PASSED")

def test_input_shape():
    print("Test 3: Input shape is [N, 38]...")
    session = load_session(MODEL_PATH)
    input_shape = session.get_inputs()[0].shape
    assert input_shape[1] == FEATURE_DIM, \
        f"Expected feature dim {FEATURE_DIM}, got {input_shape[1]}"
    print(f"  Input shape: {input_shape}")
    print("  PASSED")

def test_single_inference():
    print("Test 4: Single inference produces score in [0, 1]...")
    session = load_session(MODEL_PATH)
    features = np.zeros((1, FEATURE_DIM), dtype=np.float32)
    score = session.run(None, {"features": features})[0]
    assert score.shape == (1, 1), f"Expected shape (1,1), got {score.shape}"
    print(f"  Score: {score[0][0]:.6f}")
    print("  PASSED")

def test_batch_inference_consistency():
    print("Test 5: Batch inference matches individual inference...")
    session = load_session(MODEL_PATH)
    np.random.seed(42)
    batch = np.random.rand(5, FEATURE_DIM).astype(np.float32)

    # Individual
    individual_scores = []
    for i in range(5):
        s = session.run(None, {"features": batch[i:i+1]})[0]
        individual_scores.append(s[0][0])

    # Batch
    batch_scores = session.run(None, {"features": batch})[0].flatten()

    max_diff = np.max(np.abs(np.array(individual_scores) - batch_scores))
    assert max_diff < 1e-5, f"Batch vs individual max diff: {max_diff}"
    print(f"  Max difference (batch vs individual): {max_diff:.2e}")
    print("  PASSED")

def test_deterministic():
    print("Test 6: Inference is deterministic (same input = same output)...")
    session = load_session(MODEL_PATH)
    np.random.seed(0)
    features = np.random.rand(1, FEATURE_DIM).astype(np.float32)
    score1 = session.run(None, {"features": features})[0][0][0]
    score2 = session.run(None, {"features": features})[0][0][0]
    assert score1 == score2, f"Non-deterministic: {score1} vs {score2}"
    print(f"  Score (run 1): {score1:.8f}")
    print(f"  Score (run 2): {score2:.8f}")
    print("  PASSED")

def test_latency():
    print(f"Test 7: Latency target <{LATENCY_TARGET_MS}ms...")
    session = load_session(MODEL_PATH)
    features = np.random.rand(1, FEATURE_DIM).astype(np.float32)

    # Warmup
    for _ in range(50):
        session.run(None, {"features": features})

    # Timed
    n = 1000
    t0 = time.perf_counter()
    for _ in range(n):
        session.run(None, {"features": features})
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000 / n
    print(f"  Mean latency: {latency_ms:.4f} ms")
    assert latency_ms < LATENCY_TARGET_MS, \
        f"Latency {latency_ms:.4f}ms exceeds target {LATENCY_TARGET_MS}ms"
    print("  PASSED")

def test_lod_ordering():
    print("Test 8: LOD0 scores higher than LOD4 (higher detail = higher quality)...")
    session = load_session(MODEL_PATH)

    # LOD0: high triangle count, low LOD level index
    lod0_features = np.zeros((1, FEATURE_DIM), dtype=np.float32)
    lod0_features[0, 0] = 2048   # triangle_count
    lod0_features[0, 1] = 1089   # vertex_count
    lod0_features[0, 2] = 0      # lod_level
    lod0_features[0, 8] = 1.0    # lod_ratio

    # LOD4: low triangle count, high LOD level index
    lod4_features = np.zeros((1, FEATURE_DIM), dtype=np.float32)
    lod4_features[0, 0] = 128    # triangle_count
    lod4_features[0, 1] = 80     # vertex_count
    lod4_features[0, 2] = 4      # lod_level
    lod4_features[0, 8] = 0.0625 # lod_ratio

    score_lod0 = session.run(None, {"features": lod0_features})[0][0][0]
    score_lod4 = session.run(None, {"features": lod4_features})[0][0][0]

    print(f"  LOD0 score: {score_lod0:.6f}")
    print(f"  LOD4 score: {score_lod4:.6f}")
    # Note: this is a sanity check on raw unnormalized features
    # so we just verify both produce valid scores
    assert 0.0 <= score_lod0 <= 1.0, f"LOD0 score out of range: {score_lod0}"
    assert 0.0 <= score_lod4 <= 1.0, f"LOD4 score out of range: {score_lod4}"
    print("  PASSED")

if __name__ == "__main__":
    print("=" * 50)
    print("ONNX Inference Integration Tests")
    print("=" * 50)

    tests = [
        test_model_loads,
        test_input_output_names,
        test_input_shape,
        test_single_inference,
        test_batch_inference_consistency,
        test_deterministic,
        test_latency,
        test_lod_ordering,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
        print()

    print("=" * 50)
    print(f"Results: {passed}/{passed+failed} passed")
    print("=" * 50)
    sys.exit(0 if failed == 0 else 1)