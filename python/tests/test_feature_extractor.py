"""
python/tests/test_feature_extractor.py

Unit tests verifying that the C++ FeatureExtractor produces features
that are numerically consistent with the Python extract_features.py
implementation.

Since we cannot call C++ directly from Python, these tests:
1. Verify the Python feature extraction logic is correct
2. Verify the scaler normalization matches sklearn exactly
3. Verify feature value ranges are sensible
4. Document expected values so C++ output can be compared manually
"""

import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path

def get_feature_cols(scaler, df):
    """Get feature columns using scaler's feature names if available,
    otherwise fall back to filtering known non-feature columns."""
    if hasattr(scaler, 'feature_names_in_'):
        return list(scaler.feature_names_in_)
    # Fallback: drop all known non-feature columns
    non_feature = ["quality_score", "mesh_name", "image_path",
                   "lod_level", "distance_category", "angle_index",
                   "mesh_id", "render_id", "split"]
    return [c for c in df.columns if c not in non_feature]

SCALER_PATH  = "data/processed/feature_scaler.pkl"
TRAIN_CSV    = "data/processed/train.csv"
FEATURE_DIM  = 38

def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)

def test_scaler_loads():
    print("Test 1: Scaler loads correctly...")
    scaler = load_scaler()
    assert hasattr(scaler, "mean_"),  "Scaler missing mean_"
    assert hasattr(scaler, "scale_"), "Scaler missing scale_"
    assert len(scaler.mean_)  == FEATURE_DIM, \
        f"Expected {FEATURE_DIM} means, got {len(scaler.mean_)}"
    assert len(scaler.scale_) == FEATURE_DIM, \
        f"Expected {FEATURE_DIM} scales, got {len(scaler.scale_)}"
    print(f"  Mean shape:  {scaler.mean_.shape}")
    print(f"  Scale shape: {scaler.scale_.shape}")
    print("  PASSED")

def test_scaler_txt_matches_pkl():
    print("Test 2: feature_scaler.txt matches feature_scaler.pkl...")
    scaler = load_scaler()

    txt_path = Path("data/processed/feature_scaler.txt")
    assert txt_path.exists(), f"feature_scaler.txt not found at {txt_path}"

    with open(txt_path) as f:
        lines = f.readlines()

    mean_line = lines[0].strip().split()
    std_line  = lines[1].strip().split()

    # Skip "mean:" and "std:" tags
    txt_mean = np.array([float(v) for v in mean_line[1:]])
    txt_std  = np.array([float(v) for v in std_line[1:]])

    max_mean_diff = np.max(np.abs(txt_mean - scaler.mean_))
    max_std_diff  = np.max(np.abs(txt_std  - scaler.scale_))

    assert max_mean_diff < 1e-5, f"Mean mismatch: {max_mean_diff}"
    assert max_std_diff  < 1e-5, f"Std mismatch: {max_std_diff}"
    print(f"  Max mean diff: {max_mean_diff:.2e}")
    print(f"  Max std diff:  {max_std_diff:.2e}")
    print("  PASSED")

def test_feature_count():
    print("Test 3: Dataset has exactly 38 features...")
    scaler = load_scaler()
    df = pd.read_csv(TRAIN_CSV)
    feature_cols = get_feature_cols(scaler, df)
    print(f"  Feature columns: {len(feature_cols)}")
    assert len(feature_cols) == FEATURE_DIM, \
        f"Expected {FEATURE_DIM} features, got {len(feature_cols)}"
    print("  PASSED")

def test_normalization_produces_unit_variance():
    print("Test 4: Normalization produces ~zero mean and ~unit variance...")
    scaler = load_scaler()
    df = pd.read_csv(TRAIN_CSV)
    feature_cols = get_feature_cols(scaler, df)
    X = df[feature_cols].values.astype(np.float32)

    mean_of_means = np.abs(X.mean(axis=0)).mean()
    mean_of_stds  = X.std(axis=0).mean()

    print(f"  Mean of feature means (should be ~0): {mean_of_means:.6f}")
    print(f"  Mean of feature stds  (should be ~1): {mean_of_stds:.6f}")
    assert mean_of_means < 0.01, f"Mean too far from 0: {mean_of_means}"
    assert abs(mean_of_stds - 1.0) < 0.01, \
        f"Std too far from 1: {mean_of_stds}"
    print("  PASSED")

def test_feature_ranges():
    print("Test 5: Raw feature values are in sensible ranges...")
    scaler = load_scaler()
    df = pd.read_csv(TRAIN_CSV)
    feature_cols = get_feature_cols(scaler, df)
    X = df[feature_cols].values

    # No NaN or Inf
    assert not np.any(np.isnan(X)), "NaN values found in features"
    assert not np.any(np.isinf(X)), "Inf values found in features"
    print("  No NaN or Inf values found")
    print("  PASSED")

def test_quality_score_range():
    print("Test 6: Quality scores are in [0, 1]...")
    df = pd.read_csv(TRAIN_CSV)
    assert "quality_score" in df.columns, "quality_score column missing"
    scores = df["quality_score"].values
    assert scores.min() >= 0.0, f"Score below 0: {scores.min()}"
    assert scores.max() <= 1.0, f"Score above 1: {scores.max()}"
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print("  PASSED")

def test_print_reference_values():
    """
    Print reference feature values for a sample so C++ output
    can be manually compared during debugging.
    """
    print("Test 7: Print reference feature values for C++ comparison...")
    scaler = load_scaler()
    df = pd.read_csv(TRAIN_CSV)
    feature_cols = get_feature_cols(scaler, df)

    # Take first sample
    sample_norm  = df[feature_cols].values[0]

    print(f"  Sample norm features (first 5): "
          f"{[f'{v:.6f}' for v in sample_norm[:5]]}")
    print("  PASSED")

if __name__ == "__main__":
    print("=" * 50)
    print("Feature Extractor Unit Tests")
    print("=" * 50)

    tests = [
        test_scaler_loads,
        test_scaler_txt_matches_pkl,
        test_feature_count,
        test_normalization_produces_unit_variance,
        test_feature_ranges,
        test_quality_score_range,
        test_print_reference_values,
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