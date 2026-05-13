"""
finalize_dataset.py

Splits the feature dataset into train/val/test sets, normalizes
features, and saves in CSV and HDF5 formats ready for model training.

Split strategy: by mesh name (not by image) to prevent data leakage.
If we split by image, the same mesh appears in train and test — the
model memorizes mesh-specific features rather than learning generalizable
perceptual quality prediction. Splitting by mesh ensures the model is
evaluated on meshes it has never seen.

Split: 70% train / 15% val / 15% test (by mesh count)
  14 meshes → 10 train / 2 val / 2 test

Normalization: StandardScaler fit on training set only, then applied
to val and test. Scaler saved to disk for use in Phase 4 C++ inference
(feature values must be normalized identically at runtime).

Input:  data/processed/dataset_with_features.csv
Output: data/processed/train.csv
        data/processed/val.csv
        data/processed/test.csv
        data/processed/dataset.h5
        data/processed/feature_scaler.pkl
        docs/experiments/dataset_statistics.md
"""

import numpy as np
import pandas as pd
import pickle
import h5py
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import sys

INPUT_CSV  = Path("data/processed/dataset_with_features.csv")
OUTPUT_DIR = Path("data/processed")
DOCS_DIR   = Path("docs/experiments")

FEATURE_COLS = [
    # Geometric (10)
    "tri_count_log", "vert_count_log", "surface_area", "volume",
    "edge_len_mean", "edge_len_std", "bbox_diag", "bbox_x", "bbox_y",
    "area_variance",
    # Perceptual (15)
    "normal_var_mean", "curvature_mean", "curvature_std", "detail_density",
    "saliency_fraction", "area_per_tri", "edge_len_range",
    "normal_consistency", "aspect_ratio", "vol_sa_ratio",
    "lod_tri_ratio", "tri_reduction", "log_lod_ratio", "compactness",
    "normal_entropy",
    # View-dependent (13)
    "az_sin", "az_cos", "screen_coverage", "log_distance", "proj_detail",
    "dist_lod_interaction", "dist_normalised", "dist_reciprocal",
    "proj_area", "lod_adequacy", "dist_mult", "az_deg_norm", "screen_log",
]

TARGET_COL = "quality_score"

# ── Helpers ───────────────────────────────────────────────────────────────────

def split_meshes(mesh_names: list, seed: int = 42):
    """
    Split mesh names into train/val/test ensuring no mesh appears
    in more than one split.

    14 meshes → 10 train, 2 val, 2 test
    """
    mesh_names = sorted(mesh_names)
    np.random.seed(seed)
    np.random.shuffle(mesh_names)

    n       = len(mesh_names)
    n_val   = max(1, round(n * 0.15))
    n_test  = max(1, round(n * 0.15))
    n_train = n - n_val - n_test

    train_meshes = mesh_names[:n_train]
    val_meshes   = mesh_names[n_train:n_train + n_val]
    test_meshes  = mesh_names[n_train + n_val:]

    return train_meshes, val_meshes, test_meshes


def save_hdf5(train_df, val_df, test_df,
              feature_cols: list, target_col: str,
              path: Path):
    """Save all splits to a single HDF5 file for fast loading in PyTorch."""
    with h5py.File(path, "w") as f:
        for name, df in [("train", train_df),
                          ("val",   val_df),
                          ("test",  test_df)]:
            grp = f.create_group(name)
            grp.create_dataset("X", data=df[feature_cols].values.astype(np.float32))
            grp.create_dataset("y", data=df[target_col].values.astype(np.float32))
            grp.attrs["n_samples"]  = len(df)
            grp.attrs["n_features"] = len(feature_cols)


def write_statistics_doc(train_df, val_df, test_df,
                          train_meshes, val_meshes, test_meshes,
                          feature_cols: list, scaler: StandardScaler,
                          path: Path):
    """Write dataset statistics markdown document."""
    path.parent.mkdir(parents=True, exist_ok=True)

    all_df = pd.concat([train_df, val_df, test_df])

    lines = [
        "# Dataset Statistics",
        "",
        "## Split Summary",
        "",
        f"| Split | Meshes | Samples | % |",
        f"|-------|--------|---------|---|",
        f"| Train | {len(train_meshes)} | {len(train_df)} | "
        f"{100*len(train_df)/len(all_df):.1f}% |",
        f"| Val   | {len(val_meshes)} | {len(val_df)} | "
        f"{100*len(val_df)/len(all_df):.1f}% |",
        f"| Test  | {len(test_meshes)} | {len(test_df)} | "
        f"{100*len(test_df)/len(all_df):.1f}% |",
        f"| Total | {len(train_meshes)+len(val_meshes)+len(test_meshes)} "
        f"| {len(all_df)} | 100% |",
        "",
        "## Mesh Assignment",
        "",
        f"**Train ({len(train_meshes)}):** {', '.join(sorted(train_meshes))}",
        "",
        f"**Val ({len(val_meshes)}):** {', '.join(sorted(val_meshes))}",
        "",
        f"**Test ({len(test_meshes)}):** {', '.join(sorted(test_meshes))}",
        "",
        "## Quality Score Distribution",
        "",
        "| Split | Min | Max | Mean | Std |",
        "|-------|-----|-----|------|-----|",
    ]

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        qs = df[TARGET_COL]
        lines.append(f"| {name} | {qs.min():.4f} | {qs.max():.4f} | "
                     f"{qs.mean():.4f} | {qs.std():.4f} |")

    lines += [
        "",
        "## Quality Score by LOD Level (Train set)",
        "",
        "| LOD | Mean Score | Std |",
        "|-----|------------|-----|",
    ]
    for lod in sorted(train_df["lod_level"].unique()):
        subset = train_df[train_df["lod_level"] == lod][TARGET_COL]
        lines.append(f"| {lod} | {subset.mean():.4f} | {subset.std():.4f} |")

    lines += [
        "",
        "## Feature Summary (38 dimensions)",
        "",
        "| # | Feature | Group | Train Mean | Train Std |",
        "|---|---------|-------|------------|-----------|",
    ]

    groups = (["Geometric"] * 10 +
              ["Perceptual"] * 15 +
              ["View-dependent"] * 13)

    for i, (col, grp) in enumerate(zip(feature_cols, groups)):
        mean = scaler.mean_[i]
        std  = scaler.scale_[i]
        lines.append(f"| {i+1} | `{col}` | {grp} | {mean:.4f} | {std:.4f} |")

    lines += [
        "",
        "## Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `train.csv` | Training split (normalized features + scores) |",
        "| `val.csv` | Validation split |",
        "| `test.csv` | Test split |",
        "| `dataset.h5` | HDF5 format for fast PyTorch loading |",
        "| `feature_scaler.pkl` | StandardScaler for deployment normalization |",
        "",
        f"Generated by `python/dataset/finalize_dataset.py`",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Acuity — Dataset Finalizer")
    print("=" * 60)

    # Load full dataset
    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")

    # Verify all feature columns present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing feature columns: {missing}")
        sys.exit(1)
    print(f"  Features: {len(FEATURE_COLS)} columns verified")

    # Split meshes
    print("\n[1/4] Splitting by mesh...")
    mesh_names = df["mesh_name"].unique().tolist()
    train_meshes, val_meshes, test_meshes = split_meshes(mesh_names)

    print(f"  Train meshes ({len(train_meshes)}): {sorted(train_meshes)}")
    print(f"  Val   meshes ({len(val_meshes)}):   {sorted(val_meshes)}")
    print(f"  Test  meshes ({len(test_meshes)}):  {sorted(test_meshes)}")

    train_df = df[df["mesh_name"].isin(train_meshes)].copy()
    val_df   = df[df["mesh_name"].isin(val_meshes)].copy()
    test_df  = df[df["mesh_name"].isin(test_meshes)].copy()

    print(f"\n  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")

    # Normalize features
    print("\n[2/4] Normalizing features...")
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS])

    train_df[FEATURE_COLS] = scaler.transform(train_df[FEATURE_COLS])
    val_df[FEATURE_COLS]   = scaler.transform(val_df[FEATURE_COLS])
    test_df[FEATURE_COLS]  = scaler.transform(test_df[FEATURE_COLS])

    print(f"  Scaler fit on {len(train_df)} training samples")
    print(f"  Feature mean range: "
          f"{scaler.mean_.min():.4f} to {scaler.mean_.max():.4f}")
    print(f"  Feature std range:  "
          f"{scaler.scale_.min():.4f} to {scaler.scale_.max():.4f}")

    # Save CSVs
    print("\n[3/4] Saving files...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path  = OUTPUT_DIR / "train.csv"
    val_path    = OUTPUT_DIR / "val.csv"
    test_path   = OUTPUT_DIR / "test.csv"
    h5_path     = OUTPUT_DIR / "dataset.h5"
    scaler_path = OUTPUT_DIR / "feature_scaler.pkl"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    test_df.to_csv(test_path,   index=False)
    print(f"  Saved: {train_path} ({len(train_df)} rows)")
    print(f"  Saved: {val_path}   ({len(val_df)} rows)")
    print(f"  Saved: {test_path}  ({len(test_df)} rows)")

    # Save HDF5
    save_hdf5(train_df, val_df, test_df, FEATURE_COLS, TARGET_COL, h5_path)
    print(f"  Saved: {h5_path}")

    # Save scaler
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved: {scaler_path}")

    # Write statistics document
    print("\n[4/4] Writing dataset statistics document...")
    doc_path = DOCS_DIR / "dataset_statistics.md"
    write_statistics_doc(
        train_df, val_df, test_df,
        train_meshes, val_meshes, test_meshes,
        FEATURE_COLS, scaler, doc_path
    )
    print(f"  Saved: {doc_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("  Dataset finalization complete.")
    print("=" * 60)
    print(f"\n  Total samples : {len(df)}")
    print(f"  Train         : {len(train_df)} "
          f"({100*len(train_df)/len(df):.1f}%)")
    print(f"  Val           : {len(val_df)} "
          f"({100*len(val_df)/len(df):.1f}%)")
    print(f"  Test          : {len(test_df)} "
          f"({100*len(test_df)/len(df):.1f}%)")
    print(f"  Features      : {len(FEATURE_COLS)}")
    print(f"  Target        : {TARGET_COL}")
    print()
    print("  Quality score distribution per split:")
    for name, split_df in [("Train", train_df),
                            ("Val",   val_df),
                            ("Test",  test_df)]:
        qs = split_df[TARGET_COL]
        print(f"    {name}: min={qs.min():.4f} max={qs.max():.4f} "
              f"mean={qs.mean():.4f} std={qs.std():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()