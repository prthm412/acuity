"""
extract_features.py

Extracts 38 features per sample for the perceptual LOD quality dataset.

Feature groups:
  Geometric (10):       triangle count, vertex count, surface area,
                        volume, edge length stats, bounding box stats
  Perceptual (15):      curvature stats, normal variation, saliency proxy,
                        detail density, silhouette complexity
  View-dependent (13):  camera distance, azimuth, screen coverage proxy,
                        LOD ratio, projected detail density,
                        distance-LOD interaction terms

Input:  data/processed/dataset_raw.csv
        data/processed/lod_cache/
Output: data/processed/dataset_with_features.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time

CACHE_DIR  = Path("data/processed/lod_cache")
INPUT_CSV  = Path("data/processed/dataset_raw.csv")
OUTPUT_CSV = Path("data/processed/dataset_with_features.csv")

DIST_MULTIPLIERS = {
    "very_close": 1.2,
    "close":      2.0,
    "medium":     3.5,
    "far":        6.0,
    "very_far":   9.0,
}

# ── OBJ loader ────────────────────────────────────────────────────────────────

def load_obj(path: Path):
    """
    Load vertices and faces from OBJ file.
    Returns verts (N,3) float32 and faces (M,3) int32.
    Handles v//vn face format from our LOD cache.
    """
    verts  = []
    normals = []
    faces  = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]),
                               float(parts[2]),
                               float(parts[3])])
            elif line.startswith("vn "):
                parts = line.split()
                normals.append([float(parts[1]),
                                 float(parts[2]),
                                 float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                # Handle v, v/vt, v//vn, v/vt/vn
                indices = []
                for p in parts:
                    indices.append(int(p.split("/")[0]) - 1)
                if len(indices) == 3:
                    faces.append(indices)

    verts   = np.array(verts,   dtype=np.float32)
    normals = np.array(normals, dtype=np.float32) if normals else None
    faces   = np.array(faces,   dtype=np.int32)
    return verts, normals, faces


# ── Geometric feature helpers ─────────────────────────────────────────────────

def triangle_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return per-triangle area array."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def edge_lengths(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return all unique edge lengths."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    return np.concatenate([e0, e1, e2])


def signed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Approximate signed volume via divergence theorem."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    vol = np.sum(v0 * np.cross(v1, v2)) / 6.0
    return abs(float(vol))


def compute_vertex_normals(verts: np.ndarray,
                            faces: np.ndarray) -> np.ndarray:
    """Compute smooth vertex normals by averaging face normals."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    face_normals = face_normals / norms

    vertex_normals = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)

    lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    lengths = np.where(lengths == 0, 1e-8, lengths)
    return vertex_normals / lengths


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_mesh_features(verts: np.ndarray,
                           normals_loaded,
                           faces: np.ndarray,
                           lod0_tri_count: int) -> dict:
    """
    Extract geometric (10) and perceptual (15) features from a mesh.
    These are mesh-level features independent of viewpoint.
    """
    n_tris  = len(faces)
    n_verts = len(verts)

    # ── Geometric features (10) ───────────────────────────────────────────────

    # 1. Triangle count (log-scaled for better numeric range)
    tri_count_log = float(np.log1p(n_tris))

    # 2. Vertex count (log-scaled)
    vert_count_log = float(np.log1p(n_verts))

    # 3. Surface area (total)
    areas        = triangle_areas(verts, faces)
    surface_area = float(np.sum(areas))

    # 4. Volume
    volume = signed_volume(verts, faces)

    # 5-6. Edge length mean and std
    elens         = edge_lengths(verts, faces)
    edge_len_mean = float(np.mean(elens))
    edge_len_std  = float(np.std(elens))

    # 7-9. Bounding box dimensions and diagonal
    bbox_min  = verts.min(axis=0)
    bbox_max  = verts.max(axis=0)
    bbox_dims = bbox_max - bbox_min
    bbox_diag = float(np.linalg.norm(bbox_dims))
    bbox_x    = float(bbox_dims[0])
    bbox_y    = float(bbox_dims[1])

    # 10. Triangle area variance (uniformity of tessellation)
    area_variance = float(np.var(areas))

    # ── Perceptual features (15) ──────────────────────────────────────────────

    # Compute vertex normals if not loaded from file
    if normals_loaded is not None and len(normals_loaded) == n_verts:
        vnormals = normals_loaded
    else:
        vnormals = compute_vertex_normals(verts, faces)

    # 11. Normal variation — measures surface smoothness
    # Compute per-face normal dot products with neighbours
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn_norms = np.linalg.norm(fn, axis=1, keepdims=True)
    fn_norms = np.where(fn_norms == 0, 1e-8, fn_norms)
    fn = fn / fn_norms

    normal_var_mean = float(np.mean(np.var(fn, axis=0)))

    # 12. Curvature proxy — variance of vertex normal dot products
    # For each face, average the dot product of vertex normals
    vn0 = vnormals[faces[:, 0]]
    vn1 = vnormals[faces[:, 1]]
    vn2 = vnormals[faces[:, 2]]
    dots01 = np.sum(vn0 * vn1, axis=1)
    dots12 = np.sum(vn1 * vn2, axis=1)
    dots02 = np.sum(vn0 * vn2, axis=1)
    curvature_mean = float(np.mean(1.0 - (dots01 + dots12 + dots02) / 3.0))
    curvature_std  = float(np.std( 1.0 - (dots01 + dots12 + dots02) / 3.0))

    # 14. Detail density — triangles per unit area
    detail_density = float(n_tris / (surface_area + 1e-8))

    # 15. Saliency proxy — fraction of high-curvature faces
    curvature_per_face = 1.0 - (dots01 + dots12 + dots02) / 3.0
    high_curv_threshold = float(np.percentile(curvature_per_face, 75))
    saliency_fraction   = float(
        np.mean(curvature_per_face > high_curv_threshold)
    )

    # 16. Surface area per triangle (compactness)
    area_per_tri = float(surface_area / (n_tris + 1e-8))

    # 17. Edge length range
    edge_len_range = float(elens.max() - elens.min())

    # 18. Normal consistency — mean dot product of adjacent face normals
    # Use face normals directly as proxy
    normal_consistency = float(np.mean(np.abs(fn)))

    # 19. Aspect ratio proxy — mean edge std / mean edge length
    aspect_ratio = float(edge_len_std / (edge_len_mean + 1e-8))

    # 20. Volume to surface area ratio
    vol_sa_ratio = float(volume / (surface_area + 1e-8))

    # 21. LOD triangle ratio relative to LOD 0
    lod_tri_ratio = float(n_tris / (lod0_tri_count + 1e-8))

    # 22. Triangle count reduction (1 - ratio)
    tri_reduction = float(1.0 - lod_tri_ratio)

    # 23. Log LOD ratio
    log_lod_ratio = float(np.log1p(lod_tri_ratio))

    # 24. Mesh compactness (sphericity proxy)
    compactness = float(
        (36 * np.pi * volume ** 2) / (surface_area ** 3 + 1e-8)
    ) if surface_area > 0 else 0.0

    # 25. Normal entropy proxy
    normal_entropy = float(-np.sum(
        np.abs(fn) * np.log(np.abs(fn) + 1e-8)
    ) / (n_tris + 1e-8))

    return {
        # Geometric (10)
        "tri_count_log":    tri_count_log,
        "vert_count_log":   vert_count_log,
        "surface_area":     surface_area,
        "volume":           volume,
        "edge_len_mean":    edge_len_mean,
        "edge_len_std":     edge_len_std,
        "bbox_diag":        bbox_diag,
        "bbox_x":           bbox_x,
        "bbox_y":           bbox_y,
        "area_variance":    area_variance,
        # Perceptual (15)
        "normal_var_mean":    normal_var_mean,
        "curvature_mean":     curvature_mean,
        "curvature_std":      curvature_std,
        "detail_density":     detail_density,
        "saliency_fraction":  saliency_fraction,
        "area_per_tri":       area_per_tri,
        "edge_len_range":     edge_len_range,
        "normal_consistency": normal_consistency,
        "aspect_ratio":       aspect_ratio,
        "vol_sa_ratio":       vol_sa_ratio,
        "lod_tri_ratio":      lod_tri_ratio,
        "tri_reduction":      tri_reduction,
        "log_lod_ratio":      log_lod_ratio,
        "compactness":        compactness,
        "normal_entropy":     normal_entropy,
    }


def extract_view_features(azimuth: int,
                           dist_label: str,
                           dist_mult: float,
                           lod_tri_ratio: float,
                           surface_area: float,
                           bbox_diag: float) -> dict:
    """
    Extract 13 view-dependent features.
    These capture how the viewpoint affects perceived quality.
    """
    az_rad    = float(np.radians(azimuth))
    az_sin    = float(np.sin(az_rad))
    az_cos    = float(np.cos(az_rad))

    # Screen coverage proxy: bbox solid angle at this distance
    # Proportional to bbox_diag^2 / dist^2
    dist_actual      = dist_mult * bbox_diag
    screen_coverage  = float(bbox_diag ** 2 / (dist_actual ** 2 + 1e-8))

    # Log distance
    log_distance = float(np.log1p(dist_mult))

    # Projected detail density: triangles visible per screen area proxy
    proj_detail = float(lod_tri_ratio / (dist_mult ** 2 + 1e-8))

    # Distance-LOD interaction: how well LOD matches distance
    # High when both distance and LOD reduction are large
    dist_lod_interaction = float(dist_mult * (1.0 - lod_tri_ratio))

    # Normalised distance (0=very_close, 1=very_far)
    dist_norm_map = {
        "very_close": 0.0,
        "close":      0.25,
        "medium":     0.5,
        "far":        0.75,
        "very_far":   1.0,
    }
    dist_normalised = dist_norm_map.get(dist_label, 0.5)

    # Reciprocal distance (emphasises close distances)
    dist_reciprocal = float(1.0 / (dist_mult + 1e-8))

    # Surface area projected at this distance
    proj_area = float(surface_area / (dist_mult ** 2 + 1e-8))

    # LOD adequacy: whether LOD is appropriate for this distance
    # LOD 0 at very_far = over-quality; LOD 4 at very_close = under-quality
    lod_adequacy = float(lod_tri_ratio * dist_mult /
                         (dist_mult + 1.0 - lod_tri_ratio + 1e-8))

    return {
        "az_sin":              az_sin,
        "az_cos":              az_cos,
        "screen_coverage":     screen_coverage,
        "log_distance":        log_distance,
        "proj_detail":         proj_detail,
        "dist_lod_interaction": dist_lod_interaction,
        "dist_normalised":     dist_normalised,
        "dist_reciprocal":     dist_reciprocal,
        "proj_area":           proj_area,
        "lod_adequacy":        lod_adequacy,
        "dist_mult":           dist_mult,
        "az_deg_norm":         float(azimuth / 360.0),
        "screen_log":          float(np.log1p(screen_coverage)),
    }


# ── Cache: load mesh features once per mesh/LOD ───────────────────────────────

def build_mesh_cache(df: pd.DataFrame) -> dict:
    """
    Load each unique (mesh_name, lod_level) combination once
    and cache the extracted features. Avoids reloading OBJ files
    for every row.
    """
    cache = {}
    combos = df[["mesh_name", "lod_level"]].drop_duplicates()
    total  = len(combos)

    print(f"  Loading {total} mesh/LOD combinations from cache...")
    t = time.time()

    for _, row in combos.iterrows():
        mesh_name = row["mesh_name"]
        lod_idx   = int(row["lod_level"])
        key       = (mesh_name, lod_idx)

        lod_path  = CACHE_DIR / f"{mesh_name}_lod{lod_idx}.obj"
        lod0_path = CACHE_DIR / f"{mesh_name}_lod0.obj"

        if not lod_path.exists():
            print(f"  [WARN] Missing: {lod_path.name}")
            continue

        verts, normals, faces = load_obj(lod_path)

        # Get LOD 0 triangle count for ratio computation
        if lod_idx == 0:
            lod0_tri_count = len(faces)
        else:
            _, _, lod0_faces = load_obj(lod0_path)
            lod0_tri_count   = len(lod0_faces)

        feats = extract_mesh_features(verts, normals, faces, lod0_tri_count)
        cache[key] = feats

    print(f"  Done in {time.time()-t:.1f}s  ({total} entries cached)")
    return cache


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Acuity — Feature Extractor")
    print("=" * 60)

    # Load base dataset
    df = pd.read_csv(INPUT_CSV)
    print(f"  Input rows : {len(df)}")

    # Build mesh feature cache
    print("\n[1/3] Building mesh feature cache...")
    mesh_cache = build_mesh_cache(df)

    # Extract all features
    print(f"\n[2/3] Extracting features for {len(df)} samples...")
    t_start = time.time()
    all_features = []

    for idx, row in df.iterrows():
        key = (row["mesh_name"], int(row["lod_level"]))
        if key not in mesh_cache:
            continue

        mesh_feats = mesh_cache[key]

        view_feats = extract_view_features(
            azimuth      = int(row["azimuth"]),
            dist_label   = row["distance_label"],
            dist_mult    = float(row["distance_multiplier"]),
            lod_tri_ratio = mesh_feats["lod_tri_ratio"],
            surface_area  = mesh_feats["surface_area"],
            bbox_diag     = mesh_feats["bbox_diag"],
        )

        combined = {**mesh_feats, **view_feats}
        all_features.append(combined)

        if (idx + 1) % 500 == 0:
            print(f"  {idx+1}/{len(df)}", end="\r")

    print(f"\n  Done in {time.time()-t_start:.1f}s")

    # Merge with original dataframe
    feat_df = pd.DataFrame(all_features)
    result  = pd.concat([df.reset_index(drop=True),
                         feat_df.reset_index(drop=True)], axis=1)

    # Verify feature count
    feature_cols = [c for c in result.columns
                    if c not in df.columns]
    print(f"\n  Feature dimensions: {len(feature_cols)}")
    print(f"  Features: {feature_cols}")

    # Save
    print(f"\n[3/3] Saving to {OUTPUT_CSV}...")
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved: {len(result)} rows × {len(result.columns)} columns")

    # Quick sanity check
    print("\n  Feature value ranges (first 5 features):")
    for col in feature_cols[:5]:
        print(f"    {col:25s}  "
              f"min={result[col].min():.4f}  "
              f"max={result[col].max():.4f}  "
              f"mean={result[col].mean():.4f}")

    print("\n" + "=" * 60)
    print("  Feature extraction complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()