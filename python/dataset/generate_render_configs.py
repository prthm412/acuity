"""
python/dataset/generate_render_configs.py

Generates camera pose configurations for batch rendering.
Phase 2, Step 2.2 — Task 1.

For each mesh × LOD combination, we render from 40 camera poses:
- 8 horizontal angles:  0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
- 5 distances:          very close, close, medium, far, very far
  (expressed as multipliers of the mesh bounding sphere radius)

Output: data/processed/render_configs.json
This file is read by the C++ BatchRenderer in Step 2.2 Task 2.

Why these specific poses?
- 8 angles gives full 360° coverage at 45° intervals — captures all sides
- 5 distances tests LOD transitions: some distances should trigger lower LODs,
  others should keep high LODs. This is exactly the variation we need for
  training the perception model in Phase 3.
- Using radius multipliers (not absolute distances) makes poses mesh-agnostic:
  a small teapot and a large dragon get equivalent relative coverage.
"""

import json
import math
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Camera pose parameters ─────────────────────────────────────────────────

# 8 horizontal angles in degrees (full 360° coverage)
ANGLES_DEG = [0, 45, 90, 135, 180, 225, 270, 315]

# 5 distance multipliers relative to mesh bounding sphere radius.
# radius * multiplier = camera distance from mesh center.
# 1.5 = very close (fills frame), 6.0 = very far (tiny in frame)
DISTANCE_MULTIPLIERS = [2.0, 3.0, 4.0, 5.0, 7.0]

# Fixed elevation angle in degrees (camera looks slightly down at mesh)
ELEVATION_DEG = 20.0

# Render image resolution
IMAGE_WIDTH  = 512
IMAGE_HEIGHT = 512

# LOD levels (matches LODGenerator from Phase 1 Step 1.4)
LOD_LEVELS = [0, 1, 2, 3, 4]  # 0=100%, 1=50%, 2=25%, 3=12.5%, 4=6.25%
LOD_RATIOS = [1.0, 0.5, 0.25, 0.125, 0.0625]


def generate_poses() -> list:
    """
    Generate all 40 camera poses.
    Each pose is a dict with azimuth angle and distance multiplier.
    The C++ renderer computes the actual 3D position from these
    plus the mesh bounding sphere radius at render time.
    """
    poses = []
    pose_id = 0
    for dist_idx, dist_mult in enumerate(DISTANCE_MULTIPLIERS):
        for angle_idx, angle_deg in enumerate(ANGLES_DEG):
            poses.append({
                "pose_id"          : pose_id,
                "azimuth_deg"      : angle_deg,
                "elevation_deg"    : ELEVATION_DEG,
                "distance_mult"    : dist_mult,
                "distance_label"   : ["very_close", "close", "medium",
                                      "far", "very_far"][dist_idx],
                "angle_label"      : f"az{angle_deg:03d}",
            })
            pose_id += 1
    return poses


def generate_config(meshes: list, poses: list) -> dict:
    """
    Generate the full render configuration.
    Contains all poses, LOD info, and image settings.
    The mesh list is populated from mesh_manifest if available,
    otherwise left empty for the C++ renderer to fill.
    """
    return {
        "version"      : "1.0",
        "image_width"  : IMAGE_WIDTH,
        "image_height" : IMAGE_HEIGHT,
        "lod_levels"   : LOD_LEVELS,
        "lod_ratios"   : LOD_RATIOS,
        "poses"        : poses,
        "meshes"       : meshes,
        "total_renders": len(meshes) * len(LOD_LEVELS) * len(poses),
        "notes": (
            f"{len(poses)} poses = "
            f"{len(ANGLES_DEG)} angles x {len(DISTANCE_MULTIPLIERS)} distances. "
            f"Output naming: {{mesh_name}}_lod{{lod}}_{{angle_label}}_{{distance_label}}.png"
        )
    }


def load_mesh_list() -> list:
    """Load mesh paths from mesh_manifest.csv if it exists."""
    manifest = ROOT / "assets" / "models" / "mesh_manifest.csv"
    if not manifest.exists():
        print("  [WARN] mesh_manifest.csv not found — mesh list will be empty.")
        print("         Run python/dataset/generate_manifest.py first.")
        return []

    meshes = []
    lines = manifest.read_text().strip().splitlines()
    for line in lines[1:]:  # skip header
        parts = line.split(",")
        if len(parts) >= 2:
            meshes.append({
                "name": parts[0].strip(),
                "path": parts[1].strip()
            })
    print(f"  Loaded {len(meshes)} meshes from manifest.")
    return meshes


def print_summary(poses: list, meshes: list):
    print(f"\n── Summary ───────────────────────────────────────────────────")
    print(f"  Poses         : {len(poses)} ({len(ANGLES_DEG)} angles × "
          f"{len(DISTANCE_MULTIPLIERS)} distances)")
    print(f"  LOD levels    : {len(LOD_LEVELS)}")
    print(f"  Meshes        : {len(meshes)}")
    if meshes:
        total = len(meshes) * len(LOD_LEVELS) * len(poses)
        print(f"  Total renders : {total}")
    print(f"\n  Angles  : {ANGLES_DEG}")
    print(f"  Distances: {DISTANCE_MULTIPLIERS} × bounding radius")
    print(f"  Elevation: {ELEVATION_DEG}°")
    print(f"  Resolution: {IMAGE_WIDTH}×{IMAGE_HEIGHT}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Acuity — Render Config Generator")
    print("  Phase 2, Step 2.2 — Task 1")
    print("=" * 60)

    poses  = generate_poses()
    meshes = load_mesh_list()
    config = generate_config(meshes, poses)

    output_path = OUTPUT_DIR / "render_configs.json"
    output_path.write_text(json.dumps(config, indent=2))
    print(f"\n  Written: {output_path.relative_to(ROOT)}")

    print_summary(poses, meshes)