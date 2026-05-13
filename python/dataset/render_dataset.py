"""
python/dataset/render_dataset.py

Renders all meshes at all LOD levels from 40 camera poses using PyVista's
offscreen renderer. Works on Windows without EGL.

PyVista wraps VTK which supports Windows offscreen rendering natively
via its own software renderer — no display server required.

Output: data/processed/rendered_images/<mesh>_lod<N>_az<AAA>_<dist>.png
Total:  14 meshes × 5 LODs × 40 poses = 2800 images
"""

import numpy as np
from pathlib import Path
import sys
import time

import pyvista as pv

# Force offscreen rendering before any window is created
pv.OFF_SCREEN = True

# ── Configuration ─────────────────────────────────────────────────────────────

CACHE_DIR        = Path("data/processed/lod_cache")
OUTPUT_DIR       = Path("data/processed/rendered_images")
IMG_W, IMG_H     = 512, 512

AZIMUTHS         = [0, 45, 90, 135, 180, 225, 270, 315]
DIST_LABELS      = ["very_close", "close", "medium", "far", "very_far"]
DIST_MULTIPLIERS = [1.2, 2.0, 3.5, 6.0, 9.0]
LOD_LEVELS       = [0, 1, 2, 3, 4]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_mesh_info(mesh: pv.PolyData):
    """Return centre (3,) and bounding radius (float) of a PyVista mesh."""
    bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    centre = np.array([
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2,
    ])
    verts  = np.asarray(mesh.points)
    radius = float(np.linalg.norm(verts - centre, axis=1).max())
    return centre, radius


def camera_position(azimuth_deg: float, dist: float,
                    centre: np.ndarray, elevation_deg: float = 25.0):
    """Return (eye, focus, up) tuple for PyVista camera setup."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    eye = centre + np.array([
        dist * np.cos(el) * np.cos(az),
        dist * np.sin(el),
        dist * np.cos(el) * np.sin(az),
    ])
    return eye.tolist(), centre.tolist(), [0.0, 1.0, 0.0]


def render_one(mesh: pv.PolyData, eye, focus, up) -> np.ndarray:
    pl = pv.Plotter(off_screen=True, window_size=[IMG_W, IMG_H])
    pl.set_background([0.15, 0.15, 0.15])

    # Force normals recompute — handles low-poly meshes like woody
    mesh = mesh.compute_normals(
        cell_normals=False,
        point_normals=True,
        split_vertices=True,
        inplace=False
    )

    pl.add_mesh(
        mesh,
        color=[0.75, 0.75, 0.75],
        smooth_shading=True,
        specular=0.4,
        specular_power=25,
        ambient=0.25,
        diffuse=0.85,
        show_edges=False,
    )

    pl.camera.position = eye
    pl.camera.focal_point = focus
    pl.camera.up = up
    pl.camera.view_angle = 60.0

    img = pl.screenshot(return_img=True)
    pl.close()
    return img


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mesh_names = sorted(set(
        p.stem.rsplit("_lod", 1)[0]
        for p in CACHE_DIR.glob("*_lod0.obj")
    ))

    if not mesh_names:
        print(f"[ERROR] No LOD cache found in {CACHE_DIR}")
        sys.exit(1)

    total = len(mesh_names) * len(LOD_LEVELS) * len(AZIMUTHS) * len(DIST_LABELS)
    print("=" * 60)
    print("  Acuity — Dataset Renderer (PyVista)")
    print("=" * 60)
    print(f"  Meshes : {len(mesh_names)}")
    print(f"  LODs   : {len(LOD_LEVELS)}")
    print(f"  Poses  : {len(AZIMUTHS)} az × {len(DIST_LABELS)} dist = 40")
    print(f"  Total  : {total} images")
    print(f"  Output : {OUTPUT_DIR}")
    print("=" * 60)

    rendered = 0
    skipped  = 0
    errors   = 0
    t_start  = time.time()

    for mesh_name in mesh_names:
        print(f"\n[{mesh_name}]")

        # Load LOD 0 for bounding info (consistent across all LODs)
        lod0_path = CACHE_DIR / f"{mesh_name}_lod0.obj"
        lod0      = pv.read(str(lod0_path))
        centre, bounding_radius = get_mesh_info(lod0)

        for lod_idx in LOD_LEVELS:
            lod_path = CACHE_DIR / f"{mesh_name}_lod{lod_idx}.obj"
            if not lod_path.exists():
                print(f"  [WARN] {lod_path.name} missing, skipping")
                continue

            mesh = pv.read(str(lod_path))
            # Ensure normals exist for smooth shading
            if mesh.point_data.get_array("Normals") is None:
                mesh.compute_normals(inplace=True)

            tri_count = mesh.n_cells

            for az in AZIMUTHS:
                for dist_label, dist_mult in zip(DIST_LABELS, DIST_MULTIPLIERS):

                    out_name = (f"{mesh_name}_lod{lod_idx}"
                                f"_az{az:03d}_{dist_label}.png")
                    out_path = OUTPUT_DIR / out_name

                    if out_path.exists():
                        skipped += 1
                        continue

                    dist = dist_mult * bounding_radius
                    eye, focus, up = camera_position(az, dist, centre)

                    try:
                        img = render_one(mesh, eye, focus, up)
                        # pyvista screenshot returns numpy array — save with pillow
                        from PIL import Image
                        Image.fromarray(img).save(str(out_path))
                        rendered += 1
                    except Exception as e:
                        print(f"  [ERROR] {out_name}: {e}")
                        errors += 1

            print(f"  LOD {lod_idx}: {tri_count:>7,} cells — done")

        elapsed = time.time() - t_start
        rate    = rendered / max(elapsed, 0.001)
        print(f"  Progress: {rendered+skipped}/{total} "
              f"({rendered} new, {skipped} skip, {errors} err) "
              f"| {rate:.1f} img/s")

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"  Done : {rendered} rendered, {skipped} skipped, {errors} errors")
    print(f"  Time : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Out  : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()