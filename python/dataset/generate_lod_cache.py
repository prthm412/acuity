"""
Generates LOD cache files for all meshes using Open3D's robust mesh
simplification. Saves 5 LOD levels per mesh as OBJ files to 'data/processes/lod_cache'
in the exact format expected by the C++ BatchRenderer cache loader.

LOD levels:
0: 100%
1: 50%
2: 25%
3: 12.5%
4: 6.25%

For meshes over 100k triangles, adaptive ratios are used:
3: 15%
4: 10%
"""

import csv
import sys
from pathlib import Path

import open3d as o3d
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "data" / "processed" / "lod_cache"
MANIFEST = ROOT / "assets" / "models" / "mesh_manifest.csv"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Adaptive LOD ratios based on mesh triangle count
# under 50k triangles
LOD_RATIOS_SMALL = [1.0, 0.5, 0.25, 0.125, 0.0625]

# 50k-100k triangles
LOD_RATIOS_MEDIUM = [1.0, 0.5, 0.25, 0.125, 0.08]

# over 100k triangles
LOD_RATIOS_LARGE = [1.0, 0.5, 0.25, 0.15, 0.10]

def preprocess_obj(path: Path) -> Path:
    """Remove non-mesh lines from OBJ that confuse Open3D.
    Returns path to cleaned temp file."""
    import tempfile
    lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Keep only vertex, face, normal, comment lines
        if stripped.startswith(('v ', 'vn ', 'vt ', 'f ', '#', '')):
            cleaned.append(line)
    tmp = Path(tempfile.mktemp(suffix='.obj'))
    tmp.write_text('\n'.join(cleaned))
    return tmp


def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    # Preprocess to remove non-triangle geometry
    clean_path = preprocess_obj(path)
    mesh = o3d.io.read_triangle_mesh(str(clean_path))
    clean_path.unlink()  # delete temp file

    if not mesh.has_vertices():
        return None
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if len(np.asarray(mesh.triangles)) == 0:
        return None
    return mesh

def save_obj(mesh: o3d.geometry.TriangleMesh, path: Path):
    # Compute vertex normals so the C++ renderer can do proper lighting
    mesh.compute_vertex_normals()

    verts   = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    tris    = np.asarray(mesh.triangles)

    with open(path, "w") as f:
        f.write(f"# Acuity LOD cache\n")
        f.write(f"# Triangles: {len(tris)}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for t in tris:
            # OBJ face format with normals: f v1//vn1 v2//vn2 v3//vn3
            f.write(f"f {t[0]+1}//{t[0]+1} "
                    f"{t[1]+1}//{t[1]+1} "
                    f"{t[2]+1}//{t[2]+1}\n")

def generate_lods(mesh_name: str, mesh_path: Path) -> bool:
    print(f"\n[{mesh_name}] Loading...")
    mesh = load_mesh(mesh_path)
    if mesh is None:
        print(f"[{mesh_name}] Failed to load")
        return False

    tri_count = len(np.asarray(mesh.triangles))
    print(f"[{mesh_name}] {len(np.asarray(mesh.vertices))} vertices, "
          f"{tri_count} triangles")
    
    # Skip meshes with too few triangles for meaningful LOD generation
    if tri_count < 500:
        print(f"[{mesh_name}] Too few triangles ({tri_count}) — skipping")
        return False

    # Choose ratios based on mesh size
    if tri_count > 100000:
        ratios = LOD_RATIOS_LARGE
    elif tri_count > 50000:
        ratios = LOD_RATIOS_MEDIUM
    else:
        ratios = LOD_RATIOS_SMALL

    for lod_idx, ratio in enumerate(ratios):
        cache_path = CACHE_DIR / f"{mesh_name}_lod{lod_idx}.obj"

        # Skip if already cached
        if cache_path.exists():
            print(f"[{mesh_name}] LOD {lod_idx} already cached — skipping")
            continue

        if ratio == 1.0:
            # LOD 0 — save original mesh directly
            lod_mesh = mesh
        else:
            target_tris = max(4, int(tri_count * ratio))
            print(f"[{mesh_name}] Generating LOD {lod_idx} "
                  f"({int(ratio*100)}%) → {target_tris} triangles...")
            lod_mesh = mesh.simplify_quadric_decimation(target_tris)

            # Validate — check for explosion
            verts = np.asarray(lod_mesh.vertices)
            orig_verts = np.asarray(mesh.vertices)
            orig_max = np.max(np.linalg.norm(orig_verts, axis=1))
            lod_max  = np.max(np.linalg.norm(verts, axis=1))

            if lod_max > orig_max * 3.0:
                print(f"[{mesh_name}] LOD {lod_idx} exploded "
                      f"(max dist {lod_max:.2f} vs {orig_max:.2f}) "
                      f"— using previous LOD")
                prev_path = CACHE_DIR / f"{mesh_name}_lod{lod_idx-1}.obj"
                import shutil
                shutil.copy(prev_path, cache_path)
                continue

            actual_tris = len(np.asarray(lod_mesh.triangles))
            print(f"[{mesh_name}] LOD {lod_idx} done: {actual_tris} triangles")

        save_obj(lod_mesh, cache_path)
        print(f"[{mesh_name}] LOD {lod_idx} saved: {cache_path.name}")

    return True


def load_manifest() -> list:
    if not MANIFEST.exists():
        print(f"Manifest not found: {MANIFEST}")
        sys.exit(1)
    meshes = []
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            meshes.append({
                "name": row["name"],
                "path": ROOT / row["path"]
            })
    return meshes


if __name__ == "__main__":
    print("=" * 60)
    print("  Acuity — LOD Cache Generator (Open3D)")
    print("=" * 60)

    meshes = load_manifest()

    # Filter to only meshes in our current config
    # Edit this list to match your render_configs.json meshes
    ACTIVE_MESHES = [
        "stanford-bunny", "armadillo", "happy", "horse", "max-planck",
        "spot", "cheburashka", "cow", "fandisk", "teapot",
        "woody", "homer", "igea", "Krzeslo_0_LR"
    ]

    meshes = [m for m in meshes if m["name"] in ACTIVE_MESHES]
    print(f"\nProcessing {len(meshes)} meshes...\n")

    success = 0
    failed  = []

    for m in meshes:
        ok = generate_lods(m["name"], m["path"])
        if ok:
            success += 1
        else:
            failed.append(m["name"])

    print("\n" + "=" * 60)
    print(f"  Done: {success}/{len(meshes)} meshes cached")
    if failed:
        print(f"  Failed: {failed}")
    print(f"  Cache: {CACHE_DIR}")
    print("=" * 60)