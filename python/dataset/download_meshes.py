"""
python/dataset/download_meshes.py

Downloads the 3D mesh assets used in Acuity Phase 2 data generation.

We need 15-20 diverse meshes covering:
- Organic shapes (Stanford bunny, dragon, armadillo, buddha)
- Complex scenes (Sponza, San Miguel)
- Simple primitives (for sanity checking pipeline)
- Mechanical/hard-surface (gear, teapot)

Sources:
- Stanford 3D Scanning Repository: http://graphics.stanford.edu/data/3Dscanrep/
- Morgan McGuire's Archive: https://casual-effects.com/data/
- Thingi10K subset (curated): https://ten-thousand-models.appspot.com/

All meshes saved to: assets/models/
"""

import os
import sys
import requests
import zipfile
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm

ROOT        = Path(__file__).resolve().parent.parent.parent
MODELS_DIR  = ROOT / "assets" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _download(url: str, dest: Path, label: str = "") -> bool:
    """Download with progress bar."""
    try:
        r = requests.get(url, stream=True, timeout=60,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=label or dest.name, ncols=80
        ) as bar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [ERROR] {label}: {e}")
        return False


def _extract_zip(zip_path: Path, out_dir: Path):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    zip_path.unlink()


def _extract_gz(gz_path: Path, out_path: Path):
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()


# ── Mesh manifest ──────────────────────────────────────────────────────────
# Each entry: (label, url, output_filename, post_process)
# post_process: None | "unzip" | "ungz"
MESHES = [
    # Stanford Repository — classic research benchmarks
    (
        "Stanford Bunny",
        "https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz",
        "bunny.tar.gz",
        "tar"
    ),
    (
        "Utah Teapot (OBJ)",
        "https://graphics.stanford.edu/courses/cs148-10-summer/as3/code/as3/teapot.obj",
        "teapot.obj",
        None
    ),
    # McGuire Archive — rendering test scenes
    (
        "Crytek Sponza",
        "https://casual-effects.com/g3d/data10/research/model/CrytekSponza/sponza.zip",
        "sponza.zip",
        "unzip"
    ),
    (
        "Cornell Box",
        "https://casual-effects.com/g3d/data10/research/model/CornellBox/CornellBox-Original.zip",
        "cornellbox.zip",
        "unzip"
    ),
]

# ── Fallback: generate synthetic meshes programmatically ──────────────────
# Some Stanford URLs require registration. The script below generates
# synthetic OBJ meshes at multiple polygon counts as reliable fallbacks.
SYNTHETIC_MESHES = [
    ("sphere_hires",   5000),
    ("sphere_midres",  1000),
    ("sphere_lores",    200),
    ("torus_hires",    4000),
    ("torus_midres",    800),
    ("cube_subdivided", 512),
]


def generate_sphere_obj(output_path: Path, n_stacks: int = 40, n_slices: int = 40):
    """Generate a UV sphere OBJ file programmatically."""
    import math
    verts, faces = [], []
    for i in range(n_stacks + 1):
        phi = math.pi * i / n_stacks
        for j in range(n_slices + 1):
            theta = 2 * math.pi * j / n_slices
            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)
            verts.append((x, y, z))
    for i in range(n_stacks):
        for j in range(n_slices):
            a = i * (n_slices + 1) + j
            b = a + 1
            c = a + (n_slices + 1)
            d = c + 1
            faces.append((a + 1, c + 1, b + 1))
            faces.append((b + 1, c + 1, d + 1))
    with open(output_path, "w") as f:
        f.write(f"# Synthetic sphere — {len(faces)} triangles\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


def generate_torus_obj(output_path: Path, n_rings: int = 40, n_sides: int = 20,
                       R: float = 1.0, r: float = 0.3):
    """Generate a torus OBJ file programmatically."""
    import math
    verts, faces = [], []
    for i in range(n_rings):
        for j in range(n_sides):
            theta = 2 * math.pi * i / n_rings
            phi   = 2 * math.pi * j / n_sides
            x = (R + r * math.cos(phi)) * math.cos(theta)
            y = r * math.sin(phi)
            z = (R + r * math.cos(phi)) * math.sin(theta)
            verts.append((x, y, z))
    for i in range(n_rings):
        for j in range(n_sides):
            a = i * n_sides + j
            b = i * n_sides + (j + 1) % n_sides
            c = (i + 1) % n_rings * n_sides + j
            d = (i + 1) % n_rings * n_sides + (j + 1) % n_sides
            faces.append((a + 1, c + 1, b + 1))
            faces.append((b + 1, c + 1, d + 1))
    with open(output_path, "w") as f:
        f.write(f"# Synthetic torus — {len(faces)} triangles\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


def generate_subdivided_cube_obj(output_path: Path, subdivisions: int = 3):
    """Generate a subdivided cube OBJ file programmatically."""
    import math
    # Start from a unit cube and subdivide each face
    # Simple mid-point subdivision approach
    n = subdivisions
    verts = []
    faces = []

    def add_face_grid(p00, p10, p01, p11):
        """Add a grid of triangles for one cube face."""
        base = len(verts)
        for j in range(n + 1):
            for i in range(n + 1):
                t, s = i / n, j / n
                p = (
                    (1-t)*(1-s)*p00[0] + t*(1-s)*p10[0] + (1-t)*s*p01[0] + t*s*p11[0],
                    (1-t)*(1-s)*p00[1] + t*(1-s)*p10[1] + (1-t)*s*p01[1] + t*s*p11[1],
                    (1-t)*(1-s)*p00[2] + t*(1-s)*p10[2] + (1-t)*s*p01[2] + t*s*p11[2],
                )
                verts.append(p)
        for j in range(n):
            for i in range(n):
                a = base + j*(n+1) + i
                b = a + 1
                c = a + (n+1)
                d = c + 1
                faces.append((a+1, c+1, b+1))
                faces.append((b+1, c+1, d+1))

    # 6 faces of a unit cube
    add_face_grid((-1,-1, 1),( 1,-1, 1),(-1, 1, 1),( 1, 1, 1))
    add_face_grid(( 1,-1,-1),(-1,-1,-1),( 1, 1,-1),(-1, 1,-1))
    add_face_grid((-1,-1,-1),( 1,-1,-1),(-1,-1, 1),( 1,-1, 1))
    add_face_grid((-1, 1, 1),( 1, 1, 1),(-1, 1,-1),( 1, 1,-1))
    add_face_grid((-1,-1,-1),(-1,-1, 1),(-1, 1,-1),(-1, 1, 1))
    add_face_grid(( 1,-1, 1),( 1,-1,-1),( 1, 1, 1),( 1, 1,-1))

    with open(output_path, "w") as f:
        f.write(f"# Synthetic subdivided cube — {len(faces)} triangles\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


def generate_synthetic_meshes():
    """Generate all synthetic fallback meshes."""
    import math
    print("\n── Synthetic Mesh Generation ─────────────────────────────────")
    for name, target_faces in SYNTHETIC_MESHES:
        out = MODELS_DIR / f"{name}.obj"
        if out.exists():
            print(f"  [SKIP] {name}.obj already exists")
            continue
        stacks = int(math.sqrt(target_faces / 2))
        if "sphere" in name:
            generate_sphere_obj(out, n_stacks=stacks, n_slices=stacks)
        elif "torus" in name:
            rings = max(10, stacks)
            generate_torus_obj(out, n_rings=rings, n_sides=max(8, stacks//2))
        elif "cube" in name:
            generate_subdivided_cube_obj(out, subdivisions=6)
        size = out.stat().st_size / 1024
        print(f"  Generated {name}.obj ({size:.1f} KB)")


def download_meshes():
    """Attempt to download external meshes, skip on failure."""
    print("\n── External Mesh Downloads ───────────────────────────────────")
    print("  Note: Some Stanford URLs require manual download.")
    print("  Synthetic meshes will be used as reliable fallbacks.\n")

    for label, url, filename, post in MESHES:
        dest = MODELS_DIR / filename
        final = MODELS_DIR / filename.replace(".zip", "").replace(".tar.gz", "").replace(".gz", "")
        if dest.exists() or (final.exists() and final.is_dir()):
            print(f"  [SKIP] {label} already present")
            continue
        print(f"  Attempting: {label}")
        ok = _download(url, dest, label)
        if ok and post == "unzip":
            try:
                _extract_zip(dest, MODELS_DIR)
                print(f"  Extracted {label}")
            except Exception as e:
                print(f"  [WARN] Extract failed: {e}")
        elif ok and post == "tar":
            try:
                import tarfile
                with tarfile.open(dest) as t:
                    t.extractall(MODELS_DIR)
                dest.unlink()
                print(f"  Extracted {label}")
            except Exception as e:
                print(f"  [WARN] tar extract failed: {e}")


def write_mesh_manifest():
    """Write a manifest CSV of all available meshes."""
    import csv
    manifest_path = MODELS_DIR / "mesh_manifest.csv"
    obj_files = sorted(MODELS_DIR.rglob("*.obj")) + sorted(MODELS_DIR.rglob("*.ply"))

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "path", "size_kb", "extension"])
        for p in obj_files:
            if p.name == "mesh_manifest.csv":
                continue
            writer.writerow([p.stem, str(p.relative_to(ROOT)),
                             f"{p.stat().st_size/1024:.1f}", p.suffix])

    print(f"\n  Manifest written: {manifest_path.relative_to(ROOT)}")
    print(f"  Total meshes indexed: {len(obj_files)}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Acuity — Mesh Download Script")
    print("  Phase 2, Step 2.1")
    print("=" * 60)

    download_meshes()
    generate_synthetic_meshes()
    write_mesh_manifest()

    print("\nDone. Check assets/models/ for available meshes.")