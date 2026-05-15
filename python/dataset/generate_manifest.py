"""
python/dataset/generate_manifest.py

Scans assets/models/ and generates mesh_manifest.csv listing all
available OBJ files with their metadata.
"""

import csv
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "assets" / "models"
OUTPUT     = MODELS_DIR / "mesh_manifest.csv"


def scan_meshes() -> list:
    rows = []
    for obj in sorted(MODELS_DIR.rglob("*.obj")):
        source = obj.parent.name
        rows.append({
            "name"      : obj.stem,
            "path"      : str(obj.relative_to(ROOT)).replace("\\", "/"),
            "size_kb"   : round(obj.stat().st_size / 1024, 1),
            "source"    : source,
            "extension" : obj.suffix
        })
    return rows


if __name__ == "__main__":
    print("=" * 60)
    print("  Acuity — Mesh Manifest Generator")
    print("  Phase 2, Step 2.2")
    print("=" * 60)

    rows = scan_meshes()

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name","path","size_kb","source","extension"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Found {len(rows)} meshes:")
    for r in rows:
        print(f"  {r['source']:12s}  {r['name']:45s}  {r['size_kb']:>8.1f} KB")
    print(f"\n  Written: assets/models/mesh_manifest.csv")