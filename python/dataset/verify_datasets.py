"""
python/dataset/verify_datasets.py

Verifies that all datasets and meshes for Phase 2 are present and valid.
Prints a pass/fail report. Run this before starting Step 2.2.
"""

import json
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent.parent
DATA_RAW   = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "assets" / "models"

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"

errors = 0

def check(condition, pass_msg, fail_msg):
    global errors
    if condition:
        print(f"{PASS} {pass_msg}")
    else:
        print(f"{FAIL} {fail_msg}")
        errors += 1

def warn(condition, warn_msg):
    if not condition:
        print(f"{WARN} {warn_msg}")

# ── Q-Bench ────────────────────────────────────────────────────────────────
print("\n── Q-Bench ───────────────────────────────────────────────────")
for name in ["llvisionqa_qbench_dev.json", "llvisionqa_qbench_test.json"]:
    path = DATA_RAW / "qbench" / name
    if path.exists():
        try:
            data = json.loads(path.read_text())
            valid = isinstance(data, list) and len(data) > 100
            check(valid, f"{name} — {len(data)} items", f"{name} — invalid or too small")
        except Exception as e:
            check(False, "", f"{name} — JSON parse error: {e}")
    else:
        check(False, "", f"{name} — file not found")

# ── KADID ──────────────────────────────────────────────────────────────────
print("\n── KADID-10k ─────────────────────────────────────────────────")
kadid_csv = DATA_RAW / "kadid" / "dmos.csv"
check(kadid_csv.exists() and kadid_csv.stat().st_size > 1000,
      f"dmos.csv — {kadid_csv.stat().st_size/1024:.1f} KB" if kadid_csv.exists() else "",
      "dmos.csv — not found or too small")

# ── Meshes ─────────────────────────────────────────────────────────────────
print("\n── Mesh Assets ───────────────────────────────────────────────")
expected = {
    "stanford": [
        "stanford-bunny.obj", "armadillo.obj", "lucy.obj", "happy.obj",
        "xyzrgb_dragon.obj", "max-planck.obj", "horse.obj", "cow.obj",
        "fandisk.obj", "spot.obj", "cheburashka.obj", "suzanne.obj",
        "teapot.obj", "woody.obj"
    ],
    "mcguire": [
        "sponza.obj", "rungholt.obj", "CornellBox-Original.obj"
    ],
    "sketchfab": [
        "AmericanRockSaltMinePinkHalite.obj",
        "PRI_TyrannosaurusRexSkull.obj",
        "Krzeslo_0_LR.obj"
    ]
}

total_found = 0
for folder, files in expected.items():
    print(f"\n  {folder}/")
    for fname in files:
        path = MODELS_DIR / folder / fname
        if path.exists():
            size_kb = path.stat().st_size / 1024
            warn(size_kb > 1, f"{fname} exists but is suspiciously small ({size_kb:.1f} KB)")
            print(f"{PASS} {fname} ({size_kb:.0f} KB)")
            total_found += 1
        else:
            print(f"{FAIL} {fname} — not found")
            errors += 1

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n── Summary ───────────────────────────────────────────────────")
print(f"  Meshes found: {total_found} / {sum(len(v) for v in expected.values())}")
print(f"  Errors: {errors}")
if errors == 0:
    print("\n  All checks passed. Ready for Step 2.2.")
else:
    print("\n  Fix errors above before proceeding to Step 2.2.")