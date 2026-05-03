"""
python/dataset/download_datasets.py

Downloads and organizes the image quality assessment datasets used in Acuity.
Phase 2, Step 2.1 — Tasks 1, 2, 3.

Datasets:
- Q-Bench (ICLR 2024 Spotlight): annotation JSONs with human quality ratings.
  We use the pre-trained Q-Bench model in Step 2.3 to score our rendered mesh
  images. These JSONs tell us the expected score distribution.
  Source: https://huggingface.co/datasets/q-future/q-bench
  Paper: Wu et al., ICLR 2024 (Paper #18 in research_papers_list.md)

- KADID-10k: distortion mean opinion scores (CSV metadata only).
  Secondary cross-validation reference for score range and distribution.
  Source: http://database.mmsp-kn.de/kadid-10k-database.html

Requires: HF_TOKEN environment variable set with a Hugging Face read token.
          Get one free at https://huggingface.co/settings/tokens
"""

import os
import json
import requests
from pathlib import Path
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent.parent
DATA_RAW   = ROOT / "data" / "raw"
QBENCH_DIR = DATA_RAW / "qbench"
KADID_DIR  = DATA_RAW / "kadid"

# ── Q-Bench: direct JSON file URLs on Hugging Face ─────────────────────────
# These are the lightweight annotation files only (~1-2 MB each).
# We do NOT download the full parquet image dataset (~393MB+).
# The Q-Bench model weights (for Step 2.3) are downloaded separately.
QBENCH_FILES = [
    {
        "name" : "llvisionqa_qbench_dev.json",
        "url"  : "https://huggingface.co/datasets/teowu/LLVisionQA-QBench/resolve/main/llvisionqa_dev.json",
        "desc" : "Dev split — image filenames and human quality ratings"
    },
    {
        "name" : "llvisionqa_qbench_test.json",
        "url"  : "https://huggingface.co/datasets/teowu/LLVisionQA-QBench/resolve/main/llvisionqa_test.json",
        "desc" : "Test split — image filenames and human quality ratings"
    },
]

# ── KADID-10k: difference mean opinion scores CSV ──────────────────────────
KADID_FILES = [
    {
        "name" : "dmos.csv",
        "url"  : "https://datasets.activeloop.ai/wp-content/uploads/2022/09/dmos.csv",
        "desc" : "DMOS scores for all 10,125 distorted images"
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_headers() -> dict:
    """Build authorization headers from HF_TOKEN env variable."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("  [WARN] HF_TOKEN not set. Q-Bench may return 401.")
        print("         Set it with: export HF_TOKEN=hf_your_token_here")
        return {}
    return {"Authorization": f"Bearer {token}"}


def _download(url: str, dest: Path, label: str, headers: dict = None) -> bool:
    """Download a single file with a progress bar. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, headers=headers or {}, stream=True, timeout=60)
        if r.status_code == 401:
            print(f"  [ERROR] 401 Unauthorized — check your HF_TOKEN")
            return False
        if r.status_code == 404:
            print(f"  [ERROR] 404 Not Found — {url}")
            return False
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=f"  {label}", ncols=72
        ) as bar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [ERROR] {label}: {e}")
        return False


def _is_valid_json(path: Path) -> bool:
    """Check if a file exists and contains valid non-placeholder JSON."""
    if not path.exists():
        return False
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        # Reject placeholder files written by previous failed attempts
        if isinstance(content, dict) and "placeholder" in content.get("note", ""):
            return False
        return True
    except Exception:
        return False


# ── Q-Bench download ───────────────────────────────────────────────────────

def download_qbench():
    """Download Q-Bench annotation JSON files."""
    print("\n── Q-Bench Annotations ───────────────────────────────────────")
    QBENCH_DIR.mkdir(parents=True, exist_ok=True)
    headers = _get_headers()
    success = 0

    for item in QBENCH_FILES:
        dest = QBENCH_DIR / item["name"]

        if _is_valid_json(dest):
            count = len(json.loads(dest.read_text()))
            print(f"  [SKIP] {item['name']} already present ({count} items)")
            success += 1
            continue

        # Remove invalid/placeholder file if it exists
        if dest.exists():
            dest.unlink()

        ok = _download(item["url"], dest, item["name"], headers)
        if ok and _is_valid_json(dest):
            count = len(json.loads(dest.read_text()))
            print(f"  OK: {item['name']} — {count} items, "
                  f"{dest.stat().st_size / 1024:.1f} KB")
            success += 1
        elif ok:
            print(f"  [WARN] Downloaded but JSON validation failed: {item['name']}")
        else:
            print(f"  [FAIL] {item['name']} — see error above")

    # Write README
    readme = QBENCH_DIR / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Q-Bench Dataset\n\n"
            "Paper: Wu et al., ICLR 2024 (Spotlight)\n"
            "URL: https://github.com/Q-Future/Q-Bench\n\n"
            "## Contents\n"
            "- `llvisionqa_qbench_dev.json` — dev split annotations\n"
            "- `llvisionqa_qbench_test.json` — test split annotations\n\n"
            "## Usage in Acuity\n"
            "Step 2.3 uses the pre-trained Q-Bench model to score rendered mesh "
            "images. These JSONs document the score distribution for validation.\n\n"
            "## Full image dataset\n"
            "Full images (~50GB) available at: "
            "https://huggingface.co/datasets/q-future/q-bench\n"
        )

    return success == len(QBENCH_FILES)


# ── KADID download ─────────────────────────────────────────────────────────

def download_kadid():
    """Download KADID-10k metadata CSV."""
    print("\n── KADID-10k Metadata ────────────────────────────────────────")
    KADID_DIR.mkdir(parents=True, exist_ok=True)
    success = 0

    for item in KADID_FILES:
        dest = KADID_DIR / item["name"]
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"  [SKIP] {item['name']} already present "
                  f"({dest.stat().st_size / 1024:.1f} KB)")
            success += 1
            continue

        ok = _download(item["url"], dest, item["name"])
        if ok:
            print(f"  OK: {item['name']} — {dest.stat().st_size / 1024:.1f} KB")
            success += 1
        else:
            print(f"  [FAIL] {item['name']}")

    # Write README
    readme = KADID_DIR / "README.md"
    if not readme.exists():
        readme.write_text(
            "# KADID-10k Dataset\n\n"
            "Source: http://database.mmsp-kn.de/kadid-10k-database.html\n\n"
            "## Contents\n"
            "- `dmos.csv` — difference mean opinion scores for 10,125 images\n\n"
            "## Usage in Acuity\n"
            "Cross-validation reference. Validates that Q-Bench scores in Step 2.3 "
            "span a meaningful 0-1 range across distortion levels.\n\n"
            "## Full dataset\n"
            "Full images (~2GB) available at the source URL above.\n"
        )

    return success == len(KADID_FILES)


# ── Summary ────────────────────────────────────────────────────────────────

def print_summary(qbench_ok: bool, kadid_ok: bool):
    print("\n── Summary ───────────────────────────────────────────────────")
    for d, name, ok in [
        (QBENCH_DIR, "Q-Bench",   qbench_ok),
        (KADID_DIR,  "KADID-10k", kadid_ok),
    ]:
        status = "OK" if ok else "INCOMPLETE"
        if d.exists():
            files = [f for f in d.iterdir() if f.is_file()]
            total = sum(f.stat().st_size for f in files)
            print(f"  [{status}] {name}: {len(files)} files, "
                  f"{total / 1024:.1f} KB")
        else:
            print(f"  [MISSING] {name}")

    if qbench_ok and kadid_ok:
        print("\n  All datasets downloaded successfully.")
        print("  Next: run python/dataset/download_meshes.py")
    else:
        print("\n  Some downloads failed. Check errors above.")
        if not qbench_ok:
            print("  Q-Bench fix: make sure HF_TOKEN is set correctly.")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Acuity — Dataset Download")
    print("=" * 60)

    qbench_ok = download_qbench()
    kadid_ok  = download_kadid()
    print_summary(qbench_ok, kadid_ok)