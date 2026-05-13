"""
python/dataset/annotate_quality.py

Annotates each rendered image with a perceptual quality score using
LPIPS (Learned Perceptual Image Patch Similarity, Zhang et al., CVPR 2018).

Method:
  For each mesh at each viewpoint, LOD 0 is the full-quality reference.
  LPIPS measures perceptual similarity between each LOD render and the
  LOD 0 reference from the identical camera pose.

  quality_score = 1.0 - LPIPS(lod_image, lod0_reference)

  Score of 1.0 = perceptually identical to reference (LOD 0 itself)
  Score < 1.0  = perceptual quality loss relative to reference

LPIPS uses deep features (VGG) that correlate with human perceptual
judgment — it is not a hand-crafted formula. It is Paper #20 in the
project bibliography:
  "The Unreasonable Effectiveness of Deep Features as a Perceptual
   Metric", Zhang et al., CVPR 2018.

Input:  data/processed/rendered_images/*.png  (2800 images)
Output: data/processed/dataset_raw.csv
"""

import torch
import lpips
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image as PILImage
import re
import sys
import time

# ── Configuration ─────────────────────────────────────────────────────────────

IMAGES_DIR = Path("data/processed/rendered_images")
OUTPUT_CSV = Path("data/processed/dataset_raw.csv")

DIST_MULTIPLIERS = {
    "very_close": 1.2,
    "close":      2.0,
    "medium":     3.5,
    "far":        6.0,
    "very_far":   9.0,
}

LOD_LEVELS     = [0, 1, 2, 3, 4]
AZIMUTHS       = [0, 45, 90, 135, 180, 225, 270, 315]
DIST_LABELS    = ["very_close", "close", "medium", "far", "very_far"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_filename(stem: str) -> dict:
    pattern = r"^(.+)_lod(\d+)_az(\d+)_(very_close|very_far|close|medium|far)$"
    m = re.match(pattern, stem)
    if not m:
        raise ValueError(f"Cannot parse filename: {stem}")
    return {
        "mesh_name":           m.group(1),
        "lod_level":           int(m.group(2)),
        "azimuth":             int(m.group(3)),
        "distance_label":      m.group(4),
        "distance_multiplier": DIST_MULTIPLIERS[m.group(4)],
    }


def load_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """
    Load image as LPIPS-compatible tensor.
    LPIPS expects: float32, shape (1,3,H,W), values in [-1, 1].
    """
    img = PILImage.open(path).convert("RGB").resize((256, 256))
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0   # [0,255] → [-1,1]
    t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Discover all mesh names from rendered images
    all_paths = sorted(IMAGES_DIR.glob("*.png"))
    if not all_paths:
        print(f"[ERROR] No images found in {IMAGES_DIR}")
        sys.exit(1)

    mesh_names = sorted(set(parse_filename(p.stem)["mesh_name"]
                            for p in all_paths))

    total = len(mesh_names) * len(LOD_LEVELS) * len(AZIMUTHS) * len(DIST_LABELS)

    print("=" * 60)
    print("  Acuity — Quality Annotator (LPIPS)")
    print("=" * 60)
    print(f"  Meshes  : {len(mesh_names)}")
    print(f"  Images  : {total}")
    print(f"  Metric  : LPIPS (VGG, Zhang et al. CVPR 2018)")
    print(f"  Device  : {device}")
    print(f"  Output  : {OUTPUT_CSV}")
    print("=" * 60)

    # Load LPIPS model
    print("\n[1/3] Loading LPIPS model (VGG)...")
    loss_fn = lpips.LPIPS(net="vgg").to(device)
    loss_fn.eval()
    print("      LPIPS ready.")

    # Score all images
    print(f"\n[2/3] Scoring {total} images...")
    t_start = time.time()
    records = []
    done    = 0

    for mesh_name in mesh_names:
        # Pre-load all LOD 0 reference tensors for this mesh
        lod0_refs = {}
        for az in AZIMUTHS:
            for dist_label in DIST_LABELS:
                ref_name = f"{mesh_name}_lod0_az{az:03d}_{dist_label}.png"
                ref_path = IMAGES_DIR / ref_name
                if ref_path.exists():
                    lod0_refs[(az, dist_label)] = load_tensor(ref_path, device)

        for lod_idx in LOD_LEVELS:
            for az in AZIMUTHS:
                for dist_label in DIST_LABELS:
                    img_name = (f"{mesh_name}_lod{lod_idx}"
                                f"_az{az:03d}_{dist_label}.png")
                    img_path = IMAGES_DIR / img_name

                    if not img_path.exists():
                        continue

                    if lod_idx == 0:
                        # LOD 0 is the reference — perfect score by definition
                        quality_score = 1.0
                    else:
                        ref_key = (az, dist_label)
                        if ref_key not in lod0_refs:
                            continue

                        img_tensor = load_tensor(img_path, device)
                        ref_tensor = lod0_refs[ref_key]

                        with torch.no_grad():
                            lpips_dist = loss_fn(img_tensor, ref_tensor)

                        # Convert distance to similarity score
                        # LPIPS distance 0.0 = identical, higher = more different
                        # We clip at 1.0 so score stays in [0, 1]
                        quality_score = float(
                            torch.clamp(1.0 - lpips_dist, 0.0, 1.0).item()
                        )

                    records.append({
                        "image_path":          str(img_path),
                        "mesh_name":           mesh_name,
                        "lod_level":           lod_idx,
                        "azimuth":             az,
                        "distance_label":      dist_label,
                        "distance_multiplier": DIST_MULTIPLIERS[dist_label],
                        "quality_score":       quality_score,
                    })

                    done += 1
                    if done % 100 == 0:
                        elapsed = time.time() - t_start
                        rate    = done / elapsed
                        print(f"  {done:>4}/{total}  ({rate:.1f} img/s)", end="\r")

    print(f"\n  Scoring complete in {time.time()-t_start:.1f}s")

    # Save CSV
    print("\n[3/3] Saving CSV...")
    df = pd.DataFrame(records)
    df = df[[
        "image_path", "mesh_name", "lod_level",
        "azimuth", "distance_label", "distance_multiplier",
        "quality_score"
    ]]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n  Saved: {OUTPUT_CSV}  ({len(df)} rows)")
    print("\n  Quality score statistics:")
    print(f"    Min  : {df.quality_score.min():.4f}")
    print(f"    Max  : {df.quality_score.max():.4f}")
    print(f"    Mean : {df.quality_score.mean():.4f}")
    print(f"    Std  : {df.quality_score.std():.4f}")
    print("\n  Mean score per LOD level:")
    print(df.groupby("lod_level")["quality_score"].mean().to_string())
    print("\n  Mean score per distance:")
    print(df.groupby("distance_label")["quality_score"].mean()
            .reindex(["very_close", "close", "medium", "far", "very_far"])
            .to_string())
    print("\n" + "=" * 60)
    print("  Step 2.3 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()