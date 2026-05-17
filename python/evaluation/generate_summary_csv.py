"""
python/evaluation/generate_summary_csv.py

Generates results/benchmarks/summary.csv from individual JSON benchmark files.
"""

import json
import csv
from pathlib import Path

benchmark_dir = Path("results/benchmarks")
output_csv    = benchmark_dir / "summary.csv"

fieldnames = [
    "scene", "method", "repetition",
    "avg_fps", "min_fps", "max_fps",
    "avg_frame_ms", "p95_frame_ms",
    "avg_triangles", "min_triangles", "max_triangles",
    "gpu_memory_bytes", "avg_lod_sel_ms", "max_lod_sel_ms",
    "lod0_frames", "lod1_frames", "lod2_frames", "lod3_frames", "lod4_frames"
]

rows = []
for json_file in sorted(benchmark_dir.glob("*.json")):
    with open(json_file) as f:
        d = json.load(f)
    dist = d["lod_distribution"]
    rows.append({
        "scene":            d["scene"],
        "method":           d["method"],
        "repetition":       d["repetition"],
        "avg_fps":          d["avg_fps"],
        "min_fps":          d["min_fps"],
        "max_fps":          d["max_fps"],
        "avg_frame_ms":     d["avg_frame_ms"],
        "p95_frame_ms":     d["p95_frame_ms"],
        "avg_triangles":    d["avg_triangles"],
        "min_triangles":    d["min_triangles"],
        "max_triangles":    d["max_triangles"],
        "gpu_memory_bytes": d["gpu_memory_bytes"],
        "avg_lod_sel_ms":   d["avg_lod_sel_ms"],
        "max_lod_sel_ms":   d["max_lod_sel_ms"],
        "lod0_frames":      dist[0],
        "lod1_frames":      dist[1],
        "lod2_frames":      dist[2],
        "lod3_frames":      dist[3],
        "lod4_frames":      dist[4],
    })

with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Written {len(rows)} rows to {output_csv}")