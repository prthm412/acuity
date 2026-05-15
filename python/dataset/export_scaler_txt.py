# Exports feature_scaler.pkl to a plain text file readable by C++
import pickle
import numpy as np
from pathlib import Path

scaler_path = Path("data/processed/feature_scaler.pkl")
output_path = Path("data/processed/feature_scaler.txt")

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

with open(output_path, "w") as f:
    f.write("mean: " + " ".join(f"{v:.8f}" for v in scaler.mean_) + "\n")
    f.write("std: "  + " ".join(f"{v:.8f}" for v in scaler.scale_) + "\n")

print(f"Exported scaler to {output_path}")
print(f"Features: {len(scaler.mean_)}")