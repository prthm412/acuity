import torch
import numpy as np
import pandas as pd
import h5py
import json
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, no display needed
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

from model import LODPerceptionNet

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
HDF5_PATH = ROOT_DIR / "data/processed/dataset.h5"
TEST_CSV     = ROOT_DIR / "data/processed/test.csv"
MODELS_DIR   = ROOT_DIR / "data/models"
RESULTS_DIR  = ROOT_DIR / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


#  Load model
def load_model(checkpoint_path: Path, device: torch.device) -> LODPerceptionNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config     = checkpoint["config"]
    model      = LODPerceptionNet(config["input_dim"], config["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Checkpoint val SRCC : {checkpoint['val_srcc']:.4f}")
    return model


# Inference
def predict(model: LODPerceptionNet, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(HDF5_PATH, "r") as f:
        X = torch.tensor(f["test/X"][:], dtype=torch.float32).to(device)
        y = f["test/y"][:]

    with torch.no_grad():
        preds = model(X).cpu().squeeze(1).numpy()

    return preds, y


# Metrics
def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    srcc, _  = spearmanr(preds, targets)
    plcc, _  = pearsonr(preds, targets)
    mse      = float(np.mean((preds - targets) ** 2))
    mae      = float(np.mean(np.abs(preds - targets)))
    ss_res   = np.sum((targets - preds) ** 2)
    ss_tot   = np.sum((targets - targets.mean()) ** 2)
    r2       = float(1 - ss_res / ss_tot)
    return {
        "SRCC": float(srcc),
        "PLCC": float(plcc),
        "MSE":  mse,
        "MAE":  mae,
        "R2":   r2,
    }


def geometric_baseline_metrics(targets: np.ndarray) -> dict:
    """
    Geometric baseline: predict quality purely from LOD level.
    LOD 0 = 1.0, LOD 1 = 0.8, LOD 2 = 0.6, LOD 3 = 0.4, LOD 4 = 0.2
    This is what a distance-only LOD selector implicitly assumes.
    """
    test_df   = pd.read_csv(TEST_CSV)
    lod_map   = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2}
    baseline  = test_df["lod_level"].map(lod_map).values
    return compute_metrics(baseline, targets)


# Plots
def plot_predictions(preds, targets):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, preds, alpha=0.4, s=10, color="#534AB7")
    mn, mx = min(targets.min(), preds.min()), max(targets.max(), preds.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Ground truth quality score")
    ax.set_ylabel("Predicted quality score")
    ax.set_title("Predictions vs Ground Truth (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "predictions_vs_truth.png", dpi=150)
    plt.close(fig)
    print("  Saved: predictions_vs_truth.png")


def plot_error_distribution(preds, targets):
    errors = preds - targets
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=40, color="#1D9E75", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="red", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Prediction error (pred − truth)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution (test set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved: error_distribution.png")


def plot_per_lod(preds, targets):
    test_df  = pd.read_csv(TEST_CSV)
    lod_vals = sorted(test_df["lod_level"].unique())
    srcc_per = []
    for lod in lod_vals:
        mask        = test_df["lod_level"].values == lod
        s, _        = spearmanr(preds[mask], targets[mask])
        srcc_per.append(s)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"LOD {l}" for l in lod_vals], srcc_per, color="#378ADD")
    ax.axhline(0.75, color="red", linestyle="--", linewidth=1.2, label="Target SRCC=0.75")
    ax.set_ylabel("SRCC")
    ax.set_title("SRCC per LOD level (test set)")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "srcc_per_lod.png", dpi=150)
    plt.close(fig)
    print("  Saved: srcc_per_lod.png")


def plot_residuals(preds, targets):
    residuals = preds - targets
    fig, ax   = plt.subplots(figsize=(7, 4))
    ax.scatter(preds, residuals, alpha=0.4, s=10, color="#D85A30")
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Predicted quality score")
    ax.set_ylabel("Residual (pred − truth)")
    ax.set_title("Residual Plot (test set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "residuals.png", dpi=150)
    plt.close(fig)
    print("  Saved: residuals.png")


# Report
def write_report(metrics: dict, baseline: dict, preds: np.ndarray, targets: np.ndarray):
    lines = [
        "=" * 55,
        "  Acuity — Test Set Evaluation Report",
        "=" * 55,
        "",
        "Model: LODPerceptionNet (best_model.pth)",
        f"Test samples: {len(preds)}",
        "",
        "--- Perceptual Model Metrics ---",
        f"  SRCC : {metrics['SRCC']:.4f}   (target: > 0.75)",
        f"  PLCC : {metrics['PLCC']:.4f}",
        f"  MSE  : {metrics['MSE']:.6f}",
        f"  MAE  : {metrics['MAE']:.6f}",
        f"  R²   : {metrics['R2']:.4f}",
        "",
        "--- Geometric Baseline Metrics ---",
        f"  SRCC : {baseline['SRCC']:.4f}",
        f"  PLCC : {baseline['PLCC']:.4f}",
        f"  MSE  : {baseline['MSE']:.6f}",
        f"  MAE  : {baseline['MAE']:.6f}",
        f"  R²   : {baseline['R2']:.4f}",
        "",
        "--- Improvement over Baseline ---",
        f"  SRCC delta : {metrics['SRCC'] - baseline['SRCC']:+.4f}",
        f"  PLCC delta : {metrics['PLCC'] - baseline['PLCC']:+.4f}",
        f"  MSE delta  : {metrics['MSE']  - baseline['MSE']:+.6f}",
        "",
        "--- Score Range ---",
        f"  Prediction range : [{preds.min():.4f}, {preds.max():.4f}]",
        f"  Target range     : [{targets.min():.4f}, {targets.max():.4f}]",
        "",
        "Figures saved to results/figures/",
        "=" * 55,
    ]
    report = "\n".join(lines)
    print(report)
    with open(RESULTS_DIR / "test_metrics.txt", "w") as f:
        f.write(report)
    print(f"\n  Saved: results/test_metrics.txt")


# MAIN
def main():
    print("=" * 55)
    print("  Acuity — Model Evaluation")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Load model
    print("\n[1/4] Loading model...")
    model = load_model(MODELS_DIR / "best_model.pth", device)

    # Run inference on test set
    print("\n[2/4] Running inference on test set...")
    preds, targets = predict(model, device)
    print(f"  Predictions: {preds.shape}, range [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"  Targets    : {targets.shape}, range [{targets.min():.4f}, {targets.max():.4f}]")

    # Save predictions CSV
    test_df = pd.read_csv(TEST_CSV)
    test_df["predicted_score"] = preds
    test_df["ground_truth"]    = targets
    test_df["error"]           = preds - targets
    out_csv = RESULTS_DIR / "test_predictions.csv"
    test_df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    # Compute metrics
    print("\n[3/4] Computing metrics...")
    metrics  = compute_metrics(preds, targets)
    baseline = geometric_baseline_metrics(targets)

    # Generate plots
    print("\n[4/4] Generating figures...")
    plot_predictions(preds, targets)
    plot_error_distribution(preds, targets)
    plot_per_lod(preds, targets)
    plot_residuals(preds, targets)

    # Write report
    print()
    write_report(metrics, baseline, preds, targets)


if __name__ == "__main__":
    main()