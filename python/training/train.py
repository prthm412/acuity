"""
python/training/train.py

Trains LODPerceptionNet on the Phase 2 dataset.

Pipeline:
    1. Load train.csv and val.csv (normalized 38D features + quality scores)
    2. PyTorch Dataset wraps the HDF5 file for fast batch loading
    3. Training loop runs for up to 100 epochs with early stopping
    4. Weights & Biases logs all metrics for experiment tracking
    5. Best model (lowest val loss) is saved as data/models/best_model.pth

Key design decisions:
  - HDF5 loading: faster than CSV for repeated epoch access
  - Split by mesh (done in Phase 2): no data leakage — val meshes never
    appeared in training
  - Early stopping (patience=15): stops training if val loss does not
    improve for 15 consecutive epochs, prevents overfitting
  - LR scheduler (ReduceLROnPlateau): halves learning rate when val loss
    plateaus — allows fine-grained convergence after initial learning
  - Gradient clipping (max_norm=1.0): prevents exploding gradients,
    important with the combined ranking + correlation loss
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import h5py
import wandb
from pathlib import Path
import time
import json

from model import LODPerceptionNet, PerceptualLODLoss


# Configuration

CONFIG = {
    "input_dim":    38,
    "dropout":      0.3,
    "batch_size":   64,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "epochs":       100,
    "patience":     15,       # early stopping patience
    "lr_patience":  7,        # LR scheduler patience
    "lr_factor":    0.5,      # LR reduction factor
    "grad_clip":    1.0,
    "loss_alpha":   0.6,      # MSE weight
    "loss_beta":    0.2,      # Ranking weight
    "loss_gamma":   0.2,      # Correlation weight
    "seed":         42,
}

ROOT_DIR    = Path(__file__).resolve().parent.parent.parent
HDF5_PATH   = ROOT_DIR / "data/processed/dataset.h5"
MODELS_DIR  = ROOT_DIR / "data/models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Dataset

class LODDataset(Dataset):
    """
    PyTorch Dataset that reads features and quality scores from the HDF5 file.

    HDF5 layout (written by finalize_dataset.py):
        /train/features  : (2000, 38) float32
        /train/labels    : (2000,)    float32
        /val/features    : (400,  38) float32
        /val/labels      : (400,)     float32
        /test/features   : (400,  38) float32
        /test/labels     : (400,)     float32

    Loads the entire split into memory at init (dataset is small enough).
    This avoids repeated HDF5 reads during training, which would be slow.
    """

    def __init__(self, hdf5_path: Path, split: str):
        """
        Args:
            hdf5_path: path to dataset.h5
            split:     one of 'train', 'val', 'test'
        """
        with h5py.File(hdf5_path, "r") as f:
            self.features = torch.tensor(
                f[f"{split}/X"][:], dtype=torch.float32
            )
            self.labels = torch.tensor(
                f[f"{split}/y"][:], dtype=torch.float32
            ).unsqueeze(1)   # (N,) -> (N, 1) to match model output shape

        print(f"  [{split}] {len(self)} samples, "
              f"features: {self.features.shape}, "
              f"labels: {self.labels.shape}")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


# Metrics

def spearman_rcc(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Spearman Rank Correlation Coefficient (SRCC).
    Measures monotonic relationship between predictions and targets.
    Target metric: SRCC > 0.75.
    Range: [-1, 1], higher is better.
    """
    from scipy.stats import spearmanr
    corr, _ = spearmanr(pred, target)
    return float(corr)


def pearson_lcc(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Pearson Linear Correlation Coefficient (PLCC).
    Measures linear relationship. Companion metric to SRCC.
    """
    corr = np.corrcoef(pred, target)[0, 1]
    return float(corr)


# Training

def run_epoch(
    model:     LODPerceptionNet,
    loader:    DataLoader,
    loss_fn:   PerceptualLODLoss,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    training:  bool,
) -> tuple[float, dict]:
    """
    Run one full pass over the dataloader.
    If training=True, computes gradients and updates weights.
    If training=False (validation), runs in no_grad mode.

    Returns:
        mean_loss: average total loss over all batches
        metrics:   dict with mean MSE, rank, corr loss components
    """
    model.train(training)
    total_loss  = 0.0
    components  = {"mse": 0.0, "rank": 0.0, "corr": 0.0}
    all_preds   = []
    all_targets = []
    n_batches   = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for features, labels in loader:
            features = features.to(device)
            labels   = labels.to(device)

            preds = model(features)
            loss, comp = loss_fn(preds, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG["grad_clip"]
                )
                optimizer.step()

            total_loss += loss.item()
            for k in components:
                components[k] += comp[k]
            all_preds.append(preds.detach().cpu().squeeze(1).numpy())
            all_targets.append(labels.detach().cpu().squeeze(1).numpy())
            n_batches += 1

    mean_loss = total_loss / n_batches
    for k in components:
        components[k] /= n_batches

    preds_np   = np.concatenate(all_preds)
    targets_np = np.concatenate(all_targets)
    components["srcc"] = spearman_rcc(preds_np, targets_np)
    components["plcc"] = pearson_lcc(preds_np, targets_np)

    return mean_loss, components


def train(config: dict = CONFIG):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Data
    print("\n[1/5] Loading dataset...")
    train_ds = LODDataset(HDF5_PATH, "train")
    val_ds   = LODDataset(HDF5_PATH, "val")

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"],
        shuffle=True, num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")
    )

    # Model
    print("\n[2/5] Building model...")
    model   = LODPerceptionNet(config["input_dim"], config["dropout"]).to(device)
    loss_fn = PerceptualLODLoss(
        config["loss_alpha"], config["loss_beta"], config["loss_gamma"]
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["lr_factor"],
        patience=config["lr_patience"],
    )
    print(f"  Parameters: {model.count_parameters():,}")

    # W&B
    print("\n[3/5] Initialising Weights & Biases...")
    wandb.init(
        project="acuity-lod-perception",
        name=f"run_{int(time.time())}",
        config=config,
    )
    wandb.watch(model, log="gradients", log_freq=10)

    # Training loop
    print("\n[4/5] Training...")
    best_val_loss    = float("inf")
    best_val_srcc    = 0.0
    patience_counter = 0
    history          = []

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()

        train_loss, train_comp = run_epoch(
            model, train_loader, loss_fn, optimizer, device, training=True
        )
        val_loss, val_comp = run_epoch(
            model, val_loader, loss_fn, None, device, training=False
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        epoch_time = time.time() - t0
        log = {
            "epoch":            epoch,
            "train/loss":       train_loss,
            "train/mse":        train_comp["mse"],
            "train/rank":       train_comp["rank"],
            "train/corr":       train_comp["corr"],
            "train/srcc":       train_comp["srcc"],
            "train/plcc":       train_comp["plcc"],
            "val/loss":         val_loss,
            "val/mse":          val_comp["mse"],
            "val/rank":         val_comp["rank"],
            "val/corr":         val_comp["corr"],
            "val/srcc":         val_comp["srcc"],
            "val/plcc":         val_comp["plcc"],
            "lr":               current_lr,
            "epoch_time_s":     epoch_time,
        }
        wandb.log(log)
        history.append(log)

        # Console print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{config['epochs']} | "
                f"train={train_loss:.4f} val={val_loss:.4f} | "
                f"SRCC={val_comp['srcc']:.4f} PLCC={val_comp['plcc']:.4f} | "
                f"lr={current_lr:.2e} | {epoch_time:.1f}s"
            )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_srcc = val_comp["srcc"]
            patience_counter = 0
            torch.save(
                {
                    "epoch":      epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":   val_loss,
                    "val_srcc":   val_comp["srcc"],
                    "config":     config,
                },
                MODELS_DIR / "best_model.pth",
            )
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {config['patience']} epochs)")
                break

    # Save history
    with open(MODELS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[5/5] Training complete.")
    print(f"  Best val loss : {best_val_loss:.6f}")
    print(f"  Best val SRCC : {best_val_srcc:.4f}  (target: > 0.75)")
    print(f"  Model saved   : {MODELS_DIR / 'best_model.pth'}")

    wandb.finish()
    return best_val_srcc


if __name__ == "__main__":
    print("=" * 60)
    print("  Acuity — Model Training")
    print("=" * 60)
    srcc = train()
    print(f"\n  Final SRCC: {srcc:.4f}")
    print("=" * 60)