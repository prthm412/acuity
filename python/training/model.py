import torch
import torch.nn as nn
import torch.nn.functional as F

# Model
class LODPerceptionNet(nn.Module):
    def __init__(self, input_dim: int = 38, dropout: float = 0.3):
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1: 38 -> 256
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3: 128 -> 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 4: 64 -> 32
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Output: 32 -> 1
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 38) normalized feature tensor
        Returns:
            (batch_size, 1) quality score tensor in [0, 1]
        """
        return  self.network(x)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

# Loss-funtion

class PerceptualLODLoss(nn.Module):
    def __init__(self, alpha: float = 0.6, beta: float = 0.2, gamma: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.mse_loss = nn.MSELoss()
        self.ranking_loss = nn.MarginRankingLoss(margin=0.05)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        pred = predictions.squeeze(1)   # (B,)
        tgt = targets.squeeze(1)        # (B,)

        # Term 1: MSE
        mse = self.mse_loss(pred, tgt)

        # Term 2: Ranking
        # Build all pairs in the batch
        # pred_i, pred_j, label (+1 if tgt_i > tgt_j else -1)
        B = pred.shape[0]
        if B > 1:
            idx_i = torch.arange(B, device=pred.device).unsqueeze(1).expand(B, B)
            idx_j = torch.arange(B, device=pred.device).unsqueeze(0).expand(B, B)
            mask = idx_i != idx_j

            pred_i = pred[idx_i[mask]]
            pred_j = pred[idx_j[mask]]
            tgt_i  = tgt[idx_i[mask]]
            tgt_j  = tgt[idx_j[mask]]

            labels = torch.sign(tgt_i - tgt_j)
            # MarginRankingLoss ignores pairs where label == 0
            nonzero = labels != 0
            if nonzero.any():
                rank = self.ranking_loss(
                    pred_i[nonzero], pred_j[nonzero], labels[nonzero]
                )
            else:
                rank = torch.tensor(0.0, device=pred.device)
        else:
            rank = torch.tensor(0.0, device=pred.device)

        # Term 3: Correlation
        if B > 1:
            pred_mean = pred.mean()
            tgt_mean  = tgt.mean()
            pred_c    = pred - pred_mean
            tgt_c     = tgt - tgt_mean
            num       = (pred_c * tgt_c).sum()
            denom     = torch.sqrt((pred_c**2).sum() * (tgt_c**2).sum()) + 1e-8
            pearson   = num / denom
            corr_loss = 1.0 - pearson
        else:
            corr_loss = torch.tensor(0.0, device=pred.device)

        # Combined
        total = self.alpha * mse + self.beta * rank + self.gamma * corr_loss

        components = {
            "mse": mse.item(),
            "rank": rank.item(),
            "corr": corr_loss.item(),
        }
        
        return total, components


# Quick test

if __name__ == "__main__":
    print("=" * 50)
    print("  LODPerceptionNet: Architecture Test")
    print("=" * 50)

    model = LODPerceptionNet(input_dim=38, dropout=0.3)
    model.eval()

    # Forward pass
    batch = torch.randn(16, 38)
    output = model(batch)

    print(f"\n  Input shape  : {batch.shape}")
    print(f"  Output shape : {output.shape}")
    print(f"  Output range : [{output.min():.4f}, {output.max():.4f}]  (should be in [0,1])")
    print(f"  Parameters   : {model.count_parameters():,}")

    # Loss test
    loss_fn = PerceptualLODLoss(alpha=0.6, beta=0.2, gamma=0.2)
    targets = torch.rand(16, 1)
    loss, components = loss_fn(output, targets)

    print(f"\n  Loss test:")
    print(f"    Total loss : {loss.item():.6f}")
    print(f"    MSE        : {components['mse']:.6f}")
    print(f"    Rank       : {components['rank']:.6f}")
    print(f"    Corr       : {components['corr']:.6f}")

    # Gradient flow check
    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters())
    print(f"\n Gradient flow : {'OK: all parameters have gradients' if grad_ok else 'FAILED'}")

    print("\n" + "=" * 50)
    print("  model.py ready.")
    print("=" * 50)