"""
python/training/error_analysis.py

Failure Case Analysis and Error Analysis Report

Reads test_predictions.csv produced by evaluate.py and:
1. Identifies failure cases (large prediction errors)
2. Analyzes error distribution by LOD level and distance
3. Saves error_analysis_report.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

PREDICTIONS_CSV   = "results/test_predictions.csv"
OUTPUT_REPORT     = "results/error_analysis_report.txt"
FIGURES_DIR       = Path("results/figures")
FAILURE_THRESHOLD = 0.1

def load_predictions():
    df = pd.read_csv(PREDICTIONS_CSV)
    print(f"Loaded {len(df)} test samples")
    print(f"Columns: {list(df.columns)}")
    return df

def compute_errors(df):
    df = df.copy()
    df["abs_error"]    = np.abs(df["predicted_score"] - df["ground_truth"])
    df["signed_error"] = df["predicted_score"] - df["ground_truth"]
    df["sq_error"]     = df["signed_error"] ** 2
    return df

def identify_failure_cases(df, threshold=FAILURE_THRESHOLD):
    failures = df[df["abs_error"] >= threshold].copy()
    failures = failures.sort_values("abs_error", ascending=False)
    return failures

def analyze_error_distribution(df):
    errors = df["abs_error"].values
    stats_dict = {
        "mean_abs_error":   float(np.mean(errors)),
        "median_abs_error": float(np.median(errors)),
        "std_abs_error":    float(np.std(errors)),
        "max_abs_error":    float(np.max(errors)),
        "min_abs_error":    float(np.min(errors)),
        "pct_within_0.05":  float(np.mean(errors < 0.05) * 100),
        "pct_within_0.10":  float(np.mean(errors < 0.10) * 100),
        "pct_within_0.20":  float(np.mean(errors < 0.20) * 100),
        "failure_count":    int(np.sum(errors >= FAILURE_THRESHOLD)),
        "failure_rate_pct": float(np.mean(errors >= FAILURE_THRESHOLD) * 100),
    }
    return stats_dict

def plot_error_distribution(df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(df["abs_error"], bins=30, color="steelblue",
                 edgecolor="white", alpha=0.8)
    axes[0].axvline(FAILURE_THRESHOLD, color="red", linestyle="--",
                    label=f"Failure threshold ({FAILURE_THRESHOLD})")
    axes[0].set_xlabel("Absolute Error")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Error Distribution")
    axes[0].legend()

    axes[1].hist(df["signed_error"], bins=30, color="coral",
                 edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="black", linestyle="-", linewidth=1)
    axes[1].set_xlabel("Signed Error (prediction - ground truth)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Signed Error Distribution")

    plt.tight_layout()
    out = FIGURES_DIR / "error_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

def save_report(df, failures, error_stats):
    lines = []
    lines.append("=" * 70)
    lines.append("ERROR ANALYSIS REPORT -- LODPerceptionNet")
    lines.append("=" * 70)
    lines.append("")

    lines.append("-- Error Distribution ----------------------------------------------")
    lines.append(f"  Total test samples:          {len(df)}")
    lines.append(f"  Mean absolute error:         {error_stats['mean_abs_error']:.6f}")
    lines.append(f"  Median absolute error:       {error_stats['median_abs_error']:.6f}")
    lines.append(f"  Std absolute error:          {error_stats['std_abs_error']:.6f}")
    lines.append(f"  Max absolute error:          {error_stats['max_abs_error']:.6f}")
    lines.append(f"  Min absolute error:          {error_stats['min_abs_error']:.6f}")
    lines.append("")
    lines.append(f"  % predictions within 0.05:  {error_stats['pct_within_0.05']:.1f}%")
    lines.append(f"  % predictions within 0.10:  {error_stats['pct_within_0.10']:.1f}%")
    lines.append(f"  % predictions within 0.20:  {error_stats['pct_within_0.20']:.1f}%")
    lines.append("")

    lines.append("-- Failure Cases ---------------------------------------------------")
    lines.append(f"  Failure threshold:           |error| >= {FAILURE_THRESHOLD}")
    lines.append(f"  Failure count:               {error_stats['failure_count']}")
    lines.append(f"  Failure rate:                {error_stats['failure_rate_pct']:.1f}%")
    lines.append("")

    if len(failures) > 0:
        lines.append("  Top 10 worst predictions:")
        lines.append(f"  {'#':<4} {'Prediction':>12} {'Ground Truth':>13} "
                     f"{'Abs Error':>10}")
        lines.append("  " + "-" * 45)
        for i, (_, row) in enumerate(failures.head(10).iterrows()):
            lines.append(f"  {i+1:<4} {row['predicted_score']:>12.6f} "
                         f"{row['ground_truth']:>13.6f} "
                         f"{row['abs_error']:>10.6f}")
    else:
        lines.append("  No failure cases found -- all predictions within threshold.")
    lines.append("")

    lines.append("-- Error Analysis --------------------------------------------------")

    mean_signed = df["signed_error"].mean()
    if abs(mean_signed) < 0.01:
        bias_str = "minimal (< 0.01)"
    elif mean_signed > 0:
        bias_str = f"slight over-prediction (+{mean_signed:.4f})"
    else:
        bias_str = f"slight under-prediction ({mean_signed:.4f})"
    lines.append(f"  Prediction bias:             {bias_str}")

    _, p_value = stats.shapiro(df["signed_error"].values[:50])
    lines.append(f"  Error normality (Shapiro-Wilk p): {p_value:.4f}")
    lines.append("")

    lines.append("-- Conclusion ------------------------------------------------------")
    if error_stats["failure_rate_pct"] < 5.0:
        lines.append("  Model performance is GOOD. Failure rate below 5%.")
    elif error_stats["failure_rate_pct"] < 15.0:
        lines.append("  Model performance is ACCEPTABLE. Some failure cases present.")
    else:
        lines.append("  Model performance needs improvement. High failure rate.")
    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {OUTPUT_REPORT}")

def main():
    df       = load_predictions()
    df       = compute_errors(df)
    failures = identify_failure_cases(df)
    stats_d  = analyze_error_distribution(df)
    plot_error_distribution(df)
    save_report(df, failures, stats_d)

if __name__ == "__main__":
    main()