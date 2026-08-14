"""
04 — Explore the processed VeReMi data.

Loads the processed hybrid-feature CSV produced by 02_process_veremi.py
and prints statistics:
  - Class balance and per-attack counts
  - Feature distribution (min/max/mean/std) per class
  - Top discriminative features (z-distance between malicious and legitimate)
  - Correlation matrix of hybrid features (saved to results/figures/)

This step helps you understand the data before training. Run it once after
processing to verify the data looks sensible.

Usage:
    python scripts/04_explore_processed.py \\
        --input data/processed/veremi_processed.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from core import (
    HYBRID_FEATURES,
    KINEMATIC_FEATURES_KEPT,
    TRUST_FEATURES_ADDED,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Explore the processed VeReMi hybrid feature CSV."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to processed CSV from 02_process_veremi.py.",
    )
    parser.add_argument(
        "--output_figures", default="results/figures",
        help="Where to save exploratory figures.",
    )
    args = parser.parse_args(argv)

    if not Path(args.input).exists():
        print(f"ERROR: input file not found: {args.input}")
        print("Run 02_process_veremi.py first.")
        return 1

    print("=" * 70)
    print(f"Loading: {args.input}")
    print("=" * 70)
    df = pd.read_csv(args.input)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # --- Class balance ------------------------------------------------
    print("\n--- Class balance ---")
    print(df["label"].value_counts())
    print(f"Malicious fraction: {df['label'].mean():.1%}")

    print("\n--- Per-attack counts ---")
    print(df["attack_id"].value_counts().sort_index())

    # --- Feature statistics per class ---------------------------------
    print("\n--- Trust feature statistics by label ---")
    print(df.groupby("label")[TRUST_FEATURES_ADDED].agg(["mean", "std"]).round(4))

    # --- Class separation (z-distance) --------------------------------
    print("\n--- Top discriminative features (z-distance between classes) ---")
    leg = df[df["label"] == 0][HYBRID_FEATURES]
    mal = df[df["label"] == 1][HYBRID_FEATURES]
    sep = (mal.mean() - leg.mean()).abs() / df[HYBRID_FEATURES].std().replace(0, np.nan)
    top = sep.sort_values(ascending=False).head(15)
    for feat, val in top.items():
        tag = "[trust]   " if feat in TRUST_FEATURES_ADDED else "[kinematic]"
        print(f"  {tag} {feat:<28s} = {val:.3f}")

    # --- Per-attack-type separation -----------------------------------
    print("\n--- Per-attack-type discrimination (trust features only) ---")
    print(f"  {'attack_id':>10s}  {'n':>8s}  ", end="")
    print("  ".join(f"{f[:13]:>13s}" for f in TRUST_FEATURES_ADDED))
    for aid in sorted(df[df["label"] == 1]["attack_id"].unique()):
        sub = df[df["attack_id"] == aid][HYBRID_FEATURES]
        n = len(sub)
        if n < 3:
            continue
        s = (sub.mean() - leg.mean()).abs() / df[HYBRID_FEATURES].std().replace(0, np.nan)
        print(f"  {int(aid):>10d}  {n:>8d}  ", end="")
        print("  ".join(f"{s[f]:>13.3f}" for f in TRUST_FEATURES_ADDED))

    # --- Save a correlation heatmap (figures only, no chart in stdout) ---
    fig_dir = Path(args.output_figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        corr = df[HYBRID_FEATURES].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(HYBRID_FEATURES)))
        ax.set_yticks(range(len(HYBRID_FEATURES)))
        ax.set_xticklabels(HYBRID_FEATURES, rotation=90, fontsize=8)
        ax.set_yticklabels(HYBRID_FEATURES, fontsize=8)
        fig.colorbar(im, ax=ax)
        ax.set_title("Hybrid feature correlation matrix")
        plt.tight_layout()
        out_path = fig_dir / "feature_correlation_matrix.png"
        plt.savefig(out_path, dpi=150)
        print(f"\nSaved correlation heatmap: {out_path}")
    except ImportError:
        print("(matplotlib not installed — skipping correlation heatmap)")

    print("\nExploration done.")
    print("\nNext step: train the SAE + WOA pipeline on this data")
    print("(training script will be added in the next iteration).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
