"""
03 — Synthetic demo (no data download required).

End-to-end demonstration of the SAE + WOA pipeline on synthetic VANET
trust features. Useful to verify the algorithm logic before running on
real VeReMi data.

WARNING: the numbers produced here are NOT for publication. Synthetic
data only validates that the pipeline runs end-to-end and that the
ablation conditions behave as expected. All published results must
come from the real VeReMi Extension data.

Usage:
    python scripts/03_synthetic_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from core import (
    SAEConfig,
    StackedAutoencoder,
    SyntheticConfig,
    TrustPipeline,
    WOAConfig,
    generate_synthetic_trust_data,
    train_sae,
    train_val_test_split,
)


def evaluate_detection(y_true, y_pred, label: str) -> dict:
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  --- {label} ---")
    print(f"    Precision = {p:.4f}")
    print(f"    Recall    = {r:.4f}")
    print(f"    F1-score  = {f1:.4f}")
    print(f"    Confusion [TN FP / FN TP] = {cm.tolist()}")
    return {"precision": p, "recall": r, "f1": f1}


def main() -> int:
    print("=" * 70)
    print("Synthetic-data sanity check for the TAS-VANET pipeline")
    print("(numbers below are NOT for publication)")
    print("=" * 70)

    # ----- 1. Synthetic data ------------------------------------------
    print("\n[Step 1] Generating synthetic trust features...")
    X, y = generate_synthetic_trust_data(SyntheticConfig(n_legitimate=800, n_malicious=200))
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
    print(f"  Train: {X_train.shape[0]}  ({y_train.sum()} malicious)")
    print(f"  Val:   {X_val.shape[0]}    ({y_val.sum()} malicious)")
    print(f"  Test:  {X_test.shape[0]}   ({y_test.sum()} malicious)")

    # ----- 2. Full TAS-VANET pipeline ---------------------------------
    print("\n[Step 2] Full pipeline (WOA-optimized SAE)...")
    pipe = TrustPipeline(
        threshold_k=-0.5,
        woa_config=WOAConfig(population_size=10, max_generations=10, seed=42),
        final_epochs=150,
    )
    t0 = time.time()
    result = pipe.calibrate(X_train, X_val, y_val, y_train, verbose=True)
    elapsed = time.time() - t0
    print(f"\n  Calibration time: {elapsed:.1f}s")
    print(f"  Final WOA fitness: {result.woa_best_fitness:.5f}")

    y_pred_full = pipe.classify(X_test)
    full_metrics = evaluate_detection(y_test, y_pred_full, "TAS-VANET (SAE + WOA)")

    # ----- 3. Ablation: SAE only (no WOA) -----------------------------
    print("\n[Step 3] Ablation: SAE with default hyperparameters (no WOA)...")
    cfg = SAEConfig(
        input_dim=X.shape[1],
        hidden_dims=[16, 8],
        learning_rate=1e-3,
        sparsity_target=0.05,
        sparsity_weight=3.0,
        epochs=150,
    )
    model_default = StackedAutoencoder(cfg)
    train_sae(
        model_default,
        torch.tensor(X_train, dtype=torch.float32),
        X_val=torch.tensor(X_val, dtype=torch.float32),
        verbose=False,
    )
    model_default.eval()
    with torch.no_grad():
        _, H_train_def = model_default(torch.tensor(X_train, dtype=torch.float32))
        _, H_test_def = model_default(torch.tensor(X_test, dtype=torch.float32))
    primary_train = H_train_def.numpy()[:, 0]
    threshold = primary_train.mean() - 0.5 * primary_train.std()
    y_pred_no_woa = (H_test_def.numpy()[:, 0] < threshold).astype(int)
    sae_only_metrics = evaluate_detection(y_test, y_pred_no_woa, "SAE only (no WOA)")

    # ----- 4. Simple threshold baseline -------------------------------
    print("\n[Step 4] Baseline: simple threshold on historical_trust feature...")
    y_pred_simple = (X_test[:, 3] < 0.5).astype(int)
    simple_metrics = evaluate_detection(y_test, y_pred_simple, "Simple threshold")

    # ----- 5. Ablation summary ----------------------------------------
    print("\n" + "=" * 70)
    print("Ablation summary (synthetic data, algorithm validation only)")
    print("=" * 70)
    rows = [
        ("Simple threshold",  simple_metrics),
        ("SAE only (no WOA)", sae_only_metrics),
        ("SAE + WOA (full)",  full_metrics),
    ]
    print(f"{'Configuration':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    for name, m in rows:
        print(
            f"{name:<25} {m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} {m['f1']:>10.4f}"
        )

    # ----- 6. Save model ---------------------------------------------
    out_dir = PROJECT_ROOT / "results" / "models" / "synthetic_demo"
    pipe.save(str(out_dir))
    print(f"\nModel saved to: {out_dir}")
    print("\nReminder: real published results must come from VeReMi data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
