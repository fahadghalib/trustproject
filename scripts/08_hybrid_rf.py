"""
08 — Hybrid RF: augment Random Forest with the SAE's learned latent
representation instead of pitting SAE against RF as competitors.

Rationale
---------
Across scripts/05 (standard split) and scripts/07 (leave-one-attack-out),
Random Forest consistently outperforms TAS-VANET on raw features. Rather
than continuing to try to beat RF with the SAE classifier head alone,
this tests whether RF benefits from having the SAE's learned trust
representation available as ADDITIONAL input features:

    X_aug = [raw_features ; h]   where h = SAE.encode(raw_features)

This is a standard feature-fusion / stacking pattern (autoencoder features
feeding a tree/gradient-boosted backend), documented in the intrusion-
detection literature. If h carries information RF cannot already extract
from the raw features, X_aug should beat raw-feature RF; if h is just a
lossy compression of information RF already has, it should not.

Reuses the WOA-tuned architecture (hidden_dims, learning_rate, sparsity)
already found by scripts/05_train_full.py (results/models/final_tas_vanet/
sae_config.json), but retrains the SAE fresh rather than loading the saved
weights: the saved checkpoint predates the skip-connection classifier head
(core/sae_model.py) added since that run, so its classifier layer has an
incompatible shape. Retraining is cheap (single run, no WOA search).

Usage:
    python scripts/08_hybrid_rf.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import SAEConfig, StackedAutoencoder


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Hybrid RF + SAE latent features.")
    p.add_argument("--hidden_activation", default="sigmoid",
                   choices=["sigmoid", "leaky_relu"],
                   help="Encoder/decoder intermediate-layer activation "
                        "(bottleneck and reconstruction output always stay "
                        "Sigmoid regardless — see core/sae_model.py).")
    return p.parse_args(argv)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, accuracy_score, roc_auc_score,
    )
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = float("nan")
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "auc_roc":   auc,
    }


def main(argv=None):
    args = parse_args(argv)
    models_dir = PROJECT_ROOT / "results" / "models"
    final_dir = models_dir / "final_tas_vanet"

    with open(models_dir / "norm_params.json") as f:
        norm_params = json.load(f)
    feature_cols = list(norm_params.keys())
    lo = np.array([norm_params[c]["lo"] for c in feature_cols], dtype=np.float32)
    hi = np.array([norm_params[c]["hi"] for c in feature_cols], dtype=np.float32)
    rng = hi - lo
    rng[rng < 1e-12] = 1.0

    with open(final_dir / "sae_config.json") as f:
        cfg_dict = json.load(f)
    cfg_dict["hidden_activation"] = args.hidden_activation
    sae_cfg = SAEConfig(**cfg_dict)
    print(f"hidden_activation = {args.hidden_activation}")

    train_df = pd.read_csv(PROJECT_ROOT / "data/processed/veremi_train_processed.csv")
    test_df  = pd.read_csv(PROJECT_ROOT / "data/processed/veremi_test_processed.csv")

    X_train_raw = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train     = train_df["label"].to_numpy(dtype=np.int32)
    X_test_raw  = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test      = test_df["label"].to_numpy(dtype=np.int32)

    X_train_n = np.clip((X_train_raw - lo) / rng, 0.0, 1.0)
    X_test_n  = np.clip((X_test_raw - lo) / rng, 0.0, 1.0)

    # Retrain fresh with the WOA-tuned architecture (the saved checkpoint
    # predates the skip-connection classifier head, so its weights don't
    # match the current model shape — see module docstring).
    from core import train_sae
    torch.manual_seed(sae_cfg.seed)
    model = StackedAutoencoder(sae_cfg)
    X_train_t = torch.tensor(X_train_n, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    print("Training SAE (WOA-tuned architecture, skip-connection head) ...")
    train_sae(model, X_train_t, y_train=y_train_t, verbose=False)
    model.eval()

    with torch.no_grad():
        H_train = model.encode(torch.tensor(X_train_n, dtype=torch.float32)).numpy()
        H_test  = model.encode(torch.tensor(X_test_n,  dtype=torch.float32)).numpy()

    X_train_aug = np.concatenate([X_train_n, H_train], axis=1)
    X_test_aug  = np.concatenate([X_test_n,  H_test],  axis=1)

    from sklearn.ensemble import RandomForestClassifier

    print("=" * 70)
    print("Hybrid RF: raw features + SAE latent representation")
    print("=" * 70)
    print(f"Raw feature dim: {X_train_n.shape[1]}   Latent dim: {H_train.shape[1]}   "
          f"Augmented dim: {X_train_aug.shape[1]}")

    results = {}

    print("\nPlain RF (raw features only) ...")
    clf_plain = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    clf_plain.fit(X_train_n, y_train)
    yp = clf_plain.predict(X_test_n)
    ys = clf_plain.predict_proba(X_test_n)[:, 1]
    results["rf_plain"] = compute_metrics(y_test, yp, ys)

    hybrid_key = "rf_hybrid" if args.hidden_activation == "sigmoid" else f"rf_hybrid_{args.hidden_activation}"
    print(f"Hybrid RF (raw + SAE latent, {args.hidden_activation}) ...")
    clf_hybrid = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    clf_hybrid.fit(X_train_aug, y_train)
    yp = clf_hybrid.predict(X_test_aug)
    ys = clf_hybrid.predict_proba(X_test_aug)[:, 1]
    results[hybrid_key] = compute_metrics(y_test, yp, ys)

    # Feature importance: how much does RF actually lean on the latent
    # dims vs the raw dims? (diagnostic, not a hypothesis test)
    importances = clf_hybrid.feature_importances_
    n_raw = X_train_n.shape[1]
    raw_importance = float(importances[:n_raw].sum())
    latent_importance = float(importances[n_raw:].sum())

    print("\n" + "=" * 70)
    print("RESULTS (held-out test set)")
    print("=" * 70)
    for name, m in results.items():
        print(f"  {name:<12s} P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}")

    print(f"\nHybrid RF feature-importance split: "
          f"raw={raw_importance:.3f}  latent(SAE h)={latent_importance:.3f}")

    out_path = PROJECT_ROOT / "results" / "tables" / "hybrid_rf.csv"
    new_rows = pd.DataFrame([{"method": k, **v} for k, v in results.items()])
    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing = existing[~existing["method"].isin(new_rows["method"])]
        new_rows = pd.concat([existing, new_rows], ignore_index=True)
    new_rows.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    delta_f1 = results[hybrid_key]["f1"] - results["rf_plain"]["f1"]
    print(f"\nDelta F1 ({hybrid_key} - plain RF): {delta_f1:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
