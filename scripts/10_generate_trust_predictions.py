"""
10 — Generate per-message prediction CSV for the trust-dynamics evaluator.

scripts/tusteva.py needs a held-out stream with columns
    sender_id, timestamp, y_true, y_pred_prob
This script builds that file by running the trained TAS-VANET model
(results/models/final_tas_vanet) on the held-out test split
(data/processed/veremi_test_processed.csv), reusing the same
normalization + inference logic as scripts/06_generate_figures.py.

Column mapping:
    sender_id   <- sender_pseudo
    timestamp   <- t_curr
    y_true      <- label
    y_pred_prob <- sigmoid(classifier head logits)   (higher = more malicious)

The source data also carries a per-message `historical_trust` feature,
which is passed through as `at_col` so it can optionally be used as the
indirect-trust term (--at-col historical_trust) in tusteva.py.

Output: results/tables/trust_eval_preds.csv

Usage:
    python scripts/10_generate_trust_predictions.py
"""

from __future__ import annotations

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

MODELS_DIR = PROJECT_ROOT / "results" / "models"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
TEST_CSV   = PROJECT_ROOT / "data" / "processed" / "veremi_test_processed.csv"
OUT_CSV    = TABLES_DIR / "trust_eval_preds.csv"


def main():
    final_model_dir = MODELS_DIR / "final_tas_vanet"
    norm_path = MODELS_DIR / "norm_params.json"

    if not (final_model_dir.exists() and TEST_CSV.exists() and norm_path.exists()):
        sys.exit(
            "Missing model/test data. Run scripts/05_train_full.py first "
            f"(need {final_model_dir}, {TEST_CSV}, {norm_path})."
        )

    with open(norm_path) as f:
        norm_params = json.load(f)

    test_df = pd.read_csv(TEST_CSV)
    feat_cols = list(norm_params.keys())
    X_test_raw = test_df[feat_cols].to_numpy(dtype=np.float32)

    lo = np.array([norm_params[c]["lo"] for c in feat_cols], dtype=np.float32)
    hi = np.array([norm_params[c]["hi"] for c in feat_cols], dtype=np.float32)
    rng = hi - lo
    rng[rng < 1e-12] = 1.0
    X_test_n = np.clip((X_test_raw - lo) / rng, 0.0, 1.0)

    with open(final_model_dir / "sae_config.json") as f:
        cfg_dict = json.load(f)
    sae_cfg = SAEConfig(**cfg_dict)
    model = StackedAutoencoder(sae_cfg)
    model.load_state_dict(
        torch.load(final_model_dir / "sae_state.pt", map_location="cpu")
    )
    model.eval()

    with torch.no_grad():
        X_te_t = torch.tensor(X_test_n, dtype=torch.float32)
        _, H_te = model(X_te_t)
        logits = model.classify_logits(X_te_t, H_te)
        probs = torch.sigmoid(logits).numpy()

    out = pd.DataFrame({
        "sender_id": test_df["sender_pseudo"],
        "timestamp": test_df["t_curr"],
        "y_true": test_df["label"].astype(int),
        "y_pred_prob": probs,
        "historical_trust": test_df["historical_trust"],
    })

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    n_atk = (out.groupby("sender_id")["y_true"].mean().round().astype(int) == 1).sum()
    n_ben = out["sender_id"].nunique() - n_atk
    print(f"wrote {OUT_CSV}  ({len(out)} rows, {n_ben} benign / {n_atk} attacker senders)")


if __name__ == "__main__":
    main()
