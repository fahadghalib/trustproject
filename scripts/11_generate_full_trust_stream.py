"""
11 — Generate a FULL per-sender message stream for the trust-dynamics
evaluator (scripts/tusteva.py), instead of the heavily downsampled
15,429-row held-out test table used for Table 4.

Why this exists: results/tables/trust_eval_preds.csv (built by
scripts/10_generate_trust_predictions.py from
data/processed/veremi_test_processed.csv) has a median of just 1 message
per sender, because that table was downsampled/capped for the
per-message detection benchmark (Section 5.4/Table 4). That is too short
a stream to reproduce a "messages until isolation" trust-dynamics number
(Section 5.3 claims a mean of 12 messages). The raw, un-downsampled test
log (data/VeReMi_test_data.csv, 958,441 rows) has much longer per-sender
streams (mean 13.1, up to 12,250 messages for the most active senders),
so we rebuild the pairwise + hybrid-trust features directly from it.

Steps:
  1. Load data/VeReMi_test_data.csv (raw BSMs, test split).
  2. core.veremi_csv_to_pairwise -> per-sender consecutive-message pairs.
  3. core.build_hybrid_features -> adds node_degree, avg_dist_neighbors,
     historical_trust, estimated_energy.
  4. Apply the ALREADY-FITTED normalization (results/models/norm_params.json,
     fit on the training partition only -- unchanged here) and run the
     trained final_tas_vanet model to get y_pred_prob per message.
  5. Write sender_id/timestamp/y_true/y_pred_prob (+ historical_trust)
     to results/tables/trust_eval_preds_full.csv.

Usage:
    python scripts/11_generate_full_trust_stream.py
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

from core import (
    SAEConfig, StackedAutoencoder,
    VeReMiLoaderConfig, veremi_csv_to_pairwise,
    TrustFeatureConfig, build_hybrid_features,
)

MODELS_DIR = PROJECT_ROOT / "results" / "models"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RAW_TEST_CSV = PROJECT_ROOT / "data" / "VeReMi_test_data.csv"
OUT_CSV = TABLES_DIR / "trust_eval_preds_full.csv"


def main():
    final_model_dir = MODELS_DIR / "final_tas_vanet"
    norm_path = MODELS_DIR / "norm_params.json"

    if not (final_model_dir.exists() and RAW_TEST_CSV.exists() and norm_path.exists()):
        sys.exit(
            "Missing model/raw data. Need "
            f"{final_model_dir}, {RAW_TEST_CSV}, {norm_path}."
        )

    print("Loading raw VeReMi test data ...")
    raw = pd.read_csv(RAW_TEST_CSV)
    print(f"  {raw.shape[0]:,} raw BSM rows, {raw['senderPseudo'].nunique():,} senders")

    print("Building pairwise kinematic features ...")
    pairs = veremi_csv_to_pairwise(raw, VeReMiLoaderConfig())
    print(f"  {len(pairs):,} pairwise rows")

    print("Building hybrid trust features (node_degree/avg_dist/historical_trust) ...")
    hybrid = build_hybrid_features(pairs, TrustFeatureConfig())
    print(f"  done: {hybrid.shape}")

    hybrid.to_csv(TABLES_DIR / "trust_eval_hybrid_full_raw.csv", index=False)

    with open(norm_path) as f:
        norm_params = json.load(f)
    feat_cols = list(norm_params.keys())

    X_raw = hybrid[feat_cols].to_numpy(dtype=np.float32)
    lo = np.array([norm_params[c]["lo"] for c in feat_cols], dtype=np.float32)
    hi = np.array([norm_params[c]["hi"] for c in feat_cols], dtype=np.float32)
    rng = hi - lo
    rng[rng < 1e-12] = 1.0
    X_n = np.clip((X_raw - lo) / rng, 0.0, 1.0)

    with open(final_model_dir / "sae_config.json") as f:
        cfg_dict = json.load(f)
    sae_cfg = SAEConfig(**cfg_dict)
    model = StackedAutoencoder(sae_cfg)
    model.load_state_dict(
        torch.load(final_model_dir / "sae_state.pt", map_location="cpu")
    )
    model.eval()

    print("Running inference ...")
    with torch.no_grad():
        X_t = torch.tensor(X_n, dtype=torch.float32)
        _, H = model(X_t)
        logits = model.classify_logits(X_t, H)
        probs = torch.sigmoid(logits).numpy()

    out = pd.DataFrame({
        "sender_id": hybrid["sender_pseudo"],
        "timestamp": hybrid["t_curr"],
        "y_true": hybrid["label"].astype(int),
        "y_pred_prob": probs,
        "historical_trust": hybrid["historical_trust"],
    })

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    counts = out.groupby("sender_id").size()
    n_atk = (out.groupby("sender_id")["y_true"].mean().round().astype(int) == 1).sum()
    n_ben = out["sender_id"].nunique() - n_atk
    print(f"\nwrote {OUT_CSV}")
    print(f"  {len(out):,} rows, {n_ben} benign / {n_atk} attacker senders")
    print(f"  messages/sender: mean={counts.mean():.1f} median={counts.median():.0f} "
          f"max={counts.max()}")


if __name__ == "__main__":
    main()
