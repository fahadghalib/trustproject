"""
12 — Rebuild processed VeReMi features with the density step computed
BEFORE downsampling (fixes node_degree/avg_dist_neighbors distribution
shift diagnosed while investigating Section 5.3 trust dynamics).

Bug being fixed: scripts/02_process_veremi.py samples raw BSM rows down
to a fraction FIRST, then computes node_degree/avg_dist_neighbors (a
5-second/300m sliding-window neighbor count) on the already-thinned
stream. That understates true vehicular density: the existing
data/processed/veremi_{train,test}_processed.csv cap node_degree at 5,
while computing the same feature on the full, un-thinned raw log
(data/VeReMi_{train,test}_data.csv) reaches up to 38. The model trained
on the artificially sparse feature is miscalibrated on realistic
(longer, denser) streams -- confirmed empirically: median y_pred_prob
for BENIGN messages was 0.78 (should be well under 0.5) when the
existing model scored features built from full-density data.

Fix: compute pairwise + hybrid (density) features on the FULL raw
train/test logs first, THEN downsample per attack_id class to match the
row budget of the original processed files (so training cost / dataset
scale stays comparable to the paper's existing numbers -- only the
node_degree/avg_dist_neighbors distribution changes, not the dataset
size).

Output:
    data/processed/veremi_train_processed_v2.csv
    data/processed/veremi_test_processed_v2.csv

Usage:
    python scripts/12_rebuild_features_correct_order.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import (
    VeReMiLoaderConfig, veremi_csv_to_pairwise,
    TrustFeatureConfig, build_hybrid_features,
)

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "processed"
SEED = 42

# Row budget per attack_id, taken from the existing processed files so
# the new (correctly-ordered) datasets stay the same scale.
TRAIN_CAPS = {
    0: 39485, 1: 881, 2: 902, 3: 861, 4: 870, 5: 867, 6: 899, 7: 860,
    8: 871, 9: 921, 10: 903, 11: 900, 12: 878, 13: 6144, 14: 5984,
    15: 6028, 16: 4120, 17: 207, 19: 19,
}
TEST_CAPS = {
    0: 7828, 1: 185, 2: 189, 3: 180, 4: 163, 5: 179, 6: 179, 7: 169,
    8: 180, 9: 167, 10: 178, 11: 189, 12: 185, 13: 1541, 14: 1512,
    15: 1467, 16: 895, 17: 36, 19: 7,
}


def build_split(raw_csv: Path, caps: dict, out_csv: Path, label: str):
    print(f"=== {label} ===")
    t0 = time.time()
    raw = pd.read_csv(raw_csv)
    print(f"  loaded raw: {raw.shape} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    pairs = veremi_csv_to_pairwise(raw, VeReMiLoaderConfig())
    print(f"  pairwise: {pairs.shape} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    hybrid = build_hybrid_features(pairs, TrustFeatureConfig())
    print(f"  hybrid features (full density): {hybrid.shape} ({time.time()-t0:.1f}s)")
    print(f"  node_degree max = {hybrid['node_degree'].max():.0f} "
          f"(old processed file capped at 5)")

    rng = np.random.default_rng(SEED)
    parts = []
    for aid, grp in hybrid.groupby("attack_id", sort=False):
        n_take = caps.get(int(aid))
        if n_take is None:
            continue
        if n_take >= len(grp):
            parts.append(grp)
        else:
            idx = rng.choice(len(grp), size=n_take, replace=False)
            parts.append(grp.iloc[idx])
    out = pd.concat(parts, ignore_index=True)
    print(f"  downsampled to budget: {out.shape}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}\n")


def main():
    build_split(
        DATA_DIR / "VeReMi_train_data.csv", TRAIN_CAPS,
        OUT_DIR / "veremi_train_processed_v2.csv", "TRAIN",
    )
    build_split(
        DATA_DIR / "VeReMi_test_data.csv", TEST_CAPS,
        OUT_DIR / "veremi_test_processed_v2.csv", "TEST",
    )


if __name__ == "__main__":
    main()
