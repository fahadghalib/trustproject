"""
Synthetic VANET Trust Feature Generator (for pre-Veins validation only).

This module generates synthetic trust feature vectors that mimic what Veins
will output during the actual simulation. It is used ONLY to:

  1. Validate that the SAE + WOA pipeline runs end-to-end before integration.
  2. Provide a sanity check on the ablation logic before spending compute on
     hundreds of Veins runs.

THIS IS NOT EXPERIMENTAL DATA FOR THE PAPER. All published results must come
from Veins/OMNeT++/SUMO simulations of actual VANET behavior.

Feature schema (matches Veins extraction):
    [0] residual_energy       in [0, 1]
    [1] avg_distance_neighbors in [0, 1]  (normalized by transmission range)
    [2] node_degree            in [0, 1]  (normalized by max observed)
    [3] historical_trust       in [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SyntheticConfig:
    n_legitimate: int = 800
    n_malicious: int = 200
    seed: int = 123


def generate_synthetic_trust_data(
    config: SyntheticConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic VANET trust dataset.

    Legitimate vehicles tend to have:
        - moderate-to-high residual energy
        - moderate distance to neighbors
        - varying node degree
        - high historical trust (clustered near 1.0)

    Malicious vehicles exhibit perturbed patterns:
        - more variable energy (possibly spent on attacks)
        - either too close (cluster targeting) or too far (selfish positioning)
        - sometimes abnormally high node degree (Sybil-like)
        - low historical trust (caught misbehaving before)

    Returns
    -------
    X : np.ndarray of shape (n_total, 4), values in [0, 1]
    y : np.ndarray of shape (n_total,), 1=malicious, 0=legitimate
    """
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    # Legitimate distribution
    leg_energy = rng.beta(5, 2, cfg.n_legitimate)                       # right-skewed high
    leg_distance = rng.beta(2, 2, cfg.n_legitimate)                     # centered
    leg_degree = rng.beta(2, 3, cfg.n_legitimate)                       # mild-low
    leg_hist_trust = rng.beta(8, 2, cfg.n_legitimate)                   # high, near 1

    X_leg = np.column_stack([leg_energy, leg_distance, leg_degree, leg_hist_trust])
    y_leg = np.zeros(cfg.n_legitimate, dtype=int)

    # Malicious distribution
    mal_energy = rng.beta(2, 2, cfg.n_malicious)                        # more variable
    mal_distance = np.concatenate(
        [
            rng.beta(1.5, 5, cfg.n_malicious // 2),                     # too close
            rng.beta(5, 1.5, cfg.n_malicious - cfg.n_malicious // 2),   # too far
        ]
    )
    rng.shuffle(mal_distance)
    mal_degree = rng.beta(3, 2, cfg.n_malicious)                        # high (Sybil-like)
    mal_hist_trust = rng.beta(2, 7, cfg.n_malicious)                    # low

    X_mal = np.column_stack([mal_energy, mal_distance, mal_degree, mal_hist_trust])
    y_mal = np.ones(cfg.n_malicious, dtype=int)

    # Shuffle together
    X = np.vstack([X_leg, X_mal])
    y = np.concatenate([y_leg, y_mal])

    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train/validation/test sets (stratified by label).

    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    def split(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(idx)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)
        return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]

    tr_pos, vl_pos, te_pos = split(idx_pos)
    tr_neg, vl_neg, te_neg = split(idx_neg)
    tr = np.concatenate([tr_pos, tr_neg]); rng.shuffle(tr)
    vl = np.concatenate([vl_pos, vl_neg]); rng.shuffle(vl)
    te = np.concatenate([te_pos, te_neg]); rng.shuffle(te)
    return X[tr], y[tr], X[vl], y[vl], X[te], y[te]
