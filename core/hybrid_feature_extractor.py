"""
Hybrid Trust + Kinematic Feature Extractor for F2MD output.

This module reads F2MD v2.x output (pairwise BSM features) and adds four
trust-based features derived from the simulation data:

  1. node_degree         - sender's observed neighborhood size in a time window
  2. avg_dist_neighbors  - mean distance from sender to other vehicles in window
  3. historical_trust    - cumulative plausibility-based trust score per sender
  4. estimated_energy    - per-sender energy estimate from mobility + comms cost

The output is a single DataFrame combining:
  - Original F2MD kinematic features (selected subset)
  - Computed trust features (4 new columns)
  - The label / attack_id columns from F2MD

This becomes the input to the SAE + WOA pipeline, replacing the
trust-only feature vector used in the original paper.

NOTE on residual_energy:
  True residual energy requires Veins' BatteryModule to be enabled and logged.
  F2MD by default does NOT log energy. We provide a defensible ESTIMATE based
  on observable behavior (transmission rate + mobility), and clearly disclose
  this in the methodology section as "behavior-derived energy proxy".
  If you later enable Veins energy logging, this column can be replaced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrustFeatureConfig:
    """Tunable parameters for trust-feature derivation."""

    # Sliding window used for node_degree and avg_dist_neighbors (seconds)
    window_seconds: float = 5.0

    # Communication range used as the neighborhood cutoff (meters)
    comm_range_m: float = 300.0

    # Trust EMA smoothing factor (alpha in trust = (1-a)*prev + a*plausibility)
    trust_alpha: float = 0.2

    # Plausibility thresholds (used to score each BSM)
    max_plausible_speed: float = 50.0       # m/s (180 km/h)
    max_plausible_acc:   float =  5.0       # m/s^2 (hard braking)
    max_plausible_jerk:  float = 10.0       # m/s^3
    max_position_jump:   float = 60.0       # m per second (~216 km/h)

    # Speed-position consistency: |observed_dist - speed*dt| / (speed*dt+eps) < tol
    # Catches Constant Position attacks (speed claimed > 0 but position stationary)
    speed_pos_tolerance: float = 0.5        # 50% relative deviation allowed
    speed_pos_eps:       float = 0.1        # m, avoids div-by-zero for stationary cars

    # Energy model constants (normalized units, monotonic estimate)
    energy_initial:      float = 1.0
    energy_tx_cost:      float = 1.0e-4     # per BSM observed
    energy_mob_cost:     float = 5.0e-6     # per (m/s) per second


# ---------------------------------------------------------------------------
# Feature 1: plausibility-based trust history
# ---------------------------------------------------------------------------

def _row_plausibility(df: pd.DataFrame, cfg: TrustFeatureConfig) -> pd.Series:
    """Per-row plausibility in [0, 1].

    1.0 = fully plausible BSM; 0.0 = clearly implausible.
    Derived from FIVE physics-consistency checks.
    """
    spd_ok  = (df["speed_curr"].abs()        < cfg.max_plausible_speed).astype(float)
    acc_ok  = (df["acc_curr"].abs()          < cfg.max_plausible_acc).astype(float)
    jrk_ok  = (df["jerk"].abs()              < cfg.max_plausible_jerk).astype(float)

    # Position-jump consistency: distance moved per second should be
    # bounded by max_position_jump.
    dt_safe = df["dt"].replace(0, np.nan)
    pos_rate = (df["dist"] / dt_safe).fillna(0.0)
    pos_ok = (pos_rate < cfg.max_position_jump).astype(float)

    # NEW: speed-position consistency. Catches Constant Position attacks
    # where the BSM claims a non-zero speed but the position does not move.
    # Compares observed distance moved vs expected distance = |speed| * dt.
    expected = df["speed_curr"].abs() * df["dt"]
    deviation = (df["dist"] - expected).abs()
    # Relative error, with epsilon to avoid div-by-zero at stationary legit cars
    rel_err = deviation / (expected.abs() + cfg.speed_pos_eps)
    consistency_ok = (rel_err < cfg.speed_pos_tolerance).astype(float)

    # Equal-weight average of the five checks
    return (spd_ok + acc_ok + jrk_ok + pos_ok + consistency_ok) / 5.0


def compute_historical_trust(
    df: pd.DataFrame, cfg: TrustFeatureConfig
) -> pd.Series:
    """Per-row historical trust as EMA of past plausibility, per sender.

    trust(t) = (1 - alpha) * trust(t-1) + alpha * plausibility(t)
    """
    plaus = _row_plausibility(df, cfg)
    sender_groups = df.groupby("sender_pseudo")

    out = pd.Series(index=df.index, dtype=float)
    a = cfg.trust_alpha
    for _sender, idx in sender_groups.groups.items():
        idx_sorted = df.loc[idx].sort_values("t_curr").index
        running = 1.0   # start with full trust
        for i in idx_sorted:
            running = (1.0 - a) * running + a * plaus.loc[i]
            out.loc[i] = running
    return out


# ---------------------------------------------------------------------------
# Feature 2 & 3: node_degree and avg_distance using sliding time windows
# ---------------------------------------------------------------------------

def compute_neighborhood_features(
    df: pd.DataFrame, cfg: TrustFeatureConfig
) -> pd.DataFrame:
    """Compute (node_degree, avg_dist_neighbors) for each row.

    For each row r at time t with sender S:
      - node_degree(r): number of DISTINCT vehicles whose BSMs were observed
        anywhere in the window [t - W, t]
      - avg_dist_neighbors(r): mean Euclidean distance from S's position at
        time t to the positions of those distinct vehicles in the window,
        truncated by comm_range_m
    """
    out_degree = np.zeros(len(df), dtype=float)
    out_dist   = np.zeros(len(df), dtype=float)

    # Pre-sort once for windowed lookups
    df_sorted = df.sort_values("t_curr").reset_index()
    t_arr = df_sorted["t_curr"].to_numpy()
    sender_arr = df_sorted["sender_pseudo"].to_numpy()
    x_arr = df_sorted["x_curr"].to_numpy()
    y_arr = df_sorted["y_curr"].to_numpy()

    W = cfg.window_seconds
    R = cfg.comm_range_m
    n = len(df_sorted)

    # Two-pointer sliding window
    left = 0
    for i in range(n):
        t_i = t_arr[i]
        while t_arr[left] < t_i - W:
            left += 1

        # All rows in window: [left, i]
        window_senders = sender_arr[left : i + 1]
        window_x = x_arr[left : i + 1]
        window_y = y_arr[left : i + 1]

        # Distinct vehicles (exclude self)
        self_sender = sender_arr[i]
        mask_other = window_senders != self_sender
        if mask_other.any():
            other_senders = window_senders[mask_other]
            other_x = window_x[mask_other]
            other_y = window_y[mask_other]

            # Keep last seen position per other sender
            df_other = pd.DataFrame(
                {"s": other_senders, "x": other_x, "y": other_y}
            )
            last_seen = df_other.groupby("s").last()

            dx = last_seen["x"].to_numpy() - x_arr[i]
            dy = last_seen["y"].to_numpy() - y_arr[i]
            distances = np.sqrt(dx * dx + dy * dy)

            # Only count neighbors within comm range
            within_range = distances < R
            out_degree[i] = int(within_range.sum())
            if within_range.any():
                out_dist[i] = float(distances[within_range].mean())
            else:
                out_dist[i] = R               # capped at comm range
        else:
            out_degree[i] = 0
            out_dist[i] = R

    # Map back to original ordering
    out_df = pd.DataFrame(
        {
            "node_degree": out_degree,
            "avg_dist_neighbors": out_dist,
        },
        index=df_sorted["index"].to_numpy(),
    ).sort_index()
    return out_df


# ---------------------------------------------------------------------------
# Feature 4: behavior-derived energy estimate
# ---------------------------------------------------------------------------

def compute_estimated_energy(
    df: pd.DataFrame, cfg: TrustFeatureConfig
) -> pd.Series:
    """Per-row monotonically-decreasing energy estimate per sender.

    energy(t) = energy_initial
              - tx_cost * (#BSMs seen up to t for this sender)
              - mob_cost * cumulative (speed * dt) for this sender

    Clipped to [0, energy_initial].
    """
    out = pd.Series(index=df.index, dtype=float)
    for _sender, idx in df.groupby("sender_pseudo").groups.items():
        sub = df.loc[idx].sort_values("t_curr")
        cum_tx = np.arange(1, len(sub) + 1, dtype=float)         # 1, 2, 3...
        cum_mob = (sub["speed_curr"].abs() * sub["dt"]).cumsum().to_numpy()
        e = cfg.energy_initial - cfg.energy_tx_cost * cum_tx - cfg.energy_mob_cost * cum_mob
        e = np.clip(e, 0.0, cfg.energy_initial)
        out.loc[sub.index] = e
    return out


# ---------------------------------------------------------------------------
# Top-level: hybrid feature extraction
# ---------------------------------------------------------------------------

KINEMATIC_FEATURES_KEPT = [
    "dt",
    "dist",
    "speed_curr",
    "dv",
    "jerk",
    "acc_curr",
    "dacc",
    "dtheta",
    "heading_rate",
    "rate_msgs_per_s",
    "pos_conf_x_curr",
    "pos_conf_y_curr",
    "spd_conf_x_curr",
    "spd_conf_y_curr",
    "acc_conf_x_curr",
    "acc_conf_y_curr",
    "head_conf_x_curr",
    "head_conf_y_curr",
]
TRUST_FEATURES_ADDED = [
    "node_degree",
    "avg_dist_neighbors",
    "historical_trust",
    "estimated_energy",
]
HYBRID_FEATURES = KINEMATIC_FEATURES_KEPT + TRUST_FEATURES_ADDED


def build_hybrid_features(
    f2md_df: pd.DataFrame, cfg: TrustFeatureConfig | None = None
) -> pd.DataFrame:
    """Take a raw F2MD output frame and add the 4 trust features.

    Returns a DataFrame with columns:
        receiver_pseudo, sender_pseudo, t_curr,
        <KINEMATIC_FEATURES_KEPT>,
        <TRUST_FEATURES_ADDED>,
        label, attack_id
    """
    cfg = cfg or TrustFeatureConfig()
    df = f2md_df.copy()

    df["historical_trust"]  = compute_historical_trust(df, cfg)
    neigh = compute_neighborhood_features(df, cfg)
    df["node_degree"]       = neigh["node_degree"]
    df["avg_dist_neighbors"] = neigh["avg_dist_neighbors"]
    df["estimated_energy"]  = compute_estimated_energy(df, cfg)

    keep = (
        ["receiver_pseudo", "sender_pseudo", "t_curr"]
        + KINEMATIC_FEATURES_KEPT
        + TRUST_FEATURES_ADDED
        + ["label", "attack_id"]
    )
    return df[keep]


def normalize_for_sae(
    df: pd.DataFrame, feature_cols: Iterable[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """Min-max normalize the given feature columns to [0, 1].

    Returns:
        (normalized_df, scaler_params)
    where scaler_params is a dict {col: (min, max)} suitable for re-applying
    the same scaling at inference time on Phase 3 data.
    """
    feature_cols = list(feature_cols) if feature_cols else HYBRID_FEATURES
    out = df.copy()
    params: dict = {}
    for c in feature_cols:
        col = df[c].astype(float)
        lo, hi = float(col.min()), float(col.max())
        if hi - lo < 1e-12:
            out[c] = 0.0
            params[c] = (lo, hi)
            continue
        out[c] = (col - lo) / (hi - lo)
        params[c] = (lo, hi)
    return out, params


def apply_normalization(
    df: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Re-apply a previously-fit min-max normalization."""
    out = df.copy()
    for c, (lo, hi) in params.items():
        if c not in out.columns:
            continue
        if hi - lo < 1e-12:
            out[c] = 0.0
            continue
        out[c] = np.clip((out[c].astype(float) - lo) / (hi - lo), 0.0, 1.0)
    return out
