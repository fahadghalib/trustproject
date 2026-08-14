"""
VeReMi Extended Loader.

Reads VeReMi Extended raw BSM logs (CSV or JSON) and converts them into a
pairwise per-sender feature DataFrame compatible with hybrid_feature_extractor.

VeReMi Extended raw schema (per BSM row)
----------------------------------------
    type, sendTime, sender, senderPseudo, messageID, class,
    posx, posy, posz, posx_n, posy_n, posz_n,
    spdx, spdy, spdz, spdx_n, spdy_n, spdz_n,
    aclx, acly, aclz, aclx_n, acly_n, aclz_n,
    hedx, hedy, hedz, hedx_n, hedy_n, hedz_n

Note: positions/speeds without "_n" suffix are the values reported by the
sender in the BSM (which a malicious sender may falsify). The "_n" suffix
columns are ground-truth-with-sensor-noise that are NOT available at a
real receiver — we therefore use the non-"_n" columns as detection input,
matching what a real-world VANET detector would actually see.

Pairwise output schema
----------------------
Matches the F2MD v2 output we already extract trust features from:

    receiver_pseudo,           # set to a synthetic global value (see note)
    sender_pseudo,             # = VeReMi senderPseudo
    t_prev, t_curr, dt,
    x_prev, y_prev, x_curr, y_curr, dx, dy, dist,
    speed_prev, speed_curr, dv,
    acc_prev,   acc_curr,   dacc,    jerk,
    heading_prev, heading_curr, dtheta, heading_rate,
    rate_msgs_per_s,
    pos_conf_x_curr, pos_conf_y_curr,
    spd_conf_x_curr, spd_conf_y_curr,
    acc_conf_x_curr, acc_conf_y_curr,
    head_conf_x_curr, head_conf_y_curr,
    label,                     # 0 if class == 0 (legitimate), 1 otherwise
    attack_id,                 # the original class number
    mb_version                 # 'veremi_extended'

The confidence columns are not provided by VeReMi (which models perfect-radio
reception with no GPS jitter on the senders' reported values). We fill them
with zeros and clearly disclose this in the paper's methodology section.

receiver_pseudo note
--------------------
VeReMi raw CSVs do not encode per-receiver views - they are receiver-agnostic
BSM logs from the simulator's perspective. For trust-feature computation we
only need sender ordering and timestamps, so receiver_pseudo is set to a
constant. If you later use VeReMi's per-receiver JSON logs, this loader can
be extended to populate the real receiver_pseudo per file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VeReMiLoaderConfig:
    """Parameters for converting raw VeReMi to pairwise features."""

    # Minimum number of BSMs from a sender required to form pairs.
    # Senders with fewer BSMs are dropped (cannot compute deltas).
    min_bsms_per_sender: int = 2

    # Maximum allowed gap between consecutive BSMs from the same sender
    # (seconds). Pairs spanning longer gaps are dropped to keep deltas
    # physically meaningful.
    max_pair_dt_s: float = 5.0

    # Constant placeholder for receiver_pseudo when VeReMi raw logs lack it.
    receiver_pseudo_value: int = -1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _magnitude_xy(x: pd.Series, y: pd.Series) -> pd.Series:
    """Euclidean magnitude of a 2D vector series (z dropped - planar VANET)."""
    return np.sqrt(x.astype(float) ** 2 + y.astype(float) ** 2)


def _heading_angle_xy(x: pd.Series, y: pd.Series) -> pd.Series:
    """Heading angle in radians from a 2D heading vector series."""
    return np.arctan2(y.astype(float), x.astype(float))


def _wrap_angle(diff: pd.Series) -> pd.Series:
    """Wrap angle difference to (-pi, pi]."""
    return ((diff + np.pi) % (2 * np.pi)) - np.pi


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def veremi_csv_to_pairwise(
    veremi_df: pd.DataFrame,
    config: VeReMiLoaderConfig | None = None,
) -> pd.DataFrame:
    """Convert a raw VeReMi DataFrame to per-sender pairwise features.

    Parameters
    ----------
    veremi_df : pd.DataFrame
        Raw VeReMi BSM rows. Must contain at least sendTime, senderPseudo,
        class, posx, posy, spdx, spdy, aclx, acly, hedx, hedy.
    config : VeReMiLoaderConfig or None

    Returns
    -------
    pd.DataFrame in pairwise format (see module docstring schema).
    """
    cfg = config or VeReMiLoaderConfig()

    required = [
        "sendTime", "senderPseudo", "class",
        "posx", "posy", "spdx", "spdy",
        "aclx", "acly", "hedx", "hedy",
    ]
    missing = [c for c in required if c not in veremi_df.columns]
    if missing:
        raise ValueError(f"Missing required VeReMi columns: {missing}")

    df = veremi_df[required].copy()

    # Derive scalar speed, acceleration, heading angle from XY components.
    df["speed"] = _magnitude_xy(df["spdx"], df["spdy"])
    df["acc"]   = _magnitude_xy(df["aclx"], df["acly"])
    df["theta"] = _heading_angle_xy(df["hedx"], df["hedy"])

    # Sort and group by sender to build sequential pairs.
    df = df.sort_values(["senderPseudo", "sendTime"]).reset_index(drop=True)

    pairs: list[dict] = []
    for sender, grp in df.groupby("senderPseudo", sort=False):
        if len(grp) < cfg.min_bsms_per_sender:
            continue
        grp = grp.reset_index(drop=True)
        # Build prev/curr arrays for vectorized pairing
        t_prev   = grp["sendTime"].iloc[:-1].to_numpy()
        t_curr   = grp["sendTime"].iloc[1:].to_numpy()
        dt_arr   = t_curr - t_prev

        # Filter physically meaningful pairs only
        ok = (dt_arr > 0) & (dt_arr <= cfg.max_pair_dt_s)
        if not ok.any():
            continue

        x_prev = grp["posx"].iloc[:-1].to_numpy()[ok]
        y_prev = grp["posy"].iloc[:-1].to_numpy()[ok]
        x_curr = grp["posx"].iloc[1:].to_numpy()[ok]
        y_curr = grp["posy"].iloc[1:].to_numpy()[ok]
        sp_prev = grp["speed"].iloc[:-1].to_numpy()[ok]
        sp_curr = grp["speed"].iloc[1:].to_numpy()[ok]
        ac_prev = grp["acc"].iloc[:-1].to_numpy()[ok]
        ac_curr = grp["acc"].iloc[1:].to_numpy()[ok]
        th_prev = grp["theta"].iloc[:-1].to_numpy()[ok]
        th_curr = grp["theta"].iloc[1:].to_numpy()[ok]
        cls_curr = grp["class"].iloc[1:].to_numpy()[ok]

        t_prev = t_prev[ok]; t_curr = t_curr[ok]; dt_arr = dt_arr[ok]

        dx = x_curr - x_prev
        dy = y_curr - y_prev
        dist = np.sqrt(dx * dx + dy * dy)
        dv = sp_curr - sp_prev
        dacc = ac_curr - ac_prev
        # Jerk = da/dt
        jerk = np.divide(dacc, dt_arr, out=np.zeros_like(dacc), where=dt_arr > 0)
        dtheta = _wrap_angle(pd.Series(th_curr - th_prev)).to_numpy()
        heading_rate = np.divide(
            dtheta, dt_arr, out=np.zeros_like(dtheta), where=dt_arr > 0
        )

        # Per-sender BSM rate (messages per second across the sender's
        # observed window). Used as a coarse activity indicator.
        total_time = float(grp["sendTime"].iloc[-1] - grp["sendTime"].iloc[0])
        rate = (len(grp) / total_time) if total_time > 0 else 0.0

        for i in range(len(t_curr)):
            pairs.append({
                "receiver_pseudo": cfg.receiver_pseudo_value,
                "sender_pseudo": int(sender),
                "t_prev": float(t_prev[i]),
                "t_curr": float(t_curr[i]),
                "dt":     float(dt_arr[i]),
                "x_prev": float(x_prev[i]),
                "y_prev": float(y_prev[i]),
                "x_curr": float(x_curr[i]),
                "y_curr": float(y_curr[i]),
                "dx":     float(dx[i]),
                "dy":     float(dy[i]),
                "dist":   float(dist[i]),
                "speed_prev": float(sp_prev[i]),
                "speed_curr": float(sp_curr[i]),
                "dv":          float(dv[i]),
                "acc_prev":  float(ac_prev[i]),
                "acc_curr":  float(ac_curr[i]),
                "dacc":      float(dacc[i]),
                "jerk":      float(jerk[i]),
                "heading_prev": float(th_prev[i]),
                "heading_curr": float(th_curr[i]),
                "dtheta":       float(dtheta[i]),
                "heading_rate": float(heading_rate[i]),
                "rate_msgs_per_s": float(rate),
                # Confidence placeholders - VeReMi does not provide these.
                "pos_conf_x_curr": 0.0,
                "pos_conf_y_curr": 0.0,
                "spd_conf_x_curr": 0.0,
                "spd_conf_y_curr": 0.0,
                "acc_conf_x_curr": 0.0,
                "acc_conf_y_curr": 0.0,
                "head_conf_x_curr": 0.0,
                "head_conf_y_curr": 0.0,
                # Labels - VeReMi class==0 is legitimate, anything else is an attack
                "label": 0 if int(cls_curr[i]) == 0 else 1,
                "attack_id": int(cls_curr[i]),
                "mb_version": "veremi_extended",
            })

    if not pairs:
        return pd.DataFrame()
    return pd.DataFrame(pairs)


def load_veremi_csv(
    path: str | Path,
    config: VeReMiLoaderConfig | None = None,
) -> pd.DataFrame:
    """Load a VeReMi Extended CSV and return pairwise features."""
    raw = pd.read_csv(path)
    return veremi_csv_to_pairwise(raw, config)


def load_veremi_csvs(
    paths: Iterable[str | Path],
    config: VeReMiLoaderConfig | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load and concatenate multiple VeReMi Extended CSV files."""
    parts: list[pd.DataFrame] = []
    for p in paths:
        sub = load_veremi_csv(p, config)
        if verbose:
            print(f"  Loaded {p}: {len(sub)} pairs")
        if not sub.empty:
            parts.append(sub)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert VeReMi Extended CSV to pairwise feature format."
    )
    parser.add_argument(
        "input_csv",
        help="Path to VeReMi Extended raw CSV (one or more files).",
        nargs="+",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path (pairwise features).",
    )
    args = parser.parse_args()

    df = load_veremi_csvs(args.input_csv)
    if df.empty:
        print("No usable pairs produced - check input files.")
    else:
        df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} pairwise rows to {args.output}")
