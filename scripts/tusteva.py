#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trust_eval.py — TAS-VANET (v4)
Evaluates the trust layer (Section 5.2 "Trust dynamics") from detector
outputs, producing every number and the figure that section needs:

  * mean/median time-to-isolation for attacker senders
    (messages until TT < tau_ev)
  * % of attackers never isolated within their stream
  * benign false-eviction rate (% benign senders with TT < tau_ev ever)
  * benign privilege-suspension rate (TT < tau_ch ever)
  * fig_trust_dynamics.png/.pdf — TT trajectories

INPUT CSV (one row per received message, held-out stream):
    sender_id, timestamp, y_true, y_pred_prob
  y_true       : 1 = message from attacker sender, 0 = benign
  y_pred_prob  : detector malicious probability (hat-y in the paper)

MODEL (Eqs. 8–10 of the manuscript):
    s_t  = 1 - y_pred_prob
    DT_t = lam*s_t + (1-lam)*DT_{t-1}                      (Eq. 8)
    TT_t = Wk*DT_t + Wa*AT_t + Wb*TT_{t-1}                 (Eq. 10)
  AT (indirect trust) requires multiple observers; in this offline
  replay there is a single observer, so AT is held at the neutral 0.5
  unless you supply --at-col with per-message recommendations. State
  whichever choice you use in the manuscript.

USAGE (weights are REQUIRED on purpose — they are manuscript
placeholders and must come from you, not from a script default):
  python3 trust_eval.py preds.csv --wk 0.5 --wa 0.2 --wb 0.3 \
      --lam 0.3 --tau-ch 0.5 --tau-ev 0.3
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="prediction CSV (see header docstring)")
    p.add_argument("--wk", type=float, required=True)
    p.add_argument("--wa", type=float, required=True)
    p.add_argument("--wb", type=float, required=True)
    p.add_argument("--lam", type=float, required=True,
                   help="EWMA factor lambda_tr of Eq. 8")
    p.add_argument("--tau-ch", type=float, default=0.5)
    p.add_argument("--tau-ev", type=float, default=0.3)
    p.add_argument("--tt0", type=float, default=0.5,
                   help="initial trust (paper: neutral 0.5)")
    p.add_argument("--at-col", default=None,
                   help="optional CSV column with indirect-trust values")
    p.add_argument("--max-traj", type=int, default=30,
                   help="trajectories per class to draw")
    p.add_argument("--out", default="fig_trust_dynamics")
    return p.parse_args()


def simulate(df, a):
    """Return per-sender dict with TT trajectory and metrics."""
    res = {}
    for sid, g in df.sort_values("timestamp").groupby("sender_id"):
        s = 1.0 - g["y_pred_prob"].to_numpy(float)
        at = (g[a.at_col].to_numpy(float) if a.at_col
              else np.full(len(s), 0.5))
        dt = np.empty(len(s))
        tt = np.empty(len(s))
        prev_dt, prev_tt = a.tt0, a.tt0
        for i in range(len(s)):
            prev_dt = a.lam * s[i] + (1 - a.lam) * prev_dt
            prev_tt = a.wk * prev_dt + a.wa * at[i] + a.wb * prev_tt
            dt[i], tt[i] = prev_dt, prev_tt
        below_ev = np.nonzero(tt < a.tau_ev)[0]
        below_ch = np.nonzero(tt < a.tau_ch)[0]
        res[sid] = dict(
            attacker=int(round(g["y_true"].mean())) == 1,
            n=len(s), tt=tt,
            t_iso=(int(below_ev[0]) + 1) if len(below_ev) else None,
            ever_ev=len(below_ev) > 0,
            ever_ch=len(below_ch) > 0,
        )
    return res


def main():
    a = parse_args()
    if abs(a.wk + a.wa + a.wb - 1.0) > 1e-6:
        sys.exit("Wk+Wa+Wb must equal 1 (Eq. 10).")
    df = pd.read_csv(a.csv)
    need = {"sender_id", "timestamp", "y_true", "y_pred_prob"}
    if not need.issubset(df.columns):
        sys.exit(f"CSV must contain columns {sorted(need)}")

    res = simulate(df, a)
    atk = [r for r in res.values() if r["attacker"]]
    ben = [r for r in res.values() if not r["attacker"]]
    if not atk or not ben:
        sys.exit("Need both attacker and benign senders in the stream.")

    iso = [r["t_iso"] for r in atk if r["t_iso"] is not None]
    pct_never = 100.0 * (len(atk) - len(iso)) / len(atk)
    fe = 100.0 * sum(r["ever_ev"] for r in ben) / len(ben)
    susp = 100.0 * sum(r["ever_ch"] for r in ben) / len(ben)

    print(f"senders: {len(atk)} attackers / {len(ben)} benign")
    if iso:
        print(f"time-to-isolation  mean={np.mean(iso):.1f}  "
              f"median={np.median(iso):.0f} messages")
    print(f"attackers never isolated: {pct_never:.1f}%")
    print(f"benign false-eviction rate (TT<{a.tau_ev}): {fe:.2f}%")
    print(f"benign privilege-suspension (TT<{a.tau_ch}): {susp:.2f}%")
    print("\nPaste into Section 'Trust dynamics':")
    print(f"  isolated after a mean of {np.mean(iso):.1f} messages "
          f"(median {np.median(iso):.0f}); "
          f"{fe:.2f}% benign false-eviction."
          if iso else "  (no attacker crossed tau_ev - revisit weights)")

    # ---------------- figure ----------------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    rng = np.random.default_rng(0)
    for grp, color, lab in ((ben, "#5a5a5a", "benign"),
                            (atk, "#C1272D", "attacker")):
        pick = rng.permutation(len(grp))[:a.max_traj]
        first = True
        for i in pick:
            tt = grp[i]["tt"]
            ax.plot(np.arange(1, len(tt) + 1), tt, color=color,
                    alpha=0.45, lw=1.1,
                    label=lab if first else None)
            first = False
    ax.axhline(a.tau_ch, color="#0072B2", ls="--", lw=1.4)
    ax.axhline(a.tau_ev, color="#C1272D", ls="--", lw=1.4)
    ax.text(0.995, a.tau_ch + 0.012, r"$\tau_{CH}$", ha="right",
            transform=ax.get_yaxis_transform(), color="#0072B2")
    ax.text(0.995, a.tau_ev + 0.012, r"$\tau_{ev}$", ha="right",
            transform=ax.get_yaxis_transform(), color="#C1272D")
    ax.set_xlabel("Messages received from sender")
    ax.set_ylabel(r"Total trust $TT_j$")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, ls=":")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{a.out}.png", dpi=300)
    fig.savefig(f"{a.out}.pdf")
    print(f"\nwrote {a.out}.png / {a.out}.pdf")


if __name__ == "__main__":
    main()