"""
07 — Novel-Attack Generalization Test (Leave-One-Attack-Type-Out).

Motivation
----------
scripts/05_train_full.py compares TAS-VANET against RF/SVM/MLP on a
standard random train/test split, where every attack type present in the
test set was ALSO present in training. That setup favors purely supervised
classifiers (RF/MLP), which only need to pattern-match attack signatures
they have already seen. It does NOT test the scenario that motivates
anomaly-detection-style methods in the first place: a genuinely NEW
attack variant appearing in the field with zero labeled examples.

This script runs a Leave-One-Attack-Type-Out (LOAO) experiment: for each
attack type A, every method is trained on legitimate traffic + all OTHER
attack types (A is completely excluded from training), then evaluated on
its ability to detect attack A alone. Four methods are compared:

  tas_vanet_oneclass   SAE trained ONLY on legitimate traffic (no attack
                       labels at all, ever). Detects via reconstruction
                       error > threshold. This is the classic anomaly-
                       detection formulation and the one method here that
                       is architecturally blind to attack signatures.
  tas_vanet_supervised SAE + WOA-tuned architecture with the fine-tuned
                       classifier head (see core/sae_model.py), trained
                       on legit + other-attack labels (like scripts/05).
  rf, mlp              Same scikit-learn baselines as scripts/05, trained
                       on legit + other-attack labels.

If tas_vanet_oneclass degrades less than the supervised methods when
facing a truly unseen attack type, that is real evidence for an anomaly-
detection advantage. If it does not, that claim is not supported by this
codebase and should not be made in the paper.

Normalization
-------------
Two separate min-max normalizations are fit, each using only data
available to that method at train time (no leakage):
  - "legit" normalization: fit once on the legitimate-only training pool,
    used for tas_vanet_oneclass (which never needs anything else).
  - "fold" normalization: fit per left-out-attack fold on that fold's
    training pool (legit + other attacks), used for the three supervised
    methods.

Usage
-----
    python scripts/07_novel_attack_generalization.py
    python scripts/07_novel_attack_generalization.py --quick   # fast smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import SAEConfig, StackedAutoencoder, train_sae, HYBRID_FEATURES

METHODS = ["tas_vanet_oneclass", "tas_vanet_supervised", "rf", "mlp"]
METHOD_LABELS = {
    "tas_vanet_oneclass":   "TAS-VANET (one-class, no attack labels)",
    "tas_vanet_supervised": "TAS-VANET (SAE+WOA, supervised head)",
    "rf":                   "Random Forest",
    "mlp":                  "MLP",
}
METRICS = ["precision", "recall", "f1", "accuracy", "auc_roc"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Leave-one-attack-type-out generalization test.")
    p.add_argument("--train_csv", default="data/processed/veremi_train_processed.csv")
    p.add_argument("--test_csv",  default="data/processed/veremi_test_processed.csv")
    p.add_argument("--out_dir",   default="results")
    p.add_argument("--best_hp_json",
                   default="results/models/final_tas_vanet/sae_config.json",
                   help="Best WOA hyperparameters from scripts/05 (architecture reused, "
                        "not re-searched — see module docstring).")
    p.add_argument("--min_attack_samples", type=int, default=100,
                   help="Skip attack types with fewer than this many samples "
                        "(too few for a reliable held-out evaluation).")
    p.add_argument("--legit_eval_frac", type=float, default=0.3,
                   help="Fraction of legitimate rows reserved for evaluation "
                        "(shared across all attack folds).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--mlp_max_iter", type=int, default=200)
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke test: 3 attacks, few epochs.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_pool(args):
    train_df = pd.read_csv(PROJECT_ROOT / args.train_csv)
    test_df  = pd.read_csv(PROJECT_ROOT / args.test_csv)
    df = pd.concat([train_df, test_df], ignore_index=True)

    available = [c for c in HYBRID_FEATURES if c in df.columns]
    stds = df[available].std()
    feature_cols = [c for c in available if stds[c] > 1e-9]

    X = df[feature_cols].to_numpy(dtype=np.float32)
    attack_id = df["attack_id"].to_numpy(dtype=np.int32)
    return X, attack_id, feature_cols


def fit_normalize(X):
    lo, hi = X.min(axis=0), X.max(axis=0)
    rng = hi - lo
    rng[rng < 1e-12] = 1.0
    return {"lo": lo, "hi": hi, "rng": rng}


def apply_normalize(X, params):
    return np.clip((X - params["lo"]) / params["rng"], 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TAS-VANET one-class (trained ONCE, reused across every attack fold)
# ---------------------------------------------------------------------------

def train_oneclass(X_legit_train_norm: np.ndarray, hidden_dims, epochs: int, seed: int = 0):
    cfg = SAEConfig(
        input_dim=X_legit_train_norm.shape[1],
        hidden_dims=hidden_dims,
        learning_rate=1e-3,
        sparsity_target=0.05,
        sparsity_weight=3.0,
        epochs=epochs,
        batch_size=256,
        seed=seed,
        use_classifier_head=False,
    )
    torch.manual_seed(seed)
    model = StackedAutoencoder(cfg)
    X_t = torch.tensor(X_legit_train_norm, dtype=torch.float32)
    train_sae(model, X_t, verbose=False)

    model.eval()
    with torch.no_grad():
        X_recon, _ = model(X_t)
        err = ((X_recon - X_t) ** 2).mean(dim=1).numpy()
    tau = float(np.median(err) + 2.0 * err.std())
    return model, tau


def eval_oneclass(model, tau, X_eval_norm: np.ndarray):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_eval_norm, dtype=torch.float32)
        X_recon, _ = model(X_t)
        err = ((X_recon - X_t) ** 2).mean(dim=1).numpy()
    y_pred = (err > tau).astype(int)
    return y_pred, err


# ---------------------------------------------------------------------------
# TAS-VANET supervised (fixed WOA-tuned architecture, no re-search — see
# module docstring: in a real novel-attack scenario there is no labeled
# validation split for A to search hyperparameters against anyway)
# ---------------------------------------------------------------------------

def train_eval_supervised(X_tr, y_tr, X_ev, hidden_dims, lr, sparsity_rho,
                          sparsity_beta, epochs, seed=0):
    cfg = SAEConfig(
        input_dim=X_tr.shape[1],
        hidden_dims=hidden_dims,
        learning_rate=lr,
        sparsity_target=sparsity_rho,
        sparsity_weight=sparsity_beta,
        epochs=epochs,
        batch_size=256,
        seed=seed,
        use_classifier_head=True,
    )
    torch.manual_seed(seed)
    model = StackedAutoencoder(cfg)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    train_sae(model, X_tr_t, y_train=y_tr_t, verbose=False)

    model.eval()
    with torch.no_grad():
        X_ev_t = torch.tensor(X_ev, dtype=torch.float32)
        _, H = model(X_ev_t)
        logits = model.classify_logits(X_ev_t, H)
        probs = torch.sigmoid(logits).numpy()
    y_pred = (probs > 0.5).astype(int)
    return y_pred, probs


# ---------------------------------------------------------------------------
# Sklearn baselines
# ---------------------------------------------------------------------------

def train_eval_sklearn(method, X_tr, y_tr, X_ev, seed, mlp_max_iter):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    if method == "rf":
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    elif method == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=mlp_max_iter,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=15, tol=1e-4, random_state=seed,
        )
    else:
        raise ValueError(method)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_ev)
    y_score = clf.predict_proba(X_ev)[:, 1]
    return y_pred, y_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    if args.quick:
        args.epochs = 20
        args.mlp_max_iter = 60
        print("*** QUICK MODE — results are for smoke-testing only ***")

    print("=" * 70)
    print("Novel-Attack Generalization Test (Leave-One-Attack-Type-Out)")
    print("=" * 70)

    X, attack_id, feature_cols = load_pool(args)
    print(f"Pool: {X.shape[0]:,} rows, {X.shape[1]} features")

    counts = pd.Series(attack_id).value_counts()
    attack_ids = sorted(
        aid for aid in counts.index
        if aid != 0 and counts[aid] >= args.min_attack_samples
    )
    if args.quick:
        attack_ids = attack_ids[:3]
    print(f"Attack types tested: {attack_ids}")

    # Best architecture from scripts/05 (reused, not re-searched)
    hp_path = PROJECT_ROOT / args.best_hp_json
    if hp_path.exists():
        with open(hp_path) as f:
            best_hp = json.load(f)
        hidden_dims = best_hp["hidden_dims"]
        lr = best_hp["learning_rate"]
        rho = best_hp["sparsity_target"]
        beta = best_hp["sparsity_weight"]
        print(f"Loaded TAS-VANET architecture from {hp_path}: "
              f"hidden_dims={hidden_dims} lr={lr:.5f} rho={rho:.3f} beta={beta:.3f}")
    else:
        hidden_dims, lr, rho, beta = [32, 8], 1e-3, 0.1, 1.0
        print(f"WARNING: {hp_path} not found — using fallback architecture "
              f"hidden_dims={hidden_dims}")

    # Fixed legit train/eval split, shared across every attack fold
    legit_idx = np.where(attack_id == 0)[0]
    rng = np.random.default_rng(0)
    rng.shuffle(legit_idx)
    n_eval = int(len(legit_idx) * args.legit_eval_frac)
    legit_eval_idx, legit_train_idx = legit_idx[:n_eval], legit_idx[n_eval:]
    print(f"Legitimate pool: {len(legit_train_idx):,} train / {len(legit_eval_idx):,} eval")

    # One-class model: trained ONCE on legit-only data (unaffected by which
    # attack is held out), reused for every fold.
    legit_norm = fit_normalize(X[legit_train_idx])
    X_legit_train_n = apply_normalize(X[legit_train_idx], legit_norm)
    X_legit_eval_n  = apply_normalize(X[legit_eval_idx],  legit_norm)
    print("\nTraining TAS-VANET (one-class) once on legitimate-only data ...")
    t0 = time.time()
    oneclass_model, oneclass_tau = train_oneclass(
        X_legit_train_n, hidden_dims, epochs=args.epochs
    )
    print(f"  done in {time.time()-t0:.0f}s  (threshold tau={oneclass_tau:.6f})")

    rows = []
    for aid in attack_ids:
        t_fold = time.time()
        mal_A_idx     = np.where(attack_id == aid)[0]
        mal_other_idx = np.where((attack_id != 0) & (attack_id != aid))[0]

        train_idx = np.concatenate([legit_train_idx, mal_other_idx])
        eval_idx  = np.concatenate([legit_eval_idx, mal_A_idx])
        y_train   = np.concatenate([
            np.zeros(len(legit_train_idx), dtype=np.int32),
            np.ones(len(mal_other_idx), dtype=np.int32),
        ])
        y_eval    = np.concatenate([
            np.zeros(len(legit_eval_idx), dtype=np.int32),
            np.ones(len(mal_A_idx), dtype=np.int32),
        ])

        fold_norm = fit_normalize(X[train_idx])
        X_tr = apply_normalize(X[train_idx], fold_norm)
        X_ev = apply_normalize(X[eval_idx],  fold_norm)

        print(f"\n[attack {aid}] held out ({len(mal_A_idx):,} samples) — "
              f"train={len(train_idx):,} eval={len(eval_idx):,}")

        # tas_vanet_oneclass: same model/threshold every fold, only the
        # eval pool (which legit-normalized attack-A samples) changes.
        X_ev_legitnorm = apply_normalize(X[eval_idx], legit_norm)
        y_pred, y_score = eval_oneclass(oneclass_model, oneclass_tau, X_ev_legitnorm)
        m = compute_metrics(y_eval, y_pred, y_score)
        rows.append({"attack_id": int(aid), "n_attack": len(mal_A_idx),
                     "method": "tas_vanet_oneclass", **m})
        print(f"  {METHOD_LABELS['tas_vanet_oneclass']:<45s} F1={m['f1']:.4f}  R={m['recall']:.4f}")

        # tas_vanet_supervised
        y_pred, y_score = train_eval_supervised(
            X_tr, y_train, X_ev, hidden_dims, lr, rho, beta, args.epochs
        )
        m = compute_metrics(y_eval, y_pred, y_score)
        rows.append({"attack_id": int(aid), "n_attack": len(mal_A_idx),
                     "method": "tas_vanet_supervised", **m})
        print(f"  {METHOD_LABELS['tas_vanet_supervised']:<45s} F1={m['f1']:.4f}  R={m['recall']:.4f}")

        # rf, mlp
        for method in ["rf", "mlp"]:
            y_pred, y_score = train_eval_sklearn(
                method, X_tr, y_train, X_ev, seed=0, mlp_max_iter=args.mlp_max_iter
            )
            m = compute_metrics(y_eval, y_pred, y_score)
            rows.append({"attack_id": int(aid), "n_attack": len(mal_A_idx),
                         "method": method, **m})
            print(f"  {METHOD_LABELS[method]:<45s} F1={m['f1']:.4f}  R={m['recall']:.4f}")

        print(f"  fold done in {time.time()-t_fold:.0f}s")

    # ---- Save + summarize ----
    out_dir = PROJECT_ROOT / args.out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(rows)
    raw_path = out_dir / "novel_attack_generalization.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"\nSaved: {raw_path}")

    summary_records = []
    for method in METHODS:
        sub = raw_df[raw_df["method"] == method]
        rec = {"method": METHOD_LABELS[method]}
        for metric in METRICS:
            vals = sub[metric].dropna()
            rec[metric] = f"{vals.mean():.4f} ± {vals.std():.4f}"
        summary_records.append(rec)
    summary_df = pd.DataFrame(summary_records)
    summary_path = out_dir / "novel_attack_generalization_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    print("\n" + "=" * 70)
    print("SUMMARY — mean ± std F1 across all held-out (unseen) attack types")
    print("=" * 70)
    for method in METHODS:
        sub = raw_df[raw_df["method"] == method]["f1"]
        print(f"  {METHOD_LABELS[method]:<45s} F1={sub.mean():.4f} ± {sub.std():.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
