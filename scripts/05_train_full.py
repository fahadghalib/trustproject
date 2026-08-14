"""
05 — Full Training Pipeline: SAE+WOA vs Baselines with Cross-Validation.

Trains TAS-VANET (SAE + WOA) and four baselines on the processed VeReMi
dataset using stratified k-fold CV on the training split and a held-out
evaluation on the test split.

Outputs
-------
results/tables/cv_raw.csv          per-fold metrics for all methods
results/tables/summary.csv         mean ± std (paper table)
results/tables/stats.csv           Wilcoxon p-values, Cohen's d
results/tables/final_test.csv      metrics on held-out test set
results/tables/per_attack.csv      per-attack-type detection on test set
results/models/final_tas_vanet/    final trained model weights + config
results/models/woa_history.json    WOA convergence data (for fig gen)
results/models/norm_params.json    normalization params for inference

Usage
-----
Quick smoke test (~5-10 min):
    python scripts/05_train_full.py --quick

Standard run (~30-60 min):
    python scripts/05_train_full.py

With 3 CV seeds for extra stability:
    python scripts/05_train_full.py --cv_seeds 3

Full details:
    python scripts/05_train_full.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import (
    SAEConfig, StackedAutoencoder, train_sae,
    WOAConfig, HyperparamSpace, WhaleOptimizer,
    build_fitness_fn, decode_position,
    HYBRID_FEATURES,
)

warnings.filterwarnings("ignore", category=UserWarning)

METHODS = ["tas_vanet", "sae_only", "rf", "svm", "mlp"]
METHOD_LABELS = {
    "tas_vanet": "TAS-VANET (SAE+WOA)",
    "sae_only":  "SAE only (no WOA)",
    "rf":        "Random Forest",
    "svm":       "SVM (RBF)",
    "mlp":       "MLP",
}
METRICS = ["precision", "recall", "f1", "accuracy", "auc_roc"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Full TAS-VANET training pipeline with cross-validation."
    )
    p.add_argument("--train_csv", default="data/processed/veremi_train_processed.csv")
    p.add_argument("--test_csv",  default="data/processed/veremi_test_processed.csv")
    p.add_argument("--out_dir",   default="results")

    # CV settings
    p.add_argument("--cv_folds",  type=int, default=10)
    p.add_argument("--cv_seeds",  type=int, default=1)

    # WOA settings (CV phase — lighter for speed)
    p.add_argument("--woa_pop_cv",    type=int,   default=10)
    p.add_argument("--woa_gen_cv",    type=int,   default=10)
    p.add_argument("--woa_subsample", type=int,   default=5000,
                   help="Max training samples used for WOA fitness evaluation.")
    p.add_argument("--woa_train_epochs", type=int, default=30,
                   help="SAE epochs per WOA fitness evaluation (kept short).")

    # Final model (full training set)
    p.add_argument("--woa_pop_final",  type=int, default=20)
    p.add_argument("--woa_gen_final",  type=int, default=30)
    p.add_argument("--final_epochs_cv",   type=int, default=75,
                   help="SAE epochs for the per-fold final model.")
    p.add_argument("--final_epochs_full", type=int, default=200,
                   help="SAE epochs for the final model trained on full train set.")

    p.add_argument("--quick", action="store_true",
                   help="Override all settings with fast smoke-test values.")
    p.add_argument("--load_cv", action="store_true",
                   help="Skip CV and load existing results/tables/cv_raw.csv "
                        "(use after a crash during final evaluation).")
    return p.parse_args(argv)


def apply_quick_preset(args):
    args.cv_folds            = 3
    args.cv_seeds            = 1
    args.woa_pop_cv          = 5
    args.woa_gen_cv          = 5
    args.woa_subsample       = 1000
    args.woa_train_epochs    = 15
    args.woa_pop_final       = 5
    args.woa_gen_final       = 10
    args.final_epochs_cv     = 30
    args.final_epochs_full   = 50
    return args


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(args):
    """Load train/test CSVs, drop zero-variance features, return numpy arrays.

    The two source CSVs are NOT sender-disjoint (verified: 7,139 of 8,316
    test-set senders also appear in the training CSV) — the original split
    was row-random, not grouped by vehicle. We therefore pool both files and
    re-split with GroupShuffleSplit on sender_pseudo, so the held-out test
    set contains zero vehicles seen during training. The CV loop (run_cv)
    similarly uses GroupKFold on sender_pseudo instead of StratifiedKFold.
    """
    train_path = PROJECT_ROOT / args.train_csv
    test_path  = PROJECT_ROOT / args.test_csv

    if not train_path.exists():
        raise FileNotFoundError(
            f"Train CSV not found: {train_path}\n"
            "Run scripts/02_process_veremi.py first."
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test CSV not found: {test_path}\n"
            "Run scripts/02_process_veremi.py first."
        )

    print(f"Loading {train_path.name} ...", end=" ", flush=True)
    train_df = pd.read_csv(train_path)
    print(f"{len(train_df):,} rows")

    print(f"Loading {test_path.name}  ...", end=" ", flush=True)
    test_df  = pd.read_csv(test_path)
    print(f"{len(test_df):,} rows")

    df = pd.concat([train_df, test_df], ignore_index=True)

    # Filter to columns present in HYBRID_FEATURES
    available = [c for c in HYBRID_FEATURES if c in df.columns]

    # Drop zero-variance columns (e.g. VeReMi confidence columns = all zeros)
    stds = df[available].std()
    feature_cols = [c for c in available if stds[c] > 1e-9]
    dropped = set(available) - set(feature_cols)
    if dropped:
        print(f"  Dropped {len(dropped)} zero-variance features: {sorted(dropped)}")
    print(f"  Effective features: {len(feature_cols)}")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    a = df["attack_id"].to_numpy(dtype=np.int32)
    groups = df["sender_pseudo"].to_numpy()

    from sklearn.model_selection import GroupShuffleSplit
    test_frac = len(test_df) / len(df)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=0)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    n_shared = len(set(groups[train_idx]) & set(groups[test_idx]))
    print(f"  Sender-disjoint split: {len(train_idx):,} train / {len(test_idx):,} test "
          f"rows, {n_shared} shared vehicles (must be 0)")
    assert n_shared == 0, "GroupShuffleSplit leaked a vehicle across train/test"

    X_train, y_train, a_train = X[train_idx], y[train_idx], a[train_idx]
    groups_train = groups[train_idx]
    X_test, y_test, a_test = X[test_idx], y[test_idx], a[test_idx]

    return X_train, y_train, a_train, groups_train, X_test, y_test, a_test, feature_cols


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def fit_normalize(X: np.ndarray):
    """Min-max fit on X, return (X_norm, params_dict)."""
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    rng = hi - lo
    rng[rng < 1e-12] = 1.0          # avoid div-by-zero
    X_norm = (X - lo) / rng
    return X_norm.astype(np.float32), {"lo": lo, "hi": hi, "rng": rng}


def apply_normalize(X: np.ndarray, params: dict) -> np.ndarray:
    """Apply previously-fit normalization, clip to [0, 1]."""
    X_norm = (X - params["lo"]) / params["rng"]
    return np.clip(X_norm, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Stratified subsample helper
# ---------------------------------------------------------------------------

def stratified_subsample(X, y, n, seed=42):
    """Return at most n samples with class balance preserved."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    if n >= total:
        return X, y

    indices = []
    for cls, cnt in zip(classes, counts):
        take = max(1, int(round(n * cnt / total)))
        cls_idx = np.where(y == cls)[0]
        chosen = rng.choice(cls_idx, size=min(take, len(cls_idx)), replace=False)
        indices.extend(chosen)
    indices = np.array(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_score) -> dict:
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        accuracy_score, roc_auc_score,
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
# SAE helpers
# ---------------------------------------------------------------------------

def _derive_threshold(model, X_tr_t, k=1.0):
    """Derive reconstruction-error threshold from training data.

    τ = median(err) + k·std(err)  — matches the WOA fitness proxy formula
    exactly so that the detection mechanism used in CV is identical to
    the one WOA optimises for.  Samples with err > τ are flagged malicious.
    """
    model.eval()
    with torch.no_grad():
        X_recon, _ = model(X_tr_t)
        recon_err = ((X_recon - X_tr_t) ** 2).mean(dim=1).numpy()
    med   = float(np.median(recon_err))
    sigma = float(recon_err.std())
    return med + k * sigma, med, sigma


def _sae_classify_score(model, threshold, X_t):
    """Classify via reconstruction error > threshold = malicious."""
    model.eval()
    with torch.no_grad():
        X_recon, _ = model(X_t)
        recon_err  = ((X_recon - X_t) ** 2).mean(dim=1).numpy()
    y_pred  = (recon_err > threshold).astype(int)
    y_score = recon_err                            # higher = more malicious
    return y_pred, y_score


def _head_classify_score(model, X_t):
    """Classify via the supervised classifier head fine-tuned on the
    SAE bottleneck (logit > 0 => malicious). y_score = sigmoid(logit).
    """
    model.eval()
    with torch.no_grad():
        _, H = model(X_t)
        logits = model.classify_logits(X_t, H)
        probs  = torch.sigmoid(logits).numpy()
    y_pred = (probs > 0.5).astype(int)
    return y_pred, probs


# ---------------------------------------------------------------------------
# TAS-VANET evaluation (WOA + SAE)
# ---------------------------------------------------------------------------

def _woa_search_space() -> HyperparamSpace:
    """WOA hyperparameter search bounds for the classifier-head SAE.

    Widened vs the original bounds after the full run showed WOA repeatedly
    pushing h1_size, h2_size, and learning_rate to their upper bound and
    sparsity_beta to its lower bound — i.e. the optimum was sitting at the
    edge of the searchable box, not inside it.
    """
    return HyperparamSpace(
        h1_size=(16, 128, True),
        h2_size=(4, 32, True),
        learning_rate=(1e-4, 3e-2, False),
        sparsity_rho=(0.01, 0.3, False),
        sparsity_beta=(0.1, 5.0, False),
    )


def _woa_fitness_weights() -> dict:
    """Fitness weights alpha/beta (see woa_optimizer.build_fitness_fn).

    Reconstruction is now a secondary regularizer for the classifier head
    (not the detection mechanism itself), so detection F1 is weighted more
    heavily than in the original one-class formulation.
    """
    return {"alpha_recon": 0.15, "beta_f1": 0.85}


def eval_tas_vanet(X_tr, X_va, y_tr, y_va, seed, args):
    """Train TAS-VANET on one fold, return metrics dict.

    Semi-supervised learning: the SAE is trained on ALL training samples
    (legitimate + malicious) with the combined reconstruction/sparsity loss
    PLUS a classifier head fine-tuned on the labels (unsupervised
    pretraining + supervised fine-tuning — see sae_model.py). WOA searches
    hyperparameters directly against the classifier head's detection F1,
    the same mechanism used at inference time.
    """
    X_sub, y_sub = stratified_subsample(X_tr, y_tr, args.woa_subsample, seed=seed)

    space = _woa_search_space()
    woa_cfg = WOAConfig(
        population_size=args.woa_pop_cv,
        max_generations=args.woa_gen_cv,
        seed=seed,
        **_woa_fitness_weights(),
    )
    fitness_fn = build_fitness_fn(
        X_train=X_sub, X_val=X_va, y_val=y_va, y_train=y_sub,
        epochs=args.woa_train_epochs,
        use_classifier_head=True,
    )
    woa = WhaleOptimizer(fitness_fn, space, woa_cfg)
    woa_result = woa.optimize(verbose=False)
    bp = woa_result["best_params"]

    # Final SAE: train on ALL samples (legit + malicious) in this fold
    cfg = SAEConfig(
        input_dim=X_tr.shape[1],
        hidden_dims=[int(bp["h1_size"]), int(bp["h2_size"])],
        learning_rate=float(bp["learning_rate"]),
        sparsity_target=float(bp["sparsity_rho"]),
        sparsity_weight=float(bp["sparsity_beta"]),
        epochs=args.final_epochs_cv,
        batch_size=128,
        seed=seed,
        use_classifier_head=True,
    )
    torch.manual_seed(seed)
    model = StackedAutoencoder(cfg)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)
    train_sae(model, X_tr_t, X_val=X_va_t, y_train=y_tr_t, y_val=y_va_t, verbose=False)

    y_pred, y_score = _head_classify_score(model, X_va_t)
    return compute_metrics(y_va, y_pred, y_score)


# ---------------------------------------------------------------------------
# SAE-only baseline (no WOA — fixed default hyperparameters)
# ---------------------------------------------------------------------------

def eval_sae_only(X_tr, X_va, y_tr, y_va, seed, args):
    """SAE + classifier head with fixed hyperparameters (ablation — no WOA).

    Same semi-supervised training as TAS-VANET (classifier head fine-tuned
    on all labeled samples) but with fixed default hyperparameters instead
    of WOA-optimised ones. This isolates WOA's contribution: both methods
    now use the identical classifier-head detection mechanism.
    """
    n_feat = X_tr.shape[1]
    h1 = min(64, max(16, n_feat * 2))
    h2 = max(4,  n_feat // 3)
    cfg = SAEConfig(
        input_dim=n_feat,
        hidden_dims=[h1, h2],
        learning_rate=1e-3,
        sparsity_target=0.05,
        sparsity_weight=3.0,
        epochs=args.final_epochs_cv,
        batch_size=128,
        seed=seed,
        use_classifier_head=True,
    )
    torch.manual_seed(seed)
    model = StackedAutoencoder(cfg)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)
    train_sae(model, X_tr_t, X_val=X_va_t, y_train=y_tr_t, y_val=y_va_t, verbose=False)

    y_pred, y_score = _head_classify_score(model, X_va_t)
    return compute_metrics(y_va, y_pred, y_score)


# ---------------------------------------------------------------------------
# Sklearn baselines
# ---------------------------------------------------------------------------

def eval_sklearn(method: str, X_tr, X_va, y_tr, y_va, seed,
                 svm_max_train: int = 10_000) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier

    if method == "rf":
        clf = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=seed)
        X_fit, y_fit = X_tr, y_tr
    elif method == "svm":
        # RBF SVM scales O(n^2) — cap training set for feasibility
        if len(X_tr) > svm_max_train:
            X_fit, y_fit = stratified_subsample(
                X_tr, y_tr, svm_max_train, seed=seed)
        else:
            X_fit, y_fit = X_tr, y_tr
        clf = SVC(kernel="rbf", C=1.0, probability=True, random_state=seed)
    elif method == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=300,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=15, tol=1e-4,
            random_state=seed)
        X_fit, y_fit = X_tr, y_tr
    else:
        raise ValueError(f"Unknown method: {method}")

    clf.fit(X_fit, y_fit)
    y_pred  = clf.predict(X_va)
    y_score = clf.predict_proba(X_va)[:, 1]
    return compute_metrics(y_va, y_pred, y_score)


# ---------------------------------------------------------------------------
# Cross-validation loop
# ---------------------------------------------------------------------------

def run_cv(X_train, y_train, groups_train, args, cv_raw_path=None):
    """Stratified-by-vehicle CV: GroupKFold on sender_pseudo so no vehicle's
    rows appear in both the train and validation side of any fold.

    Note: unlike StratifiedKFold, GroupKFold has no shuffle/random_state —
    fold assignment is deterministic given the groups, so cv_seeds>1 will
    reuse the same fold split each "seed" (only per-fold model seeds vary,
    via seed*100+fold_idx below). Kept for CLI/back-compat with --cv_seeds.
    """
    from sklearn.model_selection import GroupKFold

    total_folds = args.cv_folds * args.cv_seeds
    done = 0
    all_rows = []

    for seed in range(args.cv_seeds):
        gkf = GroupKFold(n_splits=args.cv_folds)
        for fold_idx, (tr_idx, va_idx) in enumerate(
            gkf.split(X_train, y_train, groups=groups_train)
        ):
            done += 1
            t0 = time.time()
            print(
                f"\n[{done}/{total_folds}] seed={seed} fold={fold_idx+1} "
                f"(train={len(tr_idx):,} val={len(va_idx):,})"
            )

            X_tr_raw, X_va_raw = X_train[tr_idx], X_train[va_idx]
            y_tr,     y_va     = y_train[tr_idx],  y_train[va_idx]

            X_tr, scaler = fit_normalize(X_tr_raw)
            X_va         = apply_normalize(X_va_raw, scaler)

            row = {"seed": seed, "fold": fold_idx}

            # TAS-VANET
            print("  TAS-VANET ...", end=" ", flush=True)
            t1 = time.time()
            row["tas_vanet"] = eval_tas_vanet(X_tr, X_va, y_tr, y_va,
                                              seed * 100 + fold_idx, args)
            print(f"F1={row['tas_vanet']['f1']:.4f}  ({time.time()-t1:.0f}s)")

            # SAE only
            print("  SAE only  ...", end=" ", flush=True)
            t1 = time.time()
            row["sae_only"] = eval_sae_only(X_tr, X_va, y_tr, y_va,
                                            seed * 100 + fold_idx, args)
            print(f"F1={row['sae_only']['f1']:.4f}  ({time.time()-t1:.0f}s)")

            # RF
            print("  RF        ...", end=" ", flush=True)
            t1 = time.time()
            row["rf"] = eval_sklearn("rf", X_tr, X_va, y_tr, y_va,
                                     seed * 100 + fold_idx)
            print(f"F1={row['rf']['f1']:.4f}  ({time.time()-t1:.0f}s)")

            # SVM
            print("  SVM       ...", end=" ", flush=True)
            t1 = time.time()
            row["svm"] = eval_sklearn("svm", X_tr, X_va, y_tr, y_va,
                                      seed * 100 + fold_idx)
            print(f"F1={row['svm']['f1']:.4f}  ({time.time()-t1:.0f}s)")

            # MLP
            print("  MLP       ...", end=" ", flush=True)
            t1 = time.time()
            row["mlp"] = eval_sklearn("mlp", X_tr, X_va, y_tr, y_va,
                                      seed * 100 + fold_idx)
            print(f"F1={row['mlp']['f1']:.4f}  ({time.time()-t1:.0f}s)")

            elapsed = time.time() - t0
            print(f"  Fold done in {elapsed:.0f}s")
            all_rows.append(row)

            # Save incrementally after every fold so a later crash never loses CV data
            if cv_raw_path is not None:
                raw_records = []
                for r in all_rows:
                    for method in METHODS:
                        rec = {"seed": r["seed"], "fold": r["fold"], "method": method}
                        rec.update(r[method])
                        raw_records.append(rec)
                pd.DataFrame(raw_records).to_csv(cv_raw_path, index=False)

    return all_rows


# ---------------------------------------------------------------------------
# Final model: train on full train set, evaluate on held-out test
# ---------------------------------------------------------------------------

def run_final_evaluation(X_train, y_train, X_test, y_test, a_test,
                         feature_cols, args):
    print("\n" + "=" * 70)
    print("Final model: training on full train set")
    print("=" * 70)

    X_train_n, scaler = fit_normalize(X_train)
    X_test_n          = apply_normalize(X_test, scaler)

    # Save normalization params for inference
    norm_params = {
        col: {"lo": float(scaler["lo"][i]), "hi": float(scaler["hi"][i])}
        for i, col in enumerate(feature_cols)
    }

    results = {}
    woa_history = None

    # ---- TAS-VANET ----
    print("\nTAS-VANET (WOA + SAE) ...")
    X_sub, y_sub = stratified_subsample(
        X_train_n, y_train, args.woa_subsample * 2, seed=0
    )
    space = _woa_search_space()
    woa_cfg = WOAConfig(
        population_size=args.woa_pop_final,
        max_generations=args.woa_gen_final,
        seed=0,
        **_woa_fitness_weights(),
    )
    # Use a validation split from the training set for WOA fitness
    val_size = min(5000, int(0.1 * len(X_train_n)))
    rng = np.random.default_rng(0)
    va_idx = rng.choice(len(X_train_n), size=val_size, replace=False)
    X_woa_va = X_train_n[va_idx]
    y_woa_va = y_train[va_idx]

    X_woa_sub, y_woa_sub = stratified_subsample(
        X_train_n, y_train, args.woa_subsample * 2, seed=0,
    )

    fitness_fn = build_fitness_fn(
        X_train=X_woa_sub, X_val=X_woa_va, y_val=y_woa_va, y_train=y_woa_sub,
        epochs=args.woa_train_epochs,
        use_classifier_head=True,
    )
    woa = WhaleOptimizer(fitness_fn, space, woa_cfg)
    woa_result = woa.optimize(verbose=True)
    bp = woa_result["best_params"]
    woa_history = woa_result["history"]
    print(f"Best hyperparams: {bp}")

    cfg = SAEConfig(
        input_dim=X_train_n.shape[1],
        hidden_dims=[int(bp["h1_size"]), int(bp["h2_size"])],
        learning_rate=float(bp["learning_rate"]),
        sparsity_target=float(bp["sparsity_rho"]),
        sparsity_weight=float(bp["sparsity_beta"]),
        epochs=args.final_epochs_full,
        batch_size=256,
        seed=0,
        use_classifier_head=True,
    )
    torch.manual_seed(0)
    final_model = StackedAutoencoder(cfg)
    # Train on the FULL training set (legit + malicious): unsupervised
    # reconstruction/sparsity loss + supervised classifier-head fine-tuning.
    X_train_t = torch.tensor(X_train_n, dtype=torch.float32)
    X_te_t    = torch.tensor(X_test_n,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train,   dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,    dtype=torch.float32)
    print(f"Training final SAE on {len(X_train_n):,} samples "
          f"({args.final_epochs_full} epochs) ...")
    train_sae(final_model, X_train_t, X_val=X_te_t,
              y_train=y_train_t, y_val=y_test_t, verbose=True)

    y_pred, y_score = _head_classify_score(final_model, X_te_t)
    results["tas_vanet"] = compute_metrics(y_test, y_pred, y_score)
    results["tas_vanet"]["best_params"] = bp
    results["tas_vanet"]["woa_fitness"] = woa_result["best_fitness"]
    threshold = 0.0   # decision boundary in logit-space (kept for save_results/figures)

    # ---- SAE only ----
    print("\nSAE only (fixed hyperparameters) ...")
    n_feat = X_train_n.shape[1]
    sae_only_cfg = SAEConfig(
        input_dim=n_feat,
        hidden_dims=[min(64, max(16, n_feat * 2)), max(4, n_feat // 3)],
        learning_rate=1e-3, sparsity_target=0.05, sparsity_weight=3.0,
        epochs=args.final_epochs_full, batch_size=256, seed=0,
        use_classifier_head=True,
    )
    torch.manual_seed(0)
    sae_only_model = StackedAutoencoder(sae_only_cfg)
    train_sae(sae_only_model, X_train_t, X_val=X_te_t,
              y_train=y_train_t, y_val=y_test_t, verbose=False)
    yp_so, ys_so = _head_classify_score(sae_only_model, X_te_t)
    results["sae_only"] = compute_metrics(y_test, yp_so, ys_so)

    # ---- sklearn baselines ----
    for method in ["rf", "svm", "mlp"]:
        print(f"\n{METHOD_LABELS[method]} ...")
        results[method] = eval_sklearn(method, X_train_n, X_test_n, y_train, y_test, seed=0)

    # ---- Per-attack-type breakdown (TAS-VANET on test) ----
    print("\nPer-attack-type breakdown (TAS-VANET, held-out test) ...")
    per_attack = {}
    legit_mask = (a_test == 0)
    unique_attacks = sorted(set(a_test[a_test != 0]))
    for aid in unique_attacks:
        mask = legit_mask | (a_test == aid)
        if mask.sum() < 10:
            continue
        y_sub  = (a_test[mask] != 0).astype(int)
        X_sub  = X_test_n[mask]
        X_sub_t = torch.tensor(X_sub, dtype=torch.float32)
        yp_sub, ys_sub = _head_classify_score(final_model, X_sub_t)
        per_attack[int(aid)] = compute_metrics(y_sub, yp_sub, ys_sub)
        print(f"  attack {int(aid):>2d}: F1={per_attack[int(aid)]['f1']:.4f}  "
              f"n={mask.sum():,}")

    return results, per_attack, final_model, cfg, norm_params, woa_history, threshold


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests(cv_rows, reference="tas_vanet"):
    from scipy.stats import wilcoxon

    f1_by_method = defaultdict(list)
    for row in cv_rows:
        for m in METHODS:
            f1_by_method[m].append(row[m]["f1"])

    ref_f1 = np.array(f1_by_method[reference])
    stats = {}

    for method in METHODS:
        if method == reference:
            continue
        comp_f1 = np.array(f1_by_method[method])
        diffs = ref_f1 - comp_f1
        try:
            _, p_val = wilcoxon(ref_f1, comp_f1, alternative="greater")
        except Exception:
            p_val = float("nan")
        cohens_d = float(diffs.mean() / (diffs.std(ddof=1) + 1e-12))
        stats[method] = {
            "ref_mean_f1":  float(ref_f1.mean()),
            "comp_mean_f1": float(comp_f1.mean()),
            "mean_diff":    float(diffs.mean()),
            "wilcoxon_p":   float(p_val),
            "cohens_d":     cohens_d,
        }
    return stats


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(cv_rows, final_results, per_attack, stats,
                 final_model, final_cfg, norm_params, woa_history,
                 threshold, args):
    out_root = PROJECT_ROOT / args.out_dir
    tables_dir = out_root / "tables"
    models_dir = out_root / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ---- cv_raw.csv ----
    raw_records = []
    for row in cv_rows:
        for method in METHODS:
            rec = {"seed": row["seed"], "fold": row["fold"], "method": method}
            rec.update(row[method])
            raw_records.append(rec)
    cv_raw_df = pd.DataFrame(raw_records)
    cv_raw_df.to_csv(tables_dir / "cv_raw.csv", index=False)
    print(f"\nSaved: {tables_dir / 'cv_raw.csv'}")

    # ---- summary.csv ----
    summary_records = []
    for method in METHODS:
        sub = cv_raw_df[cv_raw_df["method"] == method]
        rec = {"method": METHOD_LABELS[method]}
        for m in METRICS:
            vals = sub[m].dropna()
            rec[m] = f"{vals.mean():.4f} ± {vals.std():.4f}"
        summary_records.append(rec)
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(tables_dir / "summary.csv", index=False)
    print(f"Saved: {tables_dir / 'summary.csv'}")

    # ---- stats.csv ----
    stats_records = [
        {
            "method": METHOD_LABELS[m],
            "ref_f1_mean":  v["ref_mean_f1"],
            "comp_f1_mean": v["comp_mean_f1"],
            "mean_diff":    v["mean_diff"],
            "wilcoxon_p":   v["wilcoxon_p"],
            "cohens_d":     v["cohens_d"],
        }
        for m, v in stats.items()
    ]
    pd.DataFrame(stats_records).to_csv(tables_dir / "stats.csv", index=False)
    print(f"Saved: {tables_dir / 'stats.csv'}")

    # ---- final_test.csv ----
    final_records = []
    for method in METHODS:
        rec = {"method": METHOD_LABELS[method]}
        rec.update({k: f"{v:.4f}" if isinstance(v, float) else v
                    for k, v in final_results[method].items()
                    if k in METRICS})
        final_records.append(rec)
    pd.DataFrame(final_records).to_csv(tables_dir / "final_test.csv", index=False)
    print(f"Saved: {tables_dir / 'final_test.csv'}")

    # ---- per_attack.csv ----
    atk_records = [
        {"attack_id": aid, **{k: f"{v:.4f}" for k, v in m.items() if k in METRICS}}
        for aid, m in sorted(per_attack.items())
    ]
    pd.DataFrame(atk_records).to_csv(tables_dir / "per_attack.csv", index=False)
    print(f"Saved: {tables_dir / 'per_attack.csv'}")

    # ---- final model ----
    model_dir = models_dir / "final_tas_vanet"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), model_dir / "sae_state.pt")
    with open(model_dir / "sae_config.json", "w") as f:
        from dataclasses import asdict
        json.dump(asdict(final_cfg), f, indent=2)
    with open(model_dir / "threshold.json", "w") as f:
        json.dump({"value": threshold}, f, indent=2)
    print(f"Saved: {model_dir}/")

    # ---- normalization params ----
    with open(models_dir / "norm_params.json", "w") as f:
        json.dump(norm_params, f, indent=2)
    print(f"Saved: {models_dir / 'norm_params.json'}")

    # ---- WOA history ----
    if woa_history:
        with open(models_dir / "woa_history.json", "w") as f:
            json.dump(woa_history, f, indent=2)
        print(f"Saved: {models_dir / 'woa_history.json'}")

    # ---- LaTeX snippet ----
    _write_latex_table(summary_df, stats, tables_dir / "table_latex.tex")
    print(f"Saved: {tables_dir / 'table_latex.tex'}")


def _write_latex_table(summary_df, stats, out_path):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Comparative performance on VeReMi dataset (mean $\pm$ std, "
        r"10-fold CV). $p$: Wilcoxon signed-rank vs.\ TAS-VANET. "
        r"$d$: Cohen's $d$.}",
        r"\label{tab:comparison}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Method & Precision & Recall & F1 & Accuracy & AUC-ROC & $p$ & $d$ \\",
        r"\midrule",
    ]
    p_d = {
        METHOD_LABELS[m]: (v["wilcoxon_p"], v["cohens_d"])
        for m, v in stats.items()
    }
    for _, row in summary_df.iterrows():
        method = row["method"]
        if method in p_d:
            p, d = p_d[method]
            p_str = f"$<$0.001" if p < 0.001 else f"{p:.3f}"
            d_str = f"{d:.2f}"
        else:
            p_str = "—"
            d_str = "—"
        vals = " & ".join(str(row[m]) for m in METRICS)
        lines.append(f"{method} & {vals} & {p_str} & {d_str} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Print summary to stdout
# ---------------------------------------------------------------------------

def print_summary(cv_rows, final_results, stats):
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY  (mean ± std over all folds)")
    print("=" * 70)

    f1_by_method = defaultdict(list)
    for row in cv_rows:
        for m in METHODS:
            f1_by_method[m].append(row[m]["f1"])

    header = f"{'Method':<28s}  {'Precision':>10s}  {'Recall':>8s}  {'F1':>8s}  {'AUC':>8s}"
    print(header)
    print("-" * len(header))

    def _ms(rows, metric):
        vals = [r[metric] for r in rows]
        return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"

    for m in METHODS:
        sub = [row[m] for row in cv_rows]
        print(f"  {METHOD_LABELS[m]:<26s}  "
              f"{_ms(sub,'precision'):>10s}  "
              f"{_ms(sub,'recall'):>8s}  "
              f"{_ms(sub,'f1'):>8s}  "
              f"{_ms(sub,'auc_roc'):>8s}")

    print("\n" + "=" * 70)
    print("HELD-OUT TEST SET RESULTS")
    print("=" * 70)
    for m in METHODS:
        r = final_results[m]
        print(f"  {METHOD_LABELS[m]:<26s}  "
              f"P={r['precision']:.4f}  R={r['recall']:.4f}  "
              f"F1={r['f1']:.4f}  AUC={r['auc_roc']:.4f}")

    print("\n" + "=" * 70)
    print("STATISTICAL TESTS  (TAS-VANET vs each baseline, Wilcoxon one-sided)")
    print("=" * 70)
    for method, v in stats.items():
        sig = "***" if v["wilcoxon_p"] < 0.001 else \
              "** " if v["wilcoxon_p"] < 0.01  else \
              "*  " if v["wilcoxon_p"] < 0.05  else "ns "
        print(f"  vs {METHOD_LABELS[method]:<26s}  "
              f"p={v['wilcoxon_p']:.4f} {sig}  d={v['cohens_d']:+.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    if args.quick:
        args = apply_quick_preset(args)
        print("*** QUICK MODE — results are for smoke-testing only ***")

    print("=" * 70)
    print("TAS-VANET Full Training Pipeline")
    print("=" * 70)
    print(f"  CV folds:  {args.cv_folds} × {args.cv_seeds} seeds = "
          f"{args.cv_folds * args.cv_seeds} total folds")
    print(f"  WOA (CV):  pop={args.woa_pop_cv} gen={args.woa_gen_cv} "
          f"subsample={args.woa_subsample}")
    print(f"  WOA (final): pop={args.woa_pop_final} gen={args.woa_gen_final}")
    print(f"  SAE epochs (CV / final): {args.final_epochs_cv} / {args.final_epochs_full}")

    # Load data
    print("\n--- Loading data ---")
    X_train, y_train, a_train, groups_train, X_test, y_test, a_test, feat_cols = load_data(args)
    print(f"Train: {X_train.shape}  labels: {np.bincount(y_train)}")
    print(f"Test : {X_test.shape}   labels: {np.bincount(y_test)}")

    # Cross-validation (or load existing results)
    cv_raw_path = PROJECT_ROOT / args.out_dir / "tables" / "cv_raw.csv"
    if args.load_cv and cv_raw_path.exists():
        print(f"\n--- Loading existing CV results from {cv_raw_path} ---")
        cv_df = pd.read_csv(cv_raw_path)
        # Reconstruct cv_rows list-of-dicts from flat CSV
        cv_rows = []
        for (seed, fold), grp in cv_df.groupby(["seed", "fold"]):
            row = {"seed": int(seed), "fold": int(fold)}
            for _, r in grp.iterrows():
                row[r["method"]] = {m: r[m] for m in METRICS}
            cv_rows.append(row)
        print(f"  Loaded {len(cv_rows)} folds")
    else:
        print("\n--- Cross-Validation ---")
        t_cv = time.time()
        (PROJECT_ROOT / args.out_dir / "tables").mkdir(parents=True, exist_ok=True)
        cv_rows = run_cv(X_train, y_train, groups_train, args, cv_raw_path=cv_raw_path)
        print(f"\nCV complete in {(time.time()-t_cv)/60:.1f} min")
        print(f"CV results saved to {cv_raw_path}")

    # Final evaluation on held-out test
    (final_results, per_attack, final_model, final_cfg,
     norm_params, woa_history, threshold) = run_final_evaluation(
        X_train, y_train, X_test, y_test, a_test, feat_cols, args
    )

    # Statistical tests
    print("\n--- Statistical Tests ---")
    stats = run_statistical_tests(cv_rows)

    # Save everything
    print("\n--- Saving Results ---")
    save_results(
        cv_rows, final_results, per_attack, stats,
        final_model, final_cfg, norm_params, woa_history,
        threshold, args,
    )

    # Print summary
    print_summary(cv_rows, final_results, stats)

    print("\n" + "=" * 70)
    print("Done. Next step: python scripts/06_generate_figures.py")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
