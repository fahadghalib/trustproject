"""
06 — Generate publication-quality figures from training results.

Reads the output of scripts/05_train_full.py and produces:

  Figure 1  results/figures/fig_f1_comparison.png      F1 bar chart + 95% CI
  Figure 2  results/figures/fig_roc_curves.png         ROC curves (all methods)
  Figure 3  results/figures/fig_confusion_matrix.png   Confusion matrix
  Figure 4  results/figures/fig_woa_convergence.png    WOA fitness curve
  Figure 5  results/figures/fig_per_attack_f1.png      Per-attack-type F1

All figures are also saved as PDF for LaTeX.

Usage:
    python scripts/06_generate_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TABLES_DIR  = PROJECT_ROOT / "results" / "tables"
MODELS_DIR  = PROJECT_ROOT / "results" / "models"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

METHODS = ["tas_vanet", "sae_only", "rf", "svm", "mlp"]
METHOD_LABELS = {
    "tas_vanet": "TAS-VANET\n(SAE+WOA)",
    "sae_only":  "SAE only\n(no WOA)",
    "rf":        "Random\nForest",
    "svm":       "SVM\n(RBF)",
    "mlp":       "MLP",
}
METHOD_COLORS = {
    "tas_vanet": "#2c7bb6",
    "sae_only":  "#abd9e9",
    "rf":        "#fdae61",
    "svm":       "#d7191c",
    "mlp":       "#a6d96a",
}

# Publication style
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "legend.fontsize": 10,
    "figure.dpi":      150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

DPI_SAVE = 300


def _save(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  Saved: {FIGURES_DIR / name}.png/pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: F1 bar chart with 95% CI
# ---------------------------------------------------------------------------

def fig_f1_comparison(cv_raw: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(METHODS))
    width = 0.55

    means, cis = [], []
    for m in METHODS:
        vals = cv_raw[cv_raw["method"] == m]["f1"].dropna().to_numpy()
        mean = vals.mean()
        # 95% CI via t-distribution (n-1 df)
        from scipy.stats import t as t_dist
        se = vals.std(ddof=1) / np.sqrt(len(vals))
        ci = t_dist.ppf(0.975, df=len(vals) - 1) * se if len(vals) > 1 else 0.0
        means.append(mean)
        cis.append(ci)

    bars = ax.bar(x, means, width,
                  yerr=cis, capsize=5, error_kw={"elinewidth": 1.5},
                  color=[METHOD_COLORS[m] for m in METHODS],
                  edgecolor="white", linewidth=0.8)

    # Annotate bars
    for bar, mean, ci in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ci + 0.005,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
    ax.set_ylabel("F1-score")
    ax.set_title("Detection F1 — Cross-Validation Results\n"
                 "(error bars: 95% CI)")
    ax.set_ylim(0, min(1.05, max(means) + max(cis) + 0.12))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    _save(fig, "fig_f1_comparison")


# ---------------------------------------------------------------------------
# Figure 2: ROC curves on held-out test set
# ---------------------------------------------------------------------------

def fig_roc_curves(X_test_n, y_test, all_models):
    """all_models: dict method -> (y_score,) or callable."""
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(6, 6))

    for method in METHODS:
        if method not in all_models:
            continue
        y_score = all_models[method]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=METHOD_COLORS[method], linewidth=2,
                label=f"{METHOD_LABELS[method].replace(chr(10),' ')} "
                      f"(AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Held-out Test Set")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(linestyle="--", alpha=0.3)

    _save(fig, "fig_roc_curves")


# ---------------------------------------------------------------------------
# Figure 3: Confusion matrix
# ---------------------------------------------------------------------------

def fig_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    total = cm.sum()
    cm_pct = cm / total * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)

    labels = ["Legitimate", "Malicious"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — TAS-VANET\n(held-out test set, %)")

    for i in range(2):
        for j in range(2):
            color = "white" if cm_pct[i, j] > 50 else "black"
            ax.text(j, i, f"{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", color=color, fontsize=11)

    plt.colorbar(im, ax=ax, label="% of total samples")
    plt.tight_layout()

    _save(fig, "fig_confusion_matrix")


# ---------------------------------------------------------------------------
# Figure 4: WOA convergence
# ---------------------------------------------------------------------------

def fig_woa_convergence(woa_history: list):
    gens     = [h["generation"]   for h in woa_history]
    best_fit = [h["best_fitness"]  for h in woa_history]
    mean_fit = [h["mean_fitness"]  for h in woa_history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gens, best_fit, color="#2c7bb6", linewidth=2.5, label="Best fitness")
    ax.plot(gens, mean_fit, color="#a6d96a", linewidth=1.5,
            linestyle="--", label="Mean fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness  (lower = better)")
    ax.set_title("WOA Convergence Curve")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)

    _save(fig, "fig_woa_convergence")


# ---------------------------------------------------------------------------
# Figure 5: Per-attack-type F1
# ---------------------------------------------------------------------------

def fig_per_attack_f1(per_attack_df: pd.DataFrame):
    df = per_attack_df.copy()
    df["f1"] = df["f1"].astype(float)
    df = df[df["attack_id"] > 0].sort_values("attack_id")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(df))
    bars = ax.bar(x, df["f1"].to_numpy(),
                  color="#2c7bb6", edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, df["f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{float(val):.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"A{int(v)}" for v in df["attack_id"]], fontsize=9)
    ax.set_xlabel("Attack Type")
    ax.set_ylabel("F1-score")
    ax.set_title("Per-Attack-Type Detection F1 — TAS-VANET\n(held-out test set)")
    ax.set_ylim(0, 1.12)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6,
               label="F1=0.5 baseline")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    _save(fig, "fig_per_attack_f1")


# ---------------------------------------------------------------------------
# Figure 6: Metric comparison heatmap (all methods × all metrics)
# ---------------------------------------------------------------------------

def fig_metrics_heatmap(cv_raw: pd.DataFrame):
    metrics = ["precision", "recall", "f1", "accuracy", "auc_roc"]
    metric_labels = ["Precision", "Recall", "F1", "Accuracy", "AUC-ROC"]

    data = np.zeros((len(METHODS), len(metrics)))
    for i, m in enumerate(METHODS):
        sub = cv_raw[cv_raw["method"] == m]
        for j, met in enumerate(metrics):
            data[i, j] = sub[met].mean()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(data, cmap="YlOrRd", vmin=0.4, vmax=1.0)

    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(METHODS)))
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticklabels([METHOD_LABELS[m].replace("\n", " ") for m in METHODS],
                       fontsize=9)
    ax.set_title("Mean Performance Across All CV Folds")

    for i in range(len(METHODS)):
        for j in range(len(metrics)):
            color = "white" if data[i, j] > 0.75 else "black"
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                    color=color, fontsize=9)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    _save(fig, "fig_metrics_heatmap")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TAS-VANET — Figure Generation")
    print("=" * 70)

    # ---- Load CV results (optional — skip CV figures if not ready yet) ----
    cv_path = TABLES_DIR / "cv_raw.csv"
    cv_raw  = None
    if cv_path.exists():
        cv_raw = pd.read_csv(cv_path)
        print(f"Loaded CV results: {len(cv_raw)} fold-method rows")
    else:
        print(f"WARNING: {cv_path} not found — CV figures (1, 6) will be skipped.")
        print("         Run scripts/05_train_full.py first, then re-run this script.")

    # ---- Figure 1: F1 comparison ----
    if cv_raw is not None:
        print("\nFigure 1: F1 bar chart ...")
        fig_f1_comparison(cv_raw)
    else:
        print("\nSkipping Figure 1 (no cv_raw.csv)")

    # ---- Figure 5: Per-attack ----
    atk_path = TABLES_DIR / "per_attack.csv"
    if atk_path.exists():
        print("Figure 5: Per-attack F1 ...")
        per_attack_df = pd.read_csv(atk_path)
        fig_per_attack_f1(per_attack_df)

    # ---- Figure 4: WOA convergence ----
    woa_path = MODELS_DIR / "woa_history.json"
    if woa_path.exists():
        print("Figure 4: WOA convergence ...")
        with open(woa_path) as f:
            woa_history = json.load(f)
        fig_woa_convergence(woa_history)

    # ---- Figure 6: Metrics heatmap ----
    if cv_raw is not None:
        print("Figure 6: Metrics heatmap ...")
        fig_metrics_heatmap(cv_raw)
    else:
        print("Skipping Figure 6 (no cv_raw.csv)")

    # ---- Figures 2 & 3 require model + test data ----
    final_model_dir = MODELS_DIR / "final_tas_vanet"
    test_csv = PROJECT_ROOT / "data" / "processed" / "veremi_test_processed.csv"
    norm_path = MODELS_DIR / "norm_params.json"

    if final_model_dir.exists() and test_csv.exists() and norm_path.exists():
        print("Figure 2: ROC curves ...")
        print("Figure 3: Confusion matrix ...")

        import torch
        from core import (
            SAEConfig, StackedAutoencoder, HYBRID_FEATURES,
        )
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier

        # Load normalization
        with open(norm_path) as f:
            norm_params = json.load(f)

        # Load test data
        test_df = pd.read_csv(test_csv)
        feat_cols = list(norm_params.keys())
        X_test_raw = test_df[feat_cols].to_numpy(dtype=np.float32)
        y_test      = test_df["label"].to_numpy(dtype=np.int32)

        lo  = np.array([norm_params[c]["lo"] for c in feat_cols], dtype=np.float32)
        hi  = np.array([norm_params[c]["hi"] for c in feat_cols], dtype=np.float32)
        rng = hi - lo
        rng[rng < 1e-12] = 1.0
        X_test_n = np.clip((X_test_raw - lo) / rng, 0.0, 1.0)

        # Load train data for re-fitting baselines
        train_df = pd.read_csv(
            PROJECT_ROOT / "data" / "processed" / "veremi_train_processed.csv"
        )
        X_train_raw = train_df[feat_cols].to_numpy(dtype=np.float32)
        y_train      = train_df["label"].to_numpy(dtype=np.int32)
        X_train_n    = np.clip((X_train_raw - lo) / rng, 0.0, 1.0)

        # Load final SAE model (has a fine-tuned classifier head — see
        # scripts/05_train_full.py and core/sae_model.py)
        with open(final_model_dir / "sae_config.json") as f:
            cfg_dict = json.load(f)
        sae_cfg = SAEConfig(**cfg_dict)
        final_model = StackedAutoencoder(sae_cfg)
        final_model.load_state_dict(
            torch.load(final_model_dir / "sae_state.pt", map_location="cpu")
        )

        # Classify via the supervised classifier head (logit > 0 = malicious)
        final_model.eval()
        with torch.no_grad():
            X_te_t     = torch.tensor(X_test_n, dtype=torch.float32)
            _, H_te    = final_model(X_te_t)
            logits_te  = final_model.classify_logits(X_te_t, H_te)
            probs_te   = torch.sigmoid(logits_te).numpy()
        y_pred      = (probs_te > 0.5).astype(int)
        y_score_sae = probs_te  # higher = more malicious

        all_scores = {"tas_vanet": y_score_sae}

        # Refit SAE-only (fixed hyperparameters, no WOA) for ROC — same
        # semi-supervised classifier-head training as the main pipeline.
        X_train_tr_t = torch.tensor(X_train_n, dtype=torch.float32)
        y_train_tr_t = torch.tensor(y_train, dtype=torch.float32)
        n_feat = X_train_n.shape[1]
        h1 = min(64, max(16, n_feat * 2))
        h2 = max(4, n_feat // 3)
        sae_only_cfg = SAEConfig(
            input_dim=n_feat, hidden_dims=[h1, h2],
            learning_rate=1e-3, sparsity_target=0.05, sparsity_weight=3.0,
            epochs=100, batch_size=256, seed=0,
            use_classifier_head=True,
        )
        torch.manual_seed(0)
        sae_only_model = StackedAutoencoder(sae_only_cfg)
        from core import train_sae
        train_sae(sae_only_model, X_train_tr_t, y_train=y_train_tr_t, verbose=False)
        sae_only_model.eval()
        with torch.no_grad():
            _, H_so    = sae_only_model(X_te_t)
            logits_so  = sae_only_model.classify_logits(X_te_t, H_so)
            probs_so   = torch.sigmoid(logits_so).numpy()
        all_scores["sae_only"] = probs_so   # higher = more malicious

        # Refit RF, SVM (subsampled), MLP for ROC
        rng2 = np.random.default_rng(0)
        svm_idx = rng2.choice(len(X_train_n),
                              size=min(10_000, len(X_train_n)), replace=False)
        for name, clf_cls, fit_X, kwargs in [
            ("rf",  RandomForestClassifier, X_train_n,
             {"n_estimators": 100, "n_jobs": -1, "random_state": 0}),
            ("svm", SVC, X_train_n[svm_idx],
             {"kernel": "rbf", "probability": True, "random_state": 0}),
            ("mlp", MLPClassifier, X_train_n,
             {"hidden_layer_sizes": (64, 32), "max_iter": 300, "random_state": 0}),
        ]:
            fit_y = y_train if name != "svm" else y_train[svm_idx]
            print(f"  Fitting {name} for ROC ...", end=" ", flush=True)
            clf = clf_cls(**kwargs)
            clf.fit(fit_X, fit_y)
            all_scores[name] = clf.predict_proba(X_test_n)[:, 1]
            print("done")

        fig_roc_curves(X_test_n, y_test, all_scores)
        fig_confusion_matrix(y_test, y_pred)
    else:
        print("Skipping Figures 2 & 3 (model / test data not found — run 05 first)")

    print("\n" + "=" * 70)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
