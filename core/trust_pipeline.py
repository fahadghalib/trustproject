"""
TAS-VANET Trust Pipeline: end-to-end integration of SAE + WOA + threshold logic.

This module wraps the full TAS-VANET trust evaluation flow:

  1. Train SAE with WOA-optimized hyperparameters on calibration features.
  2. Derive a dynamic authentication threshold from latent statistics.
  3. Classify new vehicles as trusted / untrusted based on overall trust
     compared to that threshold.

Classification
--------------
The primary decision mechanism is a linear classifier head fine-tuned on
the SAE bottleneck (see sae_model.StackedAutoencoder.classify_logits),
jointly trained with the reconstruction + sparsity loss using the labeled
calibration data. This is the classic unsupervised-pretraining +
supervised-fine-tuning pattern for stacked autoencoders and gives a much
stronger detection signal than a single latent component.

Threshold derivation (legacy / sensitivity-analysis output)
------------------------------------------------------------
For backward compatibility and for the paper's threshold-sensitivity
analysis, we still compute the mean/std-based statistic on the primary
latent component:

    Threshold_SAE = mu_primary + k * sigma_primary

where mu_primary and sigma_primary are the mean and standard deviation of
the primary latent component over the calibration set, and k is a tunable
coefficient (default 0.0 — i.e. the mean). This value is stored on the
PipelineResult/TrustPipeline for reporting, but classify() uses the
classifier head, not this threshold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from .sae_model import SAEConfig, StackedAutoencoder, train_sae
from .woa_optimizer import (
    HyperparamSpace,
    WOAConfig,
    WhaleOptimizer,
    build_fitness_fn,
)


@dataclass
class PipelineResult:
    """Final outputs of the calibration phase."""

    best_hyperparams: Dict[str, float]
    woa_best_fitness: float
    threshold_k: float
    threshold_mu: float
    threshold_sigma: float
    threshold_value: float
    final_sae_recon_loss: float


class TrustPipeline:
    """End-to-end TAS-VANET trust pipeline.

    Typical usage:
        pipe = TrustPipeline()
        result = pipe.calibrate(X_train, X_val, y_val)
        decisions = pipe.classify(X_new)
        pipe.save("models/trust_v1/")
    """

    def __init__(
        self,
        threshold_k: float = 0.0,
        woa_config: WOAConfig | None = None,
        final_epochs: int = 200,
    ) -> None:
        """
        Parameters
        ----------
        threshold_k : float
            Coefficient k in Threshold = mu + k*sigma. Default 0.0 (mean).
            Values in [-1, 1] are recommended; reported as a sensitivity
            parameter in the paper.
        woa_config : WOAConfig or None
            WOA settings. Defaults to 20 whales / 30 generations.
        final_epochs : int
            Number of epochs to train the FINAL SAE after WOA selects
            hyperparameters (longer than the per-iteration tuning runs).
        """
        self.threshold_k = threshold_k
        self.woa_config = woa_config or WOAConfig()
        self.final_epochs = final_epochs

        self.model: StackedAutoencoder | None = None
        self.best_hyperparams: Dict[str, float] | None = None
        self.threshold_value: float | None = None
        self.threshold_mu: float | None = None
        self.threshold_sigma: float | None = None
        self.woa_history: list | None = None

    # -- calibration --------------------------------------------------------

    def calibrate(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True,
    ) -> PipelineResult:
        """Run WOA to find best SAE hyperparameters, then train final SAE.

        Parameters
        ----------
        X_train : np.ndarray (n_train, n_features) in [0, 1]
        X_val   : np.ndarray (n_val, n_features) in [0, 1]
        y_val   : np.ndarray (n_val,) with 1=malicious, 0=legitimate
        y_train : np.ndarray (n_train,) with 1=malicious, 0=legitimate.
            Used to fine-tune the supervised classifier head (see
            sae_model.py) jointly with the reconstruction/sparsity loss.
        """
        # 1) WOA hyperparameter optimization (searches against the same
        #    classifier-head detection mechanism used at inference time)
        if verbose:
            print("Phase 2a: WOA hyperparameter search ...")
        space = HyperparamSpace()
        fitness_fn = build_fitness_fn(
            X_train=X_train,
            X_val=X_val,
            y_val=y_val,
            y_train=y_train,
            alpha=self.woa_config.alpha_recon,
            beta=self.woa_config.beta_f1,
            epochs=30,                          # short for tuning
            use_classifier_head=True,
        )
        woa = WhaleOptimizer(fitness_fn, space, self.woa_config)
        woa_result = woa.optimize(verbose=verbose)
        self.best_hyperparams = woa_result["best_params"]
        self.woa_history = woa_result["history"]

        # 2) Train final SAE with best hyperparameters (longer training)
        if verbose:
            print("\nPhase 2b: training final SAE with WOA-optimized hyperparameters ...")
            print(f"  Best params: {self.best_hyperparams}")
        cfg = SAEConfig(
            input_dim=X_train.shape[1],
            hidden_dims=[
                int(self.best_hyperparams["h1_size"]),
                int(self.best_hyperparams["h2_size"]),
            ],
            learning_rate=float(self.best_hyperparams["learning_rate"]),
            sparsity_target=float(self.best_hyperparams["sparsity_rho"]),
            sparsity_weight=float(self.best_hyperparams["sparsity_beta"]),
            epochs=self.final_epochs,
            batch_size=64,
            use_classifier_head=True,
        )
        self.model = StackedAutoencoder(cfg)
        train_hist = train_sae(
            self.model,
            torch.tensor(X_train, dtype=torch.float32),
            X_val=torch.tensor(X_val, dtype=torch.float32),
            y_train=torch.tensor(y_train, dtype=torch.float32),
            y_val=torch.tensor(y_val, dtype=torch.float32),
            verbose=verbose,
        )
        final_recon = float(train_hist["final_val_recon_loss"] or train_hist["final_train_loss"])

        # 3) Derive threshold from latent statistics
        if verbose:
            print("\nPhase 2c: deriving authentication threshold from latent statistics ...")
        self.model.eval()
        with torch.no_grad():
            _, H_train = self.model(torch.tensor(X_train, dtype=torch.float32))
        H_train_np = H_train.numpy()
        primary = H_train_np[:, 0]                 # primary latent component
        self.threshold_mu = float(primary.mean())
        self.threshold_sigma = float(primary.std())
        self.threshold_value = self.threshold_mu + self.threshold_k * self.threshold_sigma
        if verbose:
            print(
                f"  mu={self.threshold_mu:.4f}  sigma={self.threshold_sigma:.4f}  "
                f"k={self.threshold_k}  threshold={self.threshold_value:.4f}"
            )

        return PipelineResult(
            best_hyperparams=self.best_hyperparams,
            woa_best_fitness=woa_result["best_fitness"],
            threshold_k=self.threshold_k,
            threshold_mu=self.threshold_mu,
            threshold_sigma=self.threshold_sigma,
            threshold_value=self.threshold_value,
            final_sae_recon_loss=final_recon,
        )

    # -- inference ----------------------------------------------------------

    def classify(self, X: np.ndarray) -> np.ndarray:
        """Classify a batch of vehicles as trusted (0) or untrusted (1).

        Uses the supervised classifier head fine-tuned on the SAE
        bottleneck during calibrate() (logit > 0 => untrusted). This
        replaces the earlier single-latent-component threshold heuristic,
        which is still computed and stored (threshold_value) for the
        paper's sensitivity-analysis section but no longer drives the
        decision.

        Returns
        -------
        decisions : np.ndarray of int (n_samples,) with 1=untrusted, 0=trusted
        """
        if self.model is None:
            raise RuntimeError("Pipeline must be calibrated before classification.")
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            _, H = self.model(X_t)
            logits = self.model.classify_logits(X_t, H)
        return (logits.numpy() > 0).astype(int)

    # -- persistence --------------------------------------------------------

    def save(self, out_dir: str) -> None:
        """Save the trained model and threshold metadata to disk."""
        if self.model is None:
            raise RuntimeError("Nothing to save: pipeline is not calibrated.")
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), out_path / "sae_state.pt")
        with open(out_path / "sae_config.json", "w") as f:
            json.dump(asdict(self.model.config), f, indent=2)
        with open(out_path / "threshold.json", "w") as f:
            json.dump(
                {
                    "k": self.threshold_k,
                    "mu": self.threshold_mu,
                    "sigma": self.threshold_sigma,
                    "value": self.threshold_value,
                    "best_hyperparams": self.best_hyperparams,
                },
                f,
                indent=2,
            )

    def load(self, in_dir: str) -> None:
        """Reload a previously calibrated pipeline from disk."""
        in_path = Path(in_dir)
        with open(in_path / "sae_config.json") as f:
            cfg_dict = json.load(f)
        cfg = SAEConfig(**cfg_dict)
        self.model = StackedAutoencoder(cfg)
        self.model.load_state_dict(torch.load(in_path / "sae_state.pt"))

        with open(in_path / "threshold.json") as f:
            t = json.load(f)
        self.threshold_k = float(t["k"])
        self.threshold_mu = float(t["mu"])
        self.threshold_sigma = float(t["sigma"])
        self.threshold_value = float(t["value"])
        self.best_hyperparams = t["best_hyperparams"]
