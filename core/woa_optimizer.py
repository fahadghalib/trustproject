"""
Whale Optimization Algorithm (WOA) for SAE Hyperparameter Tuning.

This module implements WOA following Mirjalili & Lewis (2016) and applies it
to optimize the hyperparameters of the Stacked Autoencoder used in TAS-VANET.

Mathematical formulation
------------------------
WOA simulates three behaviors of humpback whales:

1) Encircling prey (|A| < 1):
    X(t+1) = X*(t) - A . D
    where  D = |C . X*(t) - X(t)|,
           A = 2a . r - a,
           C = 2 . r,
           a decreases linearly from 2 to 0 over iterations.

2) Bubble-net attack (spiral, with probability 0.5):
    X(t+1) = D' . exp(b . l) . cos(2*pi*l) + X*(t)
    where  D' = |X*(t) - X(t)|, b is spiral constant, l in [-1, 1].

3) Search for prey (|A| >= 1):
    X(t+1) = X_rand(t) - A . D
    where  D = |C . X_rand(t) - X(t)|.

Fitness function (the heart of the optimization)
------------------------------------------------
    f(theta) = alpha * L_recon(theta) + beta * (1 - F1_detect(theta))

where:
    theta            = SAE hyperparameter vector
    L_recon(theta)   = SAE reconstruction loss on validation split
    F1_detect(theta) = malicious-node detection F1 score on validation split
    alpha, beta      = weights (default alpha=0.3, beta=0.7)

The intuition: we want low reconstruction error AND high detection F1.
Detection is weighted higher because it directly serves the security goal.

References
----------
Mirjalili, S., & Lewis, A. (2016). The Whale Optimization Algorithm.
Advances in Engineering Software, 95, 51-67.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------

@dataclass
class HyperparamSpace:
    """Defines the search bounds for each SAE hyperparameter.

    Each entry is (low, high, is_integer).
    """

    h1_size: Tuple[int, int, bool] = (8, 32, True)
    h2_size: Tuple[int, int, bool] = (2, 8, True)
    learning_rate: Tuple[float, float, bool] = (1e-4, 1e-2, False)
    sparsity_rho: Tuple[float, float, bool] = (0.01, 0.2, False)
    sparsity_beta: Tuple[float, float, bool] = (0.5, 5.0, False)

    @property
    def names(self) -> List[str]:
        return ["h1_size", "h2_size", "learning_rate", "sparsity_rho", "sparsity_beta"]

    @property
    def bounds(self) -> List[Tuple[float, float, bool]]:
        return [
            self.h1_size,
            self.h2_size,
            self.learning_rate,
            self.sparsity_rho,
            self.sparsity_beta,
        ]

    @property
    def dim(self) -> int:
        return len(self.bounds)


def decode_position(position: np.ndarray, space: HyperparamSpace) -> Dict[str, float]:
    """Convert a continuous WOA position vector to a hyperparameter dict.

    Integer-valued hyperparameters are rounded; all values are clipped
    to their defined bounds.
    """
    decoded: Dict[str, float] = {}
    for i, name in enumerate(space.names):
        low, high, is_int = space.bounds[i]
        val = float(np.clip(position[i], low, high))
        if is_int:
            val = int(round(val))
        decoded[name] = val
    return decoded


# ---------------------------------------------------------------------------
# WOA configuration
# ---------------------------------------------------------------------------

@dataclass
class WOAConfig:
    """WOA algorithm hyperparameters."""

    population_size: int = 20
    max_generations: int = 30
    spiral_constant_b: float = 1.0     # b in the spiral equation
    alpha_recon: float = 0.3           # weight on reconstruction loss
    beta_f1: float = 0.7               # weight on (1 - F1)
    seed: int = 42


# ---------------------------------------------------------------------------
# WOA optimizer
# ---------------------------------------------------------------------------

class WhaleOptimizer:
    """Whale Optimization Algorithm for SAE hyperparameter tuning.

    Parameters
    ----------
    fitness_fn : callable
        Function mapping a hyperparameter dict to a scalar fitness value.
        Lower is better. The fitness should already combine reconstruction
        and detection objectives (see build_fitness_fn below).
    space : HyperparamSpace
        Search space definition.
    config : WOAConfig
        WOA algorithm settings.
    """

    def __init__(
        self,
        fitness_fn: Callable[[Dict[str, float]], float],
        space: HyperparamSpace,
        config: WOAConfig | None = None,
    ) -> None:
        self.fitness_fn = fitness_fn
        self.space = space
        self.config = config or WOAConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.best_position: np.ndarray | None = None
        self.best_fitness: float = float("inf")
        self.history: List[Dict] = []          # per-generation best fitness

    # -- initialization -----------------------------------------------------

    def _init_population(self) -> np.ndarray:
        """Sample a uniform random initial population."""
        n = self.config.population_size
        d = self.space.dim
        pop = np.empty((n, d))
        for j in range(d):
            low, high, _ = self.space.bounds[j]
            pop[:, j] = self.rng.uniform(low, high, size=n)
        return pop

    def _clip(self, position: np.ndarray) -> np.ndarray:
        """Clip a position to the search bounds (in-place safe)."""
        clipped = position.copy()
        for j in range(self.space.dim):
            low, high, _ = self.space.bounds[j]
            clipped[j] = np.clip(clipped[j], low, high)
        return clipped

    # -- main loop ----------------------------------------------------------

    def optimize(self, verbose: bool = False) -> Dict:
        """Run the WOA optimization loop.

        Returns
        -------
        dict with keys:
            'best_params'  : dict of optimal hyperparameter values
            'best_fitness' : float
            'history'      : list of per-generation summaries
        """
        cfg = self.config
        population = self._init_population()

        # Evaluate initial population
        fitnesses = np.array(
            [self.fitness_fn(decode_position(p, self.space)) for p in population]
        )

        best_idx = int(np.argmin(fitnesses))
        self.best_position = population[best_idx].copy()
        self.best_fitness = float(fitnesses[best_idx])

        self.history.append(
            {
                "generation": 0,
                "best_fitness": self.best_fitness,
                "mean_fitness": float(fitnesses.mean()),
            }
        )
        if verbose:
            print(f"  Gen  0: best_fitness={self.best_fitness:.5f}")

        # Main loop
        for gen in range(1, cfg.max_generations + 1):
            a = 2.0 - gen * (2.0 / cfg.max_generations)     # linearly 2 -> 0

            for i in range(cfg.population_size):
                r1 = self.rng.uniform(0, 1, size=self.space.dim)
                r2 = self.rng.uniform(0, 1, size=self.space.dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                l = self.rng.uniform(-1, 1)
                p = self.rng.uniform(0, 1)

                if p < 0.5:
                    if np.linalg.norm(A) < 1.0:
                        # ---- Encircling prey (exploit best) ----
                        D = np.abs(C * self.best_position - population[i])
                        new_pos = self.best_position - A * D
                    else:
                        # ---- Search for prey (explore around random whale) ----
                        rand_idx = self.rng.integers(0, cfg.population_size)
                        X_rand = population[rand_idx]
                        D = np.abs(C * X_rand - population[i])
                        new_pos = X_rand - A * D
                else:
                    # ---- Bubble-net attack (spiral around best) ----
                    D_prime = np.abs(self.best_position - population[i])
                    new_pos = (
                        D_prime * np.exp(cfg.spiral_constant_b * l) * np.cos(2 * np.pi * l)
                        + self.best_position
                    )

                new_pos = self._clip(new_pos)
                new_fitness = self.fitness_fn(decode_position(new_pos, self.space))

                # Update individual and global best
                population[i] = new_pos
                fitnesses[i] = new_fitness
                if new_fitness < self.best_fitness:
                    self.best_fitness = float(new_fitness)
                    self.best_position = new_pos.copy()

            self.history.append(
                {
                    "generation": gen,
                    "best_fitness": self.best_fitness,
                    "mean_fitness": float(fitnesses.mean()),
                }
            )
            if verbose and (gen % 5 == 0 or gen == cfg.max_generations):
                print(
                    f"  Gen {gen:2d}: best_fitness={self.best_fitness:.5f}  "
                    f"mean={fitnesses.mean():.5f}"
                )

        best_params = decode_position(self.best_position, self.space)
        return {
            "best_params": best_params,
            "best_fitness": self.best_fitness,
            "history": self.history,
        }


# ---------------------------------------------------------------------------
# Fitness function builder
# ---------------------------------------------------------------------------

def build_fitness_fn(
    X_train: "np.ndarray | None",
    X_val: "np.ndarray | None",
    y_val: "np.ndarray | None",
    y_train: "np.ndarray | None" = None,
    alpha: float = 0.3,
    beta: float = 0.7,
    epochs: int = 30,
    use_classifier_head: bool = False,
    classifier_weight: float = 1.5,
) -> Callable[[Dict[str, float]], float]:
    """Construct a closure that evaluates fitness for given hyperparameters.

    Fitness = alpha * recon_loss + beta * (1 - F1_detect)

    Detection labels y_val: 1 = malicious, 0 = legitimate.

    Two detection modes:
    - use_classifier_head=False (default): a node is flagged as malicious
      if its reconstruction error is above the median error of the
      training set (a simple, defensible one-class proxy).
    - use_classifier_head=True: the SAE is trained with a supervised
      classification head fine-tuned on (X_train, y_train) alongside the
      reconstruction/sparsity loss (see sae_model.py), and F1 is computed
      directly from the head's predictions on X_val. This lets WOA search
      hyperparameters against the actual detection mechanism used at
      inference time, rather than a threshold proxy.

    Parameters
    ----------
    X_train, X_val : np.ndarray
        Feature matrices in [0,1]. Required (non-None).
    y_val : np.ndarray
        Binary labels (1=malicious) aligned with X_val. Required.
    y_train : np.ndarray or None
        Binary labels aligned with X_train. Required when
        use_classifier_head=True.
    alpha, beta : float
        Weights for reconstruction and (1 - F1) terms.
    epochs : int
        Number of SAE training epochs used during tuning (kept small
        for speed; final model is trained longer outside WOA).
    use_classifier_head, classifier_weight : see sae_model.SAEConfig.
    """
    import torch
    from sklearn.metrics import f1_score

    from .sae_model import SAEConfig, StackedAutoencoder, train_sae

    if X_train is None or X_val is None or y_val is None:
        raise ValueError("X_train, X_val, and y_val are required for fitness evaluation.")
    if use_classifier_head and y_train is None:
        raise ValueError("y_train is required when use_classifier_head=True.")

    Xt = torch.tensor(X_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    Yt = torch.tensor(y_train, dtype=torch.float32) if y_train is not None else None
    Yv = torch.tensor(y_val, dtype=torch.float32) if use_classifier_head else None

    def fitness(params: Dict[str, float]) -> float:
        cfg = SAEConfig(
            input_dim=X_train.shape[1],
            hidden_dims=[int(params["h1_size"]), int(params["h2_size"])],
            learning_rate=float(params["learning_rate"]),
            sparsity_target=float(params["sparsity_rho"]),
            sparsity_weight=float(params["sparsity_beta"]),
            epochs=epochs,
            batch_size=64,
            use_classifier_head=use_classifier_head,
            classifier_weight=classifier_weight,
        )
        model = StackedAutoencoder(cfg)
        train_sae(model, Xt, X_val=Xv, y_train=Yt, y_val=Yv, verbose=False)

        model.eval()
        with torch.no_grad():
            Xv_recon, Hv = model(Xv)
            recon_per_sample = ((Xv_recon - Xv) ** 2).mean(dim=1).numpy()
            recon_loss = float(recon_per_sample.mean())

            if use_classifier_head:
                logits = model.classify_logits(Xv, Hv)
                y_pred = (logits > 0).int().numpy()
            else:
                # Median-based detection threshold (one-class tuning proxy)
                Xt_recon, _ = model(Xt)
                train_err = ((Xt_recon - Xt) ** 2).mean(dim=1).numpy()
                tau = float(np.median(train_err) + np.std(train_err))
                y_pred = (recon_per_sample > tau).astype(int)

        try:
            f1 = float(f1_score(y_val, y_pred, zero_division=0))
        except Exception:
            f1 = 0.0

        return alpha * recon_loss + beta * (1.0 - f1)

    return fitness
