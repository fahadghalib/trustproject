"""
Stacked Autoencoder for VANET Trust Feature Encoding.

This module implements the SAE component of the TAS-VANET framework.
The architecture learns a compressed latent representation of multi-dimensional
trust features, which is then used to derive a dynamic authentication threshold.

Mathematical formulation
------------------------
For each autoencoder layer in the stack:

    Encoder:  h = sigma_f(W*x + b)
    Decoder:  x' = sigma_g(W'*h + b')

The training objective is a sparse autoencoder loss:

    L_sparse(x, x') = MSE(x, x') + beta * sum_j KL(rho || rho_hat_j)

where:
    - rho        = target sparsity level (small constant, e.g. 0.05)
    - rho_hat_j  = mean activation of hidden unit j over a batch
    - KL(rho||rho_hat_j) = rho*log(rho/rho_hat_j)
                         + (1-rho)*log((1-rho)/(1-rho_hat_j))
    - beta       = sparsity penalty weight

Hidden-layer activation
-----------------------
Intermediate encoder/decoder layers may use LeakyReLU (Maas et al., 2013,
negative slope 0.1) instead of Sigmoid (config.hidden_activation), to avoid
sigmoid saturation limiting representational capacity in deeper stacks.
The encoder's bottleneck layer stays Sigmoid because the KL-sparsity term
requires activations in (0, 1); the decoder's final layer stays Sigmoid
because reconstruction targets are min-max normalized to [0, 1].

Supervised fine-tuning (classifier head with skip connection)
----------------------------------------------------------------
Optionally, a linear classification head is attached to the bottleneck,
with a skip/shortcut connection (He et al., 2016) from the raw input so
the head is not limited to whatever survives the reconstruction bottleneck:

    logit = W_c*[x ; h] + b_c
    L_cls(y, logit) = BCEWithLogits(y, logit)
    L_total = L_sparse(x, x') + gamma * L_cls(y, logit)

This follows the classic "unsupervised pretraining + supervised fine-tuning"
pattern for stacked autoencoders (Bengio et al., 2007; applied to sparse
autoencoders for intrusion/misbehavior detection by Javaid et al., 2016),
combined with a ResNet-style shortcut connection so the discriminative head
sees both the raw features and the learned trust representation. The
encoder still learns a reconstruction-regularized representation for the
trust score, but the detection decision is driven by a head trained
directly on labels with full access to the input, instead of a single
latent component compared to a heuristic threshold.

References
----------
Ng, A. (2011). Sparse autoencoder. CS294A Lecture Notes, Stanford.
Vincent, P. et al. (2010). Stacked denoising autoencoders. JMLR 11.
Bengio, Y. et al. (2007). Greedy layer-wise training of deep networks. NIPS 19.
Javaid, A. et al. (2016). A deep learning approach for network intrusion
    detection system. Proc. 9th EAI Int'l Conf. on Bio-inspired Information
    and Communications Technologies (BICT).
He, K. et al. (2016). Deep residual learning for image recognition
    (skip/shortcut connections). CVPR.
Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013). Rectifier nonlinearities
    improve neural network acoustic models. ICML Workshop on Deep Learning
    for Audio, Speech and Language Processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SAEConfig:
    """Hyperparameters defining an SAE instance.

    Attributes
    ----------
    input_dim : int
        Number of input trust features (default 4: residual energy,
        avg distance, node degree, historical trust).
    hidden_dims : list of int
        Sizes of hidden encoder layers. The decoder is a mirror image.
        For TAS-VANET we use 2 hidden layers: [h1_size, h2_size].
        The last value is the latent (bottleneck) dimension.
    learning_rate : float
        Adam optimizer learning rate.
    sparsity_target : float
        Target average activation rho (KL divergence target).
    sparsity_weight : float
        Weight beta of the KL sparsity penalty in the loss.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size for SGD.
    use_classifier_head : bool
        If True, attach a linear classification head on the bottleneck and
        fine-tune it (and the encoder) with labeled data alongside the
        unsupervised reconstruction/sparsity loss.
    classifier_weight : float
        Weight gamma of the BCE classification term in the combined loss.
    """

    input_dim: int = 4
    hidden_dims: List[int] = field(default_factory=lambda: [16, 8])
    learning_rate: float = 1e-3
    sparsity_target: float = 0.05
    sparsity_weight: float = 3.0
    epochs: int = 100
    batch_size: int = 64
    seed: int = 42

    # -- Supervised classifier head (optional) --
    use_classifier_head: bool = False
    classifier_weight: float = 1.5

    # -- Hidden-layer activation ("sigmoid" or "leaky_relu") --
    # Only the INTERMEDIATE encoder/decoder layers switch; the encoder's
    # final (bottleneck) layer and the decoder's final (reconstruction)
    # layer always stay Sigmoid, since the KL-sparsity term requires
    # bottleneck activations in (0, 1) and the reconstruction target is
    # min-max normalized to [0, 1]. See StackedAutoencoder for details.
    hidden_activation: str = "sigmoid"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StackedAutoencoder(nn.Module):
    """Symmetric stacked autoencoder.

    The encoder maps R^{input_dim} -> R^{latent_dim} through a sequence of
    fully-connected layers. The decoder mirrors the encoder structure to
    reconstruct the input. Intermediate layers use config.hidden_activation
    ("sigmoid" or "leaky_relu"); the encoder's bottleneck layer and the
    decoder's output layer always stay Sigmoid (see SAEConfig docstring).
    """

    def __init__(self, config: SAEConfig) -> None:
        super().__init__()
        self.config = config
        torch.manual_seed(config.seed)

        def _activation(is_final: bool) -> nn.Module:
            if is_final or config.hidden_activation == "sigmoid":
                return nn.Sigmoid()
            return nn.LeakyReLU(0.1)

        # Build encoder layers: input_dim -> hidden_dims[0] -> ... -> hidden_dims[-1]
        encoder_layers: List[nn.Module] = []
        prev_dim = config.input_dim
        n_enc = len(config.hidden_dims)
        for i, h_dim in enumerate(config.hidden_dims):
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(_activation(is_final=(i == n_enc - 1)))
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers (mirror of encoder)
        decoder_layers: List[nn.Module] = []
        reversed_dims = list(reversed(config.hidden_dims[:-1])) + [config.input_dim]
        prev_dim = config.hidden_dims[-1]
        n_dec = len(reversed_dims)
        for i, d_dim in enumerate(reversed_dims):
            decoder_layers.append(nn.Linear(prev_dim, d_dim))
            decoder_layers.append(_activation(is_final=(i == n_dec - 1)))
            prev_dim = d_dim
        self.decoder = nn.Sequential(*decoder_layers)

        # Optional supervised classification head with a skip/shortcut
        # connection (He et al., 2016) from the raw input: the head sees
        # [x ; h] instead of h alone, so it is not bottlenecked by the
        # dimensionality reduction that the reconstruction objective forces
        # onto the latent code.
        self.classifier: nn.Linear | None = (
            nn.Linear(config.input_dim + config.hidden_dims[-1], 1)
            if config.use_classifier_head
            else None
        )

    @property
    def latent_dim(self) -> int:
        """Return the size of the bottleneck (last encoder layer)."""
        return self.config.hidden_dims[-1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map input batch to latent representation."""
        return self.encoder(x)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Map latent representation back to input space."""
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (reconstruction, latent)."""
        h = self.encode(x)
        x_recon = self.decode(h)
        return x_recon, h

    def classify_logits(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Map [raw input ; latent representation] to classification logits.

        Requires config.use_classifier_head=True. Apply sigmoid to get
        probabilities; a raw logit > 0 corresponds to probability > 0.5.
        The skip connection to x means the head is not limited to whatever
        information survived the reconstruction bottleneck in h.
        """
        if self.classifier is None:
            raise RuntimeError(
                "Model has no classifier head (config.use_classifier_head=False)."
            )
        return self.classifier(torch.cat([x, h], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def kl_divergence(rho: float, rho_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL divergence between target sparsity rho and empirical activations.

    KL(rho || rho_hat_j) = rho*log(rho/rho_hat_j) + (1-rho)*log((1-rho)/(1-rho_hat_j))

    Parameters
    ----------
    rho : float
        Scalar target sparsity (e.g. 0.05).
    rho_hat : torch.Tensor
        Empirical mean activation per hidden unit. Shape (hidden_dim,).
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    torch.Tensor
        Scalar summed KL divergence across all hidden units.
    """
    rho_hat = torch.clamp(rho_hat, eps, 1.0 - eps)
    kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log(
        (1 - rho) / (1 - rho_hat)
    )
    return kl.sum()


def sparse_ae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    h: torch.Tensor,
    rho: float,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the sparse autoencoder loss.

    Returns
    -------
    total_loss : scalar tensor
    recon_loss : scalar tensor (MSE)
    sparsity_loss : scalar tensor (beta * sum KL)
    """
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")
    rho_hat = h.mean(dim=0)                      # empirical sparsity per unit
    sparsity_loss = beta * kl_divergence(rho, rho_hat)
    total_loss = recon_loss + sparsity_loss
    return total_loss, recon_loss, sparsity_loss


def classification_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy loss (from logits) for the classifier head."""
    return nn.functional.binary_cross_entropy_with_logits(logits, y.float())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sae(
    model: StackedAutoencoder,
    X_train: torch.Tensor,
    X_val: torch.Tensor | None = None,
    y_train: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    verbose: bool = False,
) -> dict:
    """Train an SAE with the configured hyperparameters.

    Parameters
    ----------
    model : StackedAutoencoder
        Initialized SAE instance.
    X_train : Tensor of shape (n_samples, input_dim)
        Training feature matrix, values in [0, 1].
    X_val : Tensor or None
        Optional validation set for monitoring.
    y_train, y_val : Tensor or None
        Binary labels (1=malicious). Required only if
        model.config.use_classifier_head is True: the classifier head is
        then fine-tuned jointly with the encoder via BCE loss, weighted by
        model.config.classifier_weight and added to the sparse AE loss.
    verbose : bool
        Print per-epoch losses if True.

    Returns
    -------
    dict with keys:
        'train_losses': list[float]  — per-epoch total training loss
        'val_recon_losses': list[float] or None
        'final_train_loss': float
        'final_val_recon_loss': float or None
        'val_cls_losses': list[float] or None
        'final_val_cls_loss': float or None
    """
    cfg = model.config
    use_head = cfg.use_classifier_head and model.classifier is not None
    if use_head and (y_train is None):
        raise ValueError(
            "model.config.use_classifier_head=True requires y_train for fine-tuning."
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    n = X_train.shape[0]
    batch_size = min(cfg.batch_size, n)
    train_losses: List[float] = []
    val_recon_losses: List[float] = []
    val_cls_losses: List[float] = []

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            x_batch = X_train[idx]

            optimizer.zero_grad()
            x_recon, h = model(x_batch)
            loss, _recon, _sp = sparse_ae_loss(
                x_batch, x_recon, h, cfg.sparsity_target, cfg.sparsity_weight
            )
            if use_head:
                logits = model.classify_logits(x_batch, h)
                loss = loss + cfg.classifier_weight * classification_loss(
                    logits, y_train[idx]
                )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                xv_recon, hv = model(X_val)
                v_recon = nn.functional.mse_loss(xv_recon, X_val).item()
                v_cls = None
                if use_head and y_val is not None:
                    v_cls = classification_loss(
                        model.classify_logits(X_val, hv), y_val
                    ).item()
            val_recon_losses.append(v_recon)
            if v_cls is not None:
                val_cls_losses.append(v_cls)
            if verbose and (epoch % 10 == 0 or epoch == cfg.epochs - 1):
                msg = (
                    f"  Epoch {epoch+1:3d}/{cfg.epochs}: "
                    f"train_loss={avg_train_loss:.5f}  val_recon={v_recon:.5f}"
                )
                if v_cls is not None:
                    msg += f"  val_cls={v_cls:.5f}"
                print(msg)
        elif verbose and (epoch % 10 == 0 or epoch == cfg.epochs - 1):
            print(f"  Epoch {epoch+1:3d}/{cfg.epochs}: train_loss={avg_train_loss:.5f}")

    return {
        "train_losses": train_losses,
        "val_recon_losses": val_recon_losses if X_val is not None else None,
        "final_train_loss": train_losses[-1],
        "final_val_recon_loss": val_recon_losses[-1] if X_val is not None else None,
        "val_cls_losses": val_cls_losses if val_cls_losses else None,
        "final_val_cls_loss": val_cls_losses[-1] if val_cls_losses else None,
    }
