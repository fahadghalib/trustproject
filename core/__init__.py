"""
TAS-VANET Core Package.

Hybrid kinematic-trust misbehavior detection in VANETs using a Stacked
Autoencoder with Whale Optimization Algorithm.

Public exports
--------------
SAE:
    SAEConfig, StackedAutoencoder, train_sae, sparse_ae_loss

WOA:
    WOAConfig, HyperparamSpace, WhaleOptimizer, build_fitness_fn

Pipeline:
    TrustPipeline, PipelineResult

Feature engineering:
    TrustFeatureConfig, build_hybrid_features,
    HYBRID_FEATURES, KINEMATIC_FEATURES_KEPT, TRUST_FEATURES_ADDED

VeReMi loader:
    VeReMiLoaderConfig, load_veremi_csv, veremi_csv_to_pairwise

Synthetic data (validation only):
    SyntheticConfig, generate_synthetic_trust_data, train_val_test_split
"""

__version__ = "0.1.0"

from .sae_model import (
    SAEConfig,
    StackedAutoencoder,
    train_sae,
    sparse_ae_loss,
    kl_divergence,
)
from .woa_optimizer import (
    WOAConfig,
    HyperparamSpace,
    WhaleOptimizer,
    build_fitness_fn,
    decode_position,
)
from .trust_pipeline import (
    TrustPipeline,
    PipelineResult,
)
from .hybrid_feature_extractor import (
    TrustFeatureConfig,
    build_hybrid_features,
    normalize_for_sae,
    apply_normalization,
    HYBRID_FEATURES,
    KINEMATIC_FEATURES_KEPT,
    TRUST_FEATURES_ADDED,
)
from .veremi_loader import (
    VeReMiLoaderConfig,
    load_veremi_csv,
    load_veremi_csvs,
    veremi_csv_to_pairwise,
)
from .synthetic_data import (
    SyntheticConfig,
    generate_synthetic_trust_data,
    train_val_test_split,
)

__all__ = [
    # SAE
    "SAEConfig",
    "StackedAutoencoder",
    "train_sae",
    "sparse_ae_loss",
    "kl_divergence",
    # WOA
    "WOAConfig",
    "HyperparamSpace",
    "WhaleOptimizer",
    "build_fitness_fn",
    "decode_position",
    # Pipeline
    "TrustPipeline",
    "PipelineResult",
    # Features
    "TrustFeatureConfig",
    "build_hybrid_features",
    "normalize_for_sae",
    "apply_normalization",
    "HYBRID_FEATURES",
    "KINEMATIC_FEATURES_KEPT",
    "TRUST_FEATURES_ADDED",
    # Loader
    "VeReMiLoaderConfig",
    "load_veremi_csv",
    "load_veremi_csvs",
    "veremi_csv_to_pairwise",
    # Synthetic
    "SyntheticConfig",
    "generate_synthetic_trust_data",
    "train_val_test_split",
]
