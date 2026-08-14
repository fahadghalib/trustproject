"""
02 — Process the VeReMi Extension dataset.

Reads the unified VeReMi Extension CSV (downloaded from Mendeley) and
produces a hybrid feature CSV ready for SAE+WOA training.

Pipeline stages:
  1. Stratified streaming sample (memory-efficient for large CSVs).
  2. Conversion of raw BSMs to per-sender pairwise feature rows.
  3. Computation of the 4 trust features on top of kinematic features.
  4. Output to data/processed/.

Usage:
    python scripts/02_process_veremi.py \\
        --input data/raw/VeReMi_Extension.csv \\
        --output data/processed/veremi_processed.csv \\
        --sample_fraction 0.1

For details on parameters, run with --help.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make project root importable when running from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Iterable
import numpy as np
import pandas as pd

from core import (
    VeReMiLoaderConfig, veremi_csv_to_pairwise,
    TrustFeatureConfig, build_hybrid_features,
)


# ---------------------------------------------------------------------------
# Stratified streaming sampler
# ---------------------------------------------------------------------------

def stratified_chunked_sample(
    input_csv: str | Path,
    fraction: float,
    attack_classes: Iterable[int] | None = None,
    chunksize: int = 200_000,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Stream a large CSV in chunks and stratified-sample by `class`."""
    if fraction <= 0 or fraction > 1:
        raise ValueError("fraction must be in (0, 1]")

    rng = np.random.default_rng(seed)
    keep_classes = set(int(c) for c in attack_classes) if attack_classes else None
    sampled_chunks: list[pd.DataFrame] = []
    t0 = time.time()
    total_in = 0
    total_out = 0

    reader = pd.read_csv(input_csv, chunksize=chunksize)
    for chunk_idx, chunk in enumerate(reader):
        total_in += len(chunk)
        if keep_classes is not None:
            chunk = chunk[chunk["class"].isin(keep_classes)]
        if chunk.empty:
            continue

        parts: list[pd.DataFrame] = []
        for cls, grp in chunk.groupby("class", sort=False):
            n_take = int(len(grp) * fraction)
            if n_take == 0 and fraction > 0 and len(grp) > 0:
                n_take = 1
            if n_take >= len(grp):
                parts.append(grp)
            else:
                idx = rng.choice(len(grp), size=n_take, replace=False)
                parts.append(grp.iloc[idx])
        chunk_sample = pd.concat(parts, ignore_index=True)
        sampled_chunks.append(chunk_sample)
        total_out += len(chunk_sample)

        if verbose:
            print(
                f"  chunk {chunk_idx+1}: read {len(chunk):,}, "
                f"kept {len(chunk_sample):,}, total_out={total_out:,}"
            )

    elapsed = time.time() - t0
    if verbose:
        print(
            f"\nStreaming sample done: read {total_in:,} rows, "
            f"kept {total_out:,} ({100*total_out/max(total_in,1):.2f}%), "
            f"elapsed {elapsed:.1f}s"
        )

    return pd.concat(sampled_chunks, ignore_index=True) if sampled_chunks else pd.DataFrame()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_pipeline(
    input_csv: str,
    output_csv: str,
    sample_fraction: float,
    attack_classes: list[int] | None,
    window_seconds: float,
    comm_range_m: float,
    seed: int,
) -> dict:
    print("=" * 70)
    print("Step 1: Stratified sampling")
    print("=" * 70)
    raw = stratified_chunked_sample(
        input_csv, fraction=sample_fraction,
        attack_classes=attack_classes, seed=seed,
    )
    if raw.empty:
        raise RuntimeError("Sampling produced no rows.")
    print(f"\nSampled frame: {raw.shape}")
    print("Class distribution:")
    print(raw["class"].value_counts().sort_index().to_string())
    print(f"Unique senders: {raw['senderPseudo'].nunique()}")

    print("\n" + "=" * 70)
    print("Step 2: Pairwise per-sender feature conversion")
    print("=" * 70)
    pairs = veremi_csv_to_pairwise(raw, VeReMiLoaderConfig())
    print(f"Pairwise frame: {pairs.shape}")
    if pairs.empty:
        raise RuntimeError("Pairwise conversion produced no rows.")
    print(f"Pairs balance: {pairs['label'].value_counts().to_dict()} "
          f"(malicious fraction = {pairs['label'].mean():.1%})")

    print("\n" + "=" * 70)
    print("Step 3: Hybrid feature extraction (kinematic + trust)")
    print("=" * 70)
    trust_cfg = TrustFeatureConfig(
        window_seconds=window_seconds, comm_range_m=comm_range_m,
    )
    print(f"  Trust config: window={window_seconds}s, comm_range={comm_range_m}m")
    hybrid = build_hybrid_features(pairs, trust_cfg)
    print(f"Hybrid feature frame: {hybrid.shape}")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    hybrid.to_csv(output_csv, index=False)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Output file        : {output_csv}")
    print(f"  Total rows         : {len(hybrid):,}")
    print(f"  Unique senders     : {hybrid['sender_pseudo'].nunique():,}")
    print(f"  Per-attack counts  :")
    for aid, n in hybrid["attack_id"].value_counts().sort_index().items():
        tag = "legitimate" if int(aid) == 0 else f"attack {int(aid)}"
        print(f"      {tag:>15s}: {n:>10,d}")

    return {
        "n_rows": len(hybrid),
        "n_senders": int(hybrid["sender_pseudo"].nunique()),
        "malicious_fraction": float(hybrid["label"].mean()),
        "output_path": str(output_csv),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Process VeReMi Extension CSV into hybrid features."
    )
    parser.add_argument("--input", required=True,
                        help="Path to the VeReMi Extension CSV downloaded from Mendeley.")
    parser.add_argument("--output", required=True,
                        help="Output path for the processed hybrid-feature CSV.")
    parser.add_argument("--sample_fraction", type=float, default=0.1,
                        help="Per-class sampling fraction (default 0.1).")
    parser.add_argument("--attack_classes", type=int, nargs="+", default=None,
                        help="Restrict to specific attack class IDs.")
    parser.add_argument("--window_seconds", type=float, default=5.0)
    parser.add_argument("--comm_range_m", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    process_pipeline(
        input_csv=args.input,
        output_csv=args.output,
        sample_fraction=args.sample_fraction,
        attack_classes=args.attack_classes,
        window_seconds=args.window_seconds,
        comm_range_m=args.comm_range_m,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
