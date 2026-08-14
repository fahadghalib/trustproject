"""
01 — Setup check.

Verifies that the Python environment has all required packages and that
the `core` package imports cleanly. Run this first after cloning the project.

Usage (from project root):
    python scripts/01_setup_check.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Add project root to sys.path so we can import `core` no matter where
# this script is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_PACKAGES = [
    ("numpy",        "Numerical operations"),
    ("pandas",       "Data tables"),
    ("torch",        "PyTorch — Stacked Autoencoder"),
    ("sklearn",      "scikit-learn — metrics, baselines"),
    ("matplotlib",   "Plotting figures"),
    ("openpyxl",     "Reading .xlsx files"),
]


def check_python_version() -> bool:
    """Verify Python 3.9 or later."""
    print(f"Python version: {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        print("  FAILED: Python 3.9 or later is required.")
        return False
    print("  OK")
    return True


def check_packages() -> list[str]:
    """Verify each required package is installed; return list of missing names."""
    print("\nChecking required packages:")
    missing: list[str] = []
    for pkg, desc in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"  OK   {pkg:<14s} {version:<12s} ({desc})")
        except ImportError:
            print(f"  MISS {pkg:<14s} (not installed)  ({desc})")
            missing.append(pkg)
    return missing


def check_core_package() -> bool:
    """Verify the local `core` package imports cleanly."""
    print("\nChecking local `core` package:")
    try:
        from core import (
            SAEConfig, WOAConfig, TrustPipeline,
            HYBRID_FEATURES,
        )
        print(f"  OK   core imports successfully")
        print(f"  OK   default SAE input dim:     {SAEConfig().input_dim}")
        print(f"  OK   default WOA generations:   {WOAConfig().max_generations}")
        print(f"  OK   hybrid feature dim:        {len(HYBRID_FEATURES)}")
        return True
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return False


def check_directories() -> None:
    """Verify project directory structure."""
    print("\nChecking directory structure:")
    for sub in ["data/raw", "data/processed",
                "results/models", "results/figures", "results/tables"]:
        path = PROJECT_ROOT / sub
        status = "OK  " if path.exists() else "MISS"
        print(f"  {status} {sub}/")
        path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    print("=" * 70)
    print("TAS-VANET — Environment Setup Check")
    print("=" * 70)

    py_ok = check_python_version()
    missing = check_packages()
    core_ok = check_core_package() if not missing else False
    check_directories()

    print("\n" + "=" * 70)
    if py_ok and not missing and core_ok:
        print("All checks passed. You are ready to run the pipeline.")
        print("\nNext step:")
        print("  1. Download VeReMi Extension from Mendeley (see docs/)")
        print("  2. Place the CSV in data/raw/VeReMi_Extension.csv")
        print("  3. Run scripts/02_process_veremi.py")
        return 0

    print("Some checks failed. Fix the items above before proceeding.")
    if missing:
        print(f"\nInstall missing packages with:")
        print(f"    pip install {' '.join(missing)}")
        print(f"\nOr install everything from requirements.txt:")
        print(f"    pip install -r requirements.txt")
    return 1


if __name__ == "__main__":
    sys.exit(main())
