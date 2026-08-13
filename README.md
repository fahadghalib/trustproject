# TAS-VANET

**Hybrid Kinematic-Trust Misbehavior Detection in VANETs**
**using Sparse Autoencoder with Whale-Optimized Hyperparameters**

A Python research codebase for VANET misbehavior detection using a Stacked
Autoencoder (SAE) with Whale Optimization Algorithm (WOA) hyperparameter
tuning, evaluated on the VeReMi Extension dataset.

---

## Quick start in VS Code (4 steps)

### 1. Open the project

```bash
cd TAS-VANET
code .
```

VS Code will prompt to install recommended extensions
(`ms-python.python`, `ms-python.debugpy`, etc.). Accept them.

### 2. Create a virtual environment and install dependencies

In VS Code's integrated terminal (`` Ctrl+` ``):

```bash
# Create venv
python3 -m venv .venv

# Activate (choose your OS)
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\activate              # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

Then in VS Code: open the command palette (`Cmd+Shift+P` / `Ctrl+Shift+P`)
→ **"Python: Select Interpreter"** → choose `.venv/bin/python`.

### 3. Verify the environment

Press **F5** (or use the Run/Debug menu) and select
**"01 - Setup Check"** from the dropdown. Or run directly:

```bash
python scripts/01_setup_check.py
```

You should see all checks pass. If anything fails, the script tells you
exactly what to install.

### 4. Run the synthetic demo (no data download required)

In the Debug menu select **"03 - Synthetic Demo"** and press F5.
Or run:

```bash
python scripts/03_synthetic_demo.py
```

This validates the SAE+WOA pipeline end-to-end on synthetic data in
about 30 seconds. Useful to confirm everything works before downloading
the real dataset.

---

## Project structure

```
TAS-VANET/
├── .vscode/                          # VS Code launch + settings
│   ├── launch.json                  # Run/debug configs for each script
│   ├── settings.json                # Python path + formatting
│   └── extensions.json              # Recommended extensions
├── core/                             # Library package
│   ├── __init__.py                  # Public API exports
│   ├── sae_model.py                 # Stacked Autoencoder (PyTorch)
│   ├── woa_optimizer.py             # Whale Optimization Algorithm
│   ├── trust_pipeline.py            # End-to-end calibration + classify
│   ├── hybrid_feature_extractor.py  # Kinematic + trust feature engineering
│   ├── veremi_loader.py             # VeReMi raw -> pairwise conversion
│   └── synthetic_data.py            # Synthetic data generator (validation)
├── scripts/                          # Executable entry points
│   ├── 01_setup_check.py            # Verify environment + imports
│   ├── 02_process_veremi.py         # Process the VeReMi Extension CSV
│   ├── 03_synthetic_demo.py         # End-to-end demo (no real data)
│   └── 04_explore_processed.py      # Inspect processed CSV statistics
├── data/
│   ├── raw/                         # Put VeReMi_Extension.csv here
│   └── processed/                   # Output of scripts/02 lands here
├── results/
│   ├── models/                      # Trained models (.pt + .json)
│   ├── figures/                     # Generated plots
│   └── tables/                      # CSV tables for the paper
├── docs/
│   └── DOWNLOAD_AND_PROCESS_GUIDE.md
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── .gitignore
```

---

## Workflow

### A. Set up once

```bash
python -m venv .venv
source .venv/bin/activate   # or Windows equivalent
pip install -r requirements.txt
python scripts/01_setup_check.py
```

### B. Download VeReMi Extension (manual, one-time)

See **`docs/DOWNLOAD_AND_PROCESS_GUIDE.md`** for full instructions.

In short:
1. Go to https://data.mendeley.com/datasets/k62n4z9gdz/1
2. Click "Download All", accept license, extract ZIP.
3. Place the CSV inside `data/raw/` and rename to `VeReMi_Extension.csv`.

### C. Process the dataset

```bash
python scripts/02_process_veremi.py \
    --input data/raw/VeReMi_Extension.csv \
    --output data/processed/veremi_processed.csv \
    --sample_fraction 0.1
```

In VS Code: pick **"02 - Process VeReMi (10% sample)"** and press F5.

### D. Explore the processed data

```bash
python scripts/04_explore_processed.py \
    --input data/processed/veremi_processed.csv
```

In VS Code: pick **"04 - Explore Processed Data"** and press F5.

### E. Train + evaluate

(Training script to be added once you have the processed CSV.)

---

## How imports work

The `core/` directory is a regular Python package. Scripts in `scripts/`
add the project root to `sys.path` before importing, so you can run them
directly with `python scripts/XX_name.py` from the project root.

In VS Code, `PYTHONPATH` is set to the workspace root via `.vscode/settings.json`
and `.vscode/launch.json`, so imports work both in the debugger and the
terminal.

If you write your own scripts inside `scripts/`, copy this preamble:

```python
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import SAEConfig, TrustPipeline, ...
```

For notebooks or interactive sessions started from the project root,
imports work without any setup:

```python
from core import HYBRID_FEATURES, TrustPipeline   # works
```

---

## Citation

When using this codebase, cite the datasets we depend on:

```bibtex
@misc{slama2023veremi,
    title  = {VeReMi_Extension: Dataset for Misbehaviors in VANETs},
    author = {Slama, Oumsaad and Tarhouni, Mounira and Zidi, Salah and Alaya, Bechir},
    year   = {2023},
    doi    = {10.17632/k62n4z9gdz.1},
    publisher = {Mendeley Data}
}

@inproceedings{kamel2020veremi,
    title     = {VeReMi Extension: A Dataset for Comparable Evaluation of Misbehavior Detection in VANETs},
    author    = {Kamel, J. and Wolf, M. and van Der Heijden, R. W. and Kaiser, A. and Urien, P. and Kargl, F.},
    booktitle = {IEEE ICC 2020},
    year      = {2020}
}
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'core'`**
You're running a script from outside the project root. Either:
- `cd` to the project root first, then run `python scripts/...`
- Use the VS Code launch configurations (they set `PYTHONPATH` automatically)

**`torch` install hangs or fails**
PyTorch wheels are large. For CPU-only install:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Out-of-memory errors during VeReMi processing**
Use a smaller `--sample_fraction` (try `0.05` or `0.02`) and re-run.

**Need GPU acceleration**
Install the appropriate torch build from https://pytorch.org/get-started/locally/
and confirm with:
```python
import torch
print(torch.cuda.is_available())
```

