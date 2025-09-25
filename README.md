# Code For the paper: "Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs".

## Installation

### Option 1: Using [uv](https://docs.astral.sh/uv/guides/install-python/) (recommended)

```bash
uv sync
```

**Note:** Modify the CUDA version in `pyproject.toml` if needed. The default configuration uses CUDA 12.6. Update the `[[tool.uv.index]]` URL and `torch` source to match your CUDA version.

### Option 2: Using pip

```bash
pip install -r requirements.txt
```

**Note:** Modify the CUDA version in `requirements.txt` if needed. The default PyTorch installation uses CUDA 12.6 (`torch==2.8.0+cu126`).

## Usage

### Step-by-Step Guide

Follow this order to run the complete pipeline:

1. **Generate diffmean files and AUROC results:**
   ```bash
   bash scripts/run_diffmean_analysis.sh
   ```

2. **Train the praise detector model:**
   ```bash
   bash scripts/run_roberta.sh
   ```

3. **Run steering experiments:**
   ```bash
   bash scripts/run_all_steering.sh
   ```

**Optional:** Generate diffmean vectors with subspace removal (as demonstrated in the paper):
```bash
bash scripts/run_nullspace_weight_gen.sh
```
This can be run after step 1 to generate modified diffmean vectors, which can then be used with `run_all_steering.sh` as usual.

### Scripts Overview

- `scripts/run_diffmean_analysis.sh` - Run diffmean analysis and generate diffmean files
- `scripts/run_roberta.sh` - Train RoBERTa praise detector
- `scripts/run_all_steering.sh` - Run complete steering pipeline
- `scripts/run_nullspace_weight_gen.sh` - Generate diffmean vectors with subspace removal

### Individual Components

The main components can be run individually:

- `src/praise_steering.py` - Praise-based steering experiments
- `src/truthful_steering.py` - Truthfulness steering experiments  
- `src/cross_effects.py` - Cross-effects analysis
- `src/diffmean_analysis.py` - Diffmean nullspace projection analysis
- `src/praise_roberta.py` - RoBERTa praise detection model

## Data

The `data/` directory contains:
- `praise.json` - Praise detection training data
- `truthfulqa/` - TruthfulQA train/test split
  - `train.jsonl` - Training set
  - `test.jsonl` - Test set
- `factorial/` - Factorial experimental data

## Requirements

- Python ≥3.11
- CUDA-compatible GPU (recommended)