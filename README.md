# LatentGEE

**Unsupervised latent-space batch effect correction via VAE and GEE residualization.**

LatentGEE is a research framework for detecting and correcting batch effects in high-dimensional biological data — particularly microbiome compositional data — in settings where true batch labels are **partially known or entirely unavailable**.

> 🔬 Work in progress. This repository accompanies a poster to be presented at **ISMB 2026**.

---

## Development Status

This repository is currently research-focused and under active development.
Interface changes may occur before first stable release.

## Method Overview

LatentGEE combines three components into an end-to-end correction pipeline:

```
Input X
   ↓
VAE (ZILN loss)         — latent representation learning
   ↓
HDBSCAN                 — unsupervised pseudo-batch estimation in latent space
   ↓
GEE residualization     — remove pseudo-batch + covariate effects from z
   ↓
Decode z̃               — corrected data in original space
```

**Key design choices:**
- Zero-Inflated Log-Normal (ZILN) reconstruction loss for sparse compositional data
- Density-based pseudo-batch estimation (HDBSCAN) — no batch labels required
- GEE with Exchangeable correlation structure for within-cluster residualization
- Optional adjustment for clinical covariates during GEE step
- Hyperparameter optimization via Optuna (PERMANOVA R² as evaluation metric)

---

## Repository Structure

```
latentgee/
├── examples/
│   ├── prototype.py       # Main training & evaluation script
│   └── config.yaml        # Hyperparameter search space & tuning settings
├── src/                   # Core modules (in development)
├── models/                # Saved model checkpoints
├── experiments/           # Experiment scripts
├── archive/               # Previous iterations
├── core.py
└── README.md
```

---

## Installation

```bash
git clone https://github.com/gwwctw/latentgee.git
cd latentgee
pip install -r requirements.txt
```

**Dependencies:** `torch`, `optuna`, `hdbscan`, `statsmodels`, `scikit-bio`, `scikit-learn`, `pandas`, `numpy`, `pyyaml`

---

## Quick Start

See [`examples/prototype.py`](examples/prototype.py) and [`examples/config.yaml`](examples/config.yaml) for a full working example using the HIVRC microbiome dataset.

To run:

```bash
python examples/prototype.py
```

---

## Configuration

Experiments are controlled via `config.yaml`. Key sections:

```yaml
data:
  zero_prevalence_cutoff: [0.005, 0.01, 0.03, 0.05, 0.07, 0.10]  # treated as hyperparameter

search_space:
  model:
    latent_dim: [8, 64]
    beta_kl: [0.01, 0.8]
    kl_warmup_ratio: [0.2, 0.8]
  clustering:
    min_cluster_size: [3, 20]

tuning:
  n_trials: 5000
  direction: minimize      # PERMANOVA R² (Study-based)
  pruner: false
```

---

## Evaluation

LatentGEE uses **PERMANOVA R²** (based on true batch labels, e.g. Study) computed on the GEE-residualized latent space as the Optuna objective:

- Lower R² → batch effect better removed
- Evaluated on `z̃` (post-GEE residual), not raw latent `z`
- Permutations set to 99 for efficiency during hyperparameter search

---

## Development Status

| Component | Status |
|---|---|
| VAE + ZILN loss | ✅ Complete |
| HDBSCAN pseudo-batch | ✅ Complete |
| GEE residualization | ✅ Complete |
| Optuna hyperparameter tuning | ✅ Complete |
| Baseline comparison (ComBat, Harmony, scVI) | 🔄 Planned |
| Ablation study | 🔄 Planned |
| Package release | 🔄 Planned |

---
