**Unsupervised latent-space batch effect correction via VAE and GEE residualization.**

LatentGEE is a Python framework for detecting and correcting batch effects in high-dimensional biological data.
It combines:

* Variational Autoencoders (VAE) for latent representation learning
* Unsupervised pseudo-batch estimation in latent space
* Generalized Estimating Equations (GEE) for residual-based batch correction

The method is designed for settings where true batch labels are partially known or unknown.

---

## 🧪 Development Status

This repository is currently research-focused and under active development.
Interface changes may occur before first stable release.

---

## 🔬 Method Overview

The correction pipeline consists of:

1. Learn latent representation using a VAE
2. Estimate pseudo-batch structure via clustering in latent space
3. Fit a GEE model using pseudo-batch (and optional clinical covariates)
4. Remove only the pseudo-batch component
5. Decode corrected latent representation back to data space

This approach allows preservation of biological signal while reducing unwanted batch variation.

---

## 📦 Installation

### 1. Clone repository

```bash
git clone https://github.com/<your-username>/LatentGEE.git
cd LatentGEE
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Quick Start

Minimal example:

```python
import numpy as np
from latentgee.pipeline.runner import LatentGEEPipeline
from latentgee.config.schema import ModelConfig, TrainConfig, EvalConfig

X = np.random.rand(200, 100)

model_cfg = ModelConfig(input_dim=100)
train_cfg = TrainConfig(epochs=50)
eval_cfg = EvalConfig()

pipeline = LatentGEEPipeline(X, model_cfg, train_cfg, eval_cfg)

pipeline.fit()
results = pipeline.evaluate()

X_corrected = pipeline.correct_and_decode()
```

---

## ⚙️ Configuration

Experiments are controlled via `config.yaml`.

Example:

```yaml
seed: 42

data:
  zero_prevalence_cutoff: 0.1
  standardize_latent: true

training:
  epochs: 100
  batch_size: 256
  learning_rate: 1e-3
```

To run an experiment:

```bash
python experiments/run_scene2.py --config config.yaml
```

---

## 🔁 Reproducibility
All experiments were run with:
- Python 3.10
- Torch 2.1.0
- CUDA 12.1

To reproduce:
1. Install dependencies from requirements.txt
2. Set seed in config.yaml
3. Run experiments/run_scene2.py

## 💾 Model Saving & Loading

```python
from latentgee.pipeline.checkpoint import save_model, load_model

save_model(model, "model.pt")

model = load_model(model, "model.pt")
```

Saved checkpoints include:

* model state_dict
* configuration
* training parameters
* seed

---

## 📊 Evaluation Metrics

LatentGEE supports:

* Silhouette score
* PERMANOVA R²
* Before/after batch comparison
* Biological signal preservation analysis

---


