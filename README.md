# latentgee
Pseudo-batch estimation in latent space with GEE-based residualization for batch effect removal.

좋아 👍
논문 제출 + GitHub 공개 + 내부 배포까지 모두 커버되는 **연구용 패키지 README 템플릿**을 깔끔하게 작성해줄게.
(과장 없이, 재현성 강조, 리뷰어 친화적 톤으로 구성)

---

# 📄 README.md 템플릿

아래 그대로 복붙해서 시작해도 되고, 이름만 바꿔도 돼.

---

# LatentGEE

**Unsupervised latent-space batch effect correction via VAE and GEE residualization.**

LatentGEE is a Python framework for detecting and correcting batch effects in high-dimensional biological data.
It combines:

* Variational Autoencoders (VAE) for latent representation learning
* Unsupervised pseudo-batch estimation in latent space
* Generalized Estimating Equations (GEE) for residual-based batch correction

The method is designed for settings where true batch labels are partially known or unknown.

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

All experiments should specify:

* Python version
* PyTorch version
* CUDA version (if applicable)
* Random seed

To ensure deterministic behavior:

```python
from latentgee.utils.seed import set_seed
set_seed(42)
```

The repository has been tested with:

* Python 3.10
* Torch 2.1.0
* CUDA 12.1

Dependencies are listed in `requirements.txt`.

---

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

## 📁 Repository Structure

```
latentgee/
    core/        # Model definitions and statistical methods
    pipeline/    # Training and execution orchestration
    config/      # Configuration schemas
experiments/     # Reproducible experiment scripts
```

---

## 📚 Citation

If you use this repository in your research, please cite:

```
<To be updated after publication>
```

---

## 🧪 Development Status

This repository is currently research-focused and under active development.
Interface changes may occur before first stable release.

---

## 📝 License

Specify license here (e.g., MIT License).

---

## 🤝 Contact

For questions or collaboration:

*  Name
*  Institution
*  Email




🔥 가장 중요한 설계 원칙

core는 절대 pipeline을 몰라야 한다.
core의 역할은: 알고리즘 그 자체 (VAE, loss 함수, GEE residual, clustering, metric,... 모델 정의와 수학적 연산!)
pipeline은 core를 import해도 되지만
core가 pipeline을 import하면 구조가 무너진다.

🔎 판단 기준 하나만 기억해

이 질문을 던져봐:

이 코드는 모델의 본질인가,
아니면 모델을 어떻게 돌릴지에 대한 전략인가?

core/        ← 본질/ 알고리즘 (VAE, GEE, clustering)
    ↑
pipeline/    ← 실행 orchestration/실행 전략 
    ↑
cli.py       ← 엔트리 포인트

📦 가장 이상적인 구조
core/
    [v]networks.py   ← FlexibleMLP
    [v]vae.py        ← VAE만
    [v]losses.py
    [v]gee.py
    [ ]clustering.py
    [ ]evaluation.py
config/
    [v]schema.py        ← dataclass 정의
    [ ]loader.py        ← yaml → dataclass 매핑
    [v]searchspace.py   ← optuna suggest 로직
pipeline/ # "학습 실행 전략 (training orchestration)"
    trainer.py      ← LatentGEEModule 여기
    corrector.py
    evaluator.py
    [v]runner.py       <-- LatentGEEPipeline
    [v]checkpoint.py
    tuner.py


🎯 3️⃣ 논문 supplementary 관점에서 중요한 것

리뷰어가 보는 건 3가지야:

① 재현 가능성

config.yaml 포함

random seed 고정

requirements.txt 포함

② 구조의 명확성

core / pipeline 분리

함수 docstring 충분히 작성

③ 실행 예제 제공
experiments/run_example.py
이 파일이 정말 중요함.

🎯 5️⃣ 가장 현실적인 로드맵

지금:

GitHub + 논문 supplementary용 clean repo 만들기

논문 accept 후:

pip 배포

이게 자연스러운 순서야.

🎯 6️⃣ 그럼 지금 당장 꼭 해야 할 것

✔ sys.path.append 제거
✔ requirements.txt 작성
✔ README에 최소 실행 예시 추가
✔ seed 고정 옵션 추가
✔ 모델 저장/로드 구조 정리