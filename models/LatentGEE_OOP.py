# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: latentgee
#     language: python
#     name: python3
# ---

# %%
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

import sys, torch, os
import pandas as pd
import numpy as np
import torch, optuna
import torch.nn as nn
import yaml, numbers, math

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from torch.amp import autocast, GradScaler

from LatentGEE import(
    VAE,
    train_vae,
    ziln_nll,
    evaluate_latentgee_u,
    gee_latent_residual,
    decode_batch_corrected_latent,
    permanova_r2
)


# =========================
# Helper 함수
# =========================
def _safe_silhouette(X: np.ndarray, labels: np.ndarray, metric: str = "braycurtis") -> float:
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    if len(uniq) < 2 or np.any(counts < 2):
        return float("nan")
    try:
        return float(silhouette_score(X, labels, metric=metric))
    except Exception:
        return float("nan")

def _eval_block(X: np.ndarray, labels: np.ndarray, sil_metric: str, r2_metric: str,
                permutations: int, standardize: bool) -> Dict[str, float]:
    X_in = np.asarray(X, dtype=np.float64)
    if standardize:
        X_in = StandardScaler().fit_transform(X_in)
    sil = _safe_silhouette(X_in, labels, metric=sil_metric)
    _, r2 = permanova_r2(X_in, grouping=np.asarray(labels), metric=r2_metric, permutations=permutations)
    return {"silhouette": float(sil), "r2": float(r2)}

# ====== 1) YAML 헬퍼 ======    
def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
  
def _as_none(x):
    return None if (isinstance(x, str) and x.lower() == "none") else x

def suggest_auto(trial: optuna.Trial, name: str, spec):
    """YAML spec을 보고 Optuna의 suggest_*를 자동 선택."""
    if isinstance(spec, dict):
        if "loguniform" in spec:
            low, high = spec["loguniform"]
            return trial.suggest_float(name, float(low), float(high))
        raise ValueError(f"Unsupported dict spec for {name}: {spec}")
    if isinstance(spec, (list, tuple)):
        vals = [_as_none(v) for v in spec]
        if len(vals) == 2 and all(isinstance(v, numbers.Number) for v in vals):
            low, high = vals
            if float(low).is_integer() and float(high).is_integer():
                return trial.suggest_int(name, int(low), int(high))
            else:
                return trial.suggest_float(name, float(low), float(high))
        return trial.suggest_categorical(name, vals)
    return spec  # 단일 고정값

# ====== 2) 데이터 로더(Zero prevalence cutoff 캐시 지원) ======
# 전역 캐시 (메모리 or 디스크)
_DATASET_CACHE = {}

def get_dataset_for_cutoff(cutoff: float):
    """
    cutoff별 X_tensor를 CPU 텐서로 반환하고 input_dim을 리턴.
    1) 메모리 캐시 있으면 바로 사용
    2) 없으면 디스크 캐시(파일) 있나 확인 -> 로드
    3) 둘 다 없으면 원시데이터로부터 전처리 수행 -> 캐시/저장
    """
    key = f"zp_{cutoff:.4f}"
    if key in _DATASET_CACHE:
        X = _DATASET_CACHE[key]
        return X, X.shape[1]

    # (A) 디스크 캐시가 있다면:
    pkl_path = f".../preprocessed/hivrc_scene2_zp{cutoff:.2f}.pt"
    if os.path.exists(pkl_path):
        X = torch.load(pkl_path, map_location="cpu")
        _DATASET_CACHE[key] = X
        return X, X.shape[1]

    # (B) 원시데이터로부터 전처리 (여기서는 의사코드)
    # raw = load_raw(...)
    # X_np = preprocess_by_zero_prevalence(raw, cutoff=cutoff)  # np.ndarray (N,D)
    # X = torch.tensor(X_np, dtype=torch.float32)
    # torch.save(X, pkl_path)
    # _DATASET_CACHE[key] = X
    # return X, X.shape[1]
    raise FileNotFoundError(f"no dataset for cutoff={cutoff}; add preprocessing or cached file.")

# =========================
# Configs
# =========================
@dataclass
class DataConfig:
    zero_prevalence_cutoff: list[float]
    standardize_latent: list[bool]
    tensor_path: str

# 모델 탐색 공간
@dataclass
class ModelSearchSpace:
    n_layers: list[int]
    latent_dim: list[int]
    base_dim: list[int]
    strategy: list[str]
    dropout_rate: list[float]
    beta_kl: list[float]
    kl_warmup_epochs: list[int]
    norm: list[str]
    init: list[str]

@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 256
    num_workers: int = max(2, (os.cpu_count() or 4) - 2)
    amp: bool = field(default_factory=lambda: torch.cuda.is_available())

@dataclass
class ModelConfig:
    input_dim: int
    latent_dim: int = 16
    n_layers: int = 2
    base_dim: int = 128
    strategy: str = "constant"   # {'constant','halve','double'}
    dropout: float = 0.0
    activation: str = "relu"

@dataclass
class EvalConfig:
    # HDBSCAN pseudo-batch
    hdb_min_cluster_size: int = 10
    hdb_min_samples: Optional[int] = None
    hdb_metric: str = "braycurtis"  # e.g. 'euclidean','cosine','braycurtis' (하이픈 X)
    allow_noise: bool = True
    # PERMANOVA
    permanova_metric: str = "braycurtis"
    permanova_permutations: int = 999

@dataclass
class TuningConfig:
    n_trials: int = 25
    timeout: Optional[int] = None
    pruner: bool = True

# =========================
# 학습 탐색 공간
# =========================
@dataclass
class TrainingSearchSpace:
    epochs: list[int]
    batch_size: list[int]
    learning_rate: dict
    weight_decay: dict
    optimizer: list[str]
    amp: list[bool]
    grad_clip_norm: list[float]
    scheduler: list[str]
    warmup_epochs: list[int]

# 클러스터링 탐색 공간
@dataclass
class ClusteringSearchSpace:
    min_cluster_size: list[int]
    min_samples: list[Any]     # "None"이 문자열로 들어오므로 Any
    metric: list[str]
    cluster_selection_method: list[str]
    allow_single_cluster: list[bool]

# 평가 옵션
@dataclass
class EvaluationConfig:
    silhouette_metric: list[str]
    noise_handling: list[str]

# Safeguards
@dataclass
class SafeguardsConfig:
    eps: float
    max_oom_retries: int

# 전체 config wrapper
@dataclass
class FullConfig:
    data: DataConfig
    search_space: dict
    safeguards: SafeguardsConfig

# =========================
# Data Module
# =========================
class DataModule:
    def __init__(self, X: np.ndarray, train_cfg: TrainConfig):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        self.X = torch.tensor(X, dtype=torch.float32)
        self.train_cfg = train_cfg

    def make_loader(self) -> DataLoader:
        ds = TensorDataset(self.X.cpu())
        return DataLoader(
            ds,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=self.train_cfg.num_workers,
            prefetch_factor=4 if self.train_cfg.num_workers > 0 else None,
            persistent_workers=(self.train_cfg.num_workers > 0),
            pin_memory=(self.train_cfg.device == "cuda"),
            drop_last=True,
        )
# =========================
# LatentGEEModule (Model + Train loop)
# =========================
class LatentGEEModule(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        # VAE 객체 생성 (LatentGEE.py에 있는 VAE 클래스) 
        self.vae = VAE(
            input_dim=model_cfg.input_dim,
            latent_dim=model_cfg.latent_dim,
            n_layers=model_cfg.n_layers,
            base_dim=model_cfg.base_dim,
            strategy=model_cfg.strategy,
            dropout_rate=model_cfg.dropout,
            activation=model_cfg.activation,
        )

    def forward(self, x):
        """
        VAE forward 호출 (encode -> reparameterize -> decode)
        """
        return self.vae(x)

    def fit(self, loader: DataLoader, train_cfg: TrainConfig) -> float:
        """
        주어진 DataLoader로 VAE 학습 수행 (ZILN NLL + KL 손실).
        마지막 배치의 loss(float)를 반환.
        """
        device = train_cfg.device
        self.to(device)
        self.train()

        opt = torch.optim.Adam(self.parameters(), lr=train_cfg.lr)
        use_amp = train_cfg.amp and (device == "cuda")
        scaler = GradScaler(device="cuda", enabled=use_amp)

        last_loss = float("inf")

        for ep in range(train_cfg.epochs):
            for (xb,) in loader:
                xb = xb.to(device, non_blocking=use_amp)
                with autocast(device_type="cuda", enabled=use_amp):
                    (pi, mu_x, logsigma_x), mu_z, logvar_z, _ = self.vae(xb)
                    recon_nll = ziln_nll(xb, pi, mu_x, logsigma_x)
                    kl = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
                    loss = recon_nll + kl

                if torch.isnan(loss):
                    return float("inf")

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                last_loss = float(loss.detach().cpu().item())

        return last_loss

    @torch.no_grad()
    def encode_mu(self, X_tensor: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
        """입력 X를 μ(latent mean)으로 인코딩."""
        self.eval()
        dev = device or next(self.parameters()).device
        mu, _ = self.vae.encode(X_tensor.to(dev, non_blocking=(dev.type == "cuda")))
        return mu.detach().cpu()

    @torch.no_grad()
    def decode_from_latent(self, z: np.ndarray) -> np.ndarray:
        """latent z에서 X를 디코딩 (ZILN 기대값으로 복원)."""
        self.eval()
        device = next(self.parameters()).device
        z_tensor = torch.tensor(z, dtype=torch.float32, device=device)
        pi, mu_x, log_sigma_x = self.vae.decode(z_tensor)
        sigma = log_sigma_x.exp()
        x_hat = (1 - pi) * torch.exp(mu_x + 0.5 * sigma * sigma)
        return x_hat.detach().cpu().numpy()


# -----------------------
# Evaluator (silhouette / PERMANOVA R²)
# -----------------------
class BatchEffectEvaluator:
    def __init__(self, eval_cfg: EvalConfig):
        self.cfg = eval_cfg

    @torch.no_grad()
    def silhouette_from_model(self, model: LatentGEEModule, X: np.ndarray) -> Tuple[np.ndarray, float]:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        labels, sil = evaluate_latentgee_u(
            model=model.vae,
            X_tensor=X_tensor,
            min_cluster_size=self.cfg.hdb_min_cluster_size,
            min_samples=self.cfg.hdb_min_samples,
            allow_noise=self.cfg.allow_noise,
            metric=self.cfg.hdb_metric,
        )
        return labels, float(sil)

    def permanova_r2_from_matrix(self, X: np.ndarray, labels: np.ndarray) -> Tuple[Any, float]:
        X_std = StandardScaler().fit_transform(X)
        res, r2 = permanova_r2(
            X_std,
            grouping=labels,
            metric=self.cfg.permanova_metric,
            permutations=self.cfg.permanova_permutations,
        )
        return res, float(r2)
    
# -----------------------
# Corrector (latent GEE residual → decode)
# -----------------------
class BatchCorrector:
    @torch.no_grad()
    def correct_and_decode(
        self,
        model: LatentGEEModule,
        X: np.ndarray,
        labels: np.ndarray,                    # ← 반드시 외부에서 라벨을 받아옴 (pseudo-batch or real batch)
        save_path: Optional[str] = None
    ) -> np.ndarray:
        # 1) encode μ
        X_tensor = torch.tensor(X, dtype=torch.float32)
        mu = model.encode_mu(X_tensor)                        # torch.Tensor
        mu_np = mu.detach().cpu().numpy().astype("float32")   # np.ndarray (N,k)

        # 2) residualize (labels는 np.ndarray)
        z_tilde = gee_latent_residual(mu_np, labels)          # np.ndarray (N,k)

        # 3) decode
        x_corr = model.decode_from_latent(z_tilde)            # np.ndarray (N,D)
        if save_path:
            np.save(save_path, x_corr)
        return x_corr
# -----------------------
# Optuna Tuner
# -----------------------
class OptunaTuner:
    def __init__(self, X: np.ndarray, base_model_cfg: ModelConfig, train_cfg: TrainConfig, eval_cfg: EvalConfig):
        self.X = X
        self.base_model_cfg = base_model_cfg
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg

    def _suggest(self, trial: optuna.Trial) -> ModelConfig:
        # 예시: latent_dim, base_dim, n_layers, dropout, activation 탐색
        latent_dim = trial.suggest_int("latent_dim", 8, 64, step=8)
        base_dim   = trial.suggest_categorical("base_dim", [64, 128, 256, 512])
        n_layers   = trial.suggest_int("n_layers", 1, 3)
        dropout    = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
        activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "gelu"])
        strategy   = trial.suggest_categorical("strategy", ["constant", "halve", "double"])
        return ModelConfig(
            input_dim=self.base_model_cfg.input_dim,
            latent_dim=latent_dim,
            n_layers=n_layers,
            base_dim=base_dim,
            strategy=strategy,
            dropout=dropout,
            activation=activation,
        )
    
    def objective(self, trial: optuna.Trial) -> float:
        # 단일 목표: pseudo-batch silhouette (↓)
        model_cfg = self._suggest(trial)
        dm = DataModule(self.X, self.train_cfg)
        loader = dm.make_loader()

        model = LatentGEEModule(model_cfg)
        _ = model.fit(loader, self.train_cfg)

        evaluator = BatchEffectEvaluator(self.eval_cfg)
        _, sil = evaluator.silhouette_from_model(model, self.X)

        trial.report(float(sil), step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return float(sil)

    def tune(self, tune_cfg: TuningConfig) -> Tuple[optuna.Study, Dict[str, Any]]:
        pruner = optuna.pruners.MedianPruner() if tune_cfg.pruner else optuna.pruners.NopPruner()
        study = optuna.create_study(direction="minimize", pruner=pruner)  # silhouette ↓
        study.optimize(self.objective, n_trials=tune_cfg.n_trials, timeout=tune_cfg.timeout)
        return study, study.best_params

    def objective_multi(self, trial: optuna.Trial, y_bio: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Multi-objective: (sil_batch ↓, r2_batch ↓, sil_bio ↑, r2_bio ↑)
        """
        model_cfg = self._suggest(trial)
        dm = DataModule(self.X, self.train_cfg)
        loader = dm.make_loader()

        model = LatentGEEModule(model_cfg)
        _ = model.fit(loader, self.train_cfg)

        # batch metrics
        evaluator = BatchEffectEvaluator(self.eval_cfg)
        labels_batch, sil_batch = evaluator.silhouette_from_model(model, self.X)

        X_std = StandardScaler().fit_transform(self.X)
        _, r2_batch = permanova_r2(
            X_std, grouping=labels_batch,
            metric=self.eval_cfg.permanova_metric,
            permutations=self.eval_cfg.permanova_permutations,
        )

        # bio metrics (encoder μ 기준)
        with torch.no_grad():
            mu = model.encode_mu(torch.tensor(self.X, dtype=torch.float32, device=self.train_cfg.device))
        Z = mu.detach().cpu().numpy()
        Z_std = StandardScaler().fit_transform(Z)

        sil_bio = silhouette_score(Z_std, y_bio, metric="euclidean")
        _, r2_bio = permanova_r2(
            Z_std, grouping=y_bio,
            metric=self.eval_cfg.permanova_metric,
            permutations=self.eval_cfg.permanova_permutations,
        )

        return float(sil_batch), float(r2_batch), float(sil_bio), float(r2_bio)


# =========================
# Pipeline
# =========================
class LatentGEEPipeline:
    def __init__(self, X: np.ndarray, model_cfg: ModelConfig, train_cfg: TrainConfig, eval_cfg: EvalConfig):
        self.X = X
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.model: Optional[LatentGEEModule] = None

    def fit(self):
        print("[LatentGEEPipeline] Starting: preparing model training")
        print(f"  - Data size: {self.X.shape}, batch_size={self.train_cfg.batch_size}, epochs={self.train_cfg.epochs}")
        print("[LatentGEEPipeline] Training started...")
        
        # DataModule 시그니처: (X, train_cfg)
        dm = DataModule(self.X, self.train_cfg)
        loader = dm.make_loader()
        self.model = LatentGEEModule(self.model_cfg)
        self.model.fit(loader, self.train_cfg)
        print("[LatentGEEPipeline] Training finished")

    @torch.no_grad()
    def encode_mu(self) -> np.ndarray:
        """학습된 모델로 X를 μ(latent mean)로 인코딩해서 numpy(float32)로 반환."""

        print("[LatentGEEPipeline] Encoding: computing latent mean (μ)")
        
        assert self.model is not None, "Call fit() first."
        self.model.eval()
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        mu = self.model.encode_mu(X_tensor)                # torch.Tensor on CPU (우리 구현 기준)
        mu_np = mu.detach().cpu().numpy().astype("float32")
        print(f"[LatentGEEPipeline] Encoding finished: shape={mu_np.shape}")
        return mu_np

    def evaluate(self) -> Dict[str, Any]:
        """배치 지표 평가 (pseudo-batch silhouette & PERMANOVA R²)."""
        print("[LatentGEEPipeline] Evaluation started: pseudo-batch silhouette & PERMANOVA R²")

        assert self.model is not None, "Call fit() first."
        evaluator = BatchEffectEvaluator(self.eval_cfg)
        labels, sil = evaluator.silhouette_from_model(self.model, self.X)
        res, r2 = evaluator.permanova_r2_from_matrix(self.X, labels)
        print(f"[LatentGEEPipeline] Evaluation finished: silhouette={sil:.4f}, R²={r2:.4f}")
        return {"silhouette_batch": sil, "permanova_r2_batch": r2, "labels": labels, "permanova": res}

    @torch.no_grad()
    def correct_and_decode(self, save_path: Optional[str] = None) -> np.ndarray:
        """
        1) μ 인코딩 → 2) pseudo-batch 라벨 추정 → 3) latent residualize → 4) decode → 5) (선택) 저장
        """
        print("[LatentGEEPipeline] Correction + decoding started")
        print("  1) Encoding latent mean (μ)...")
        assert self.model is not None, "Call fit() first."
        self.model.eval()

        # 1) μ 인코딩
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        mu = self.model.encode_mu(X_tensor)                # torch.Tensor on CPU
        mu_np = mu.detach().cpu().numpy().astype("float32")

        # 2) 라벨 추정 (evaluate_latentgee_u는 model.vae를 받도록 설계됨)
        print("  2) Estimating pseudo-batch labels...")        
        labels, _ = evaluate_latentgee_u(
            model=self.model.vae,
            X_tensor=X_tensor,
            min_cluster_size=self.eval_cfg.hdb_min_cluster_size,
            min_samples=self.eval_cfg.hdb_min_samples,
            allow_noise=True,
            metric=self.eval_cfg.hdb_metric,
        )

        # 3) residualize
        print("  3) Running GEE residualization (removing batch effect)...")
        z_tilde = gee_latent_residual(mu_np, labels)       # np.ndarray (N,k), LatentGEE.py 시그니처 기준

        # 4) decode
        print("  4) Decoding corrected latent back to data space...")
        x_corr = self.model.decode_from_latent(z_tilde)    # np.ndarray (N,D)

        # 5) 저장(옵션)
        if save_path:
            np.save(save_path, x_corr)
            print(f"[LatentGEEPipeline] Corrected data saved to {save_path}")

        return x_corr
    
    @torch.no_grad()
    def correct_decode_and_evaluate_strict(
        self,                                   # LatentGEEPipeline 인스턴스로 바인딩해서 사용
        *,
        original_batch_labels: np.ndarray,       # 입력 데이터의 "실제" 배치 라벨 (필수, N,)
        original_bio_labels:   np.ndarray,       # 입력 데이터의 "실제" 바이오/임상 라벨 (필수, N,)
        silhouette_metric: str = "braycurtis",
        r2_metric: str = "braycurtis",
        permutations: int = 999,
        standardize: bool = True,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        흐름:
        1) 원본 X에서 pseudo-batch 라벨 생성 (HDBSCAN→GEE-residual→silhouette 유틸 재사용; 항상 수행)
        2) before_batch: 원본 X에서 pseudo-batch 라벨 기준 평가
        3) X를 보정: pseudo-batch를 그룹 변수로 GEE 적합, y_hat에서 residual을 빼 batch effect 제거(= gee_latent_residual)
        4) 디코드 → X_corr
        5) after_batch: X_corr에서 input데이터의 original batch 라벨 기준 평가
        6) after_bio  : X_corr에서 input데이터의 original bio 라벨 기준 평가

        반환:
        {
            'before_batch': {...},   # (pseudo-batch 기준) ↓가 좋음
            'after_batch':  {...},   # (original batch 기준) ↓가 좋음
            'after_bio':    {...},   # (original bio 기준)   ↑가 좋음
            'X_corr':       np.ndarray
        }
        """
        print("[LatentGEEPipeline] Strict evaluation started")
        assert self.model is not None, "Call fit() first."
        self.model.eval()

        # 1) pseudo-batch 생성 (항상 수행)
        print("  1) Generating pseudo-batch (HDBSCAN)...")
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        pseudo_labels, _ = evaluate_latentgee_u(
            model=self.model.vae,   # 주의: wrapper가 아닌 내부 VAE를 넘김
            X_tensor=X_tensor,
            min_cluster_size=self.eval_cfg.hdb_min_cluster_size,
            min_samples=self.eval_cfg.hdb_min_samples,
            allow_noise=self.eval_cfg.allow_noise,
            metric=self.eval_cfg.hdb_metric,
        )

        # 2) BEFORE (원본 X, pseudo-batch 기준 평가)
        print("  2) BEFORE evaluation (pseudo-batch based)...")
        before_batch = _eval_block(
            self.X, pseudo_labels,
            sil_metric=silhouette_metric, r2_metric=r2_metric,
            permutations=permutations, standardize=standardize
        )

        # 3) 보정(잠재공간): pseudo-batch를 그룹으로 GEE 적합 → batch 효과 제거
        #   - 구현은 LatentGEE.py의 gee_latent_residual이 수행(= y_hat - residual과 동치인 batch 효과 제거 결과)
        print("  3) Performing correction (latent residualization)...")
        mu = self.model.encode_mu(X_tensor)                              # torch.Tensor (N,k)
        mu_np = mu.detach().cpu().numpy().astype("float32")              # np.ndarray (N,k)
        z_tilde = gee_latent_residual(mu_np, pseudo_labels)              # np.ndarray (N,k), batch-corrected latent

        # 4) 디코드 → X_corr
        print("  4) Decoding corrected latent to data space...")
        X_corr = self.model.decode_from_latent(z_tilde)                  # np.ndarray (N,D)
        if save_path:
            np.save(save_path, X_corr)

        # 5) AFTER: 보정된 X에서 "original batch" 기준 평가
        print("  5) AFTER evaluation (original batch labels)...")
        after_batch = _eval_block(
            X_corr, np.asarray(original_batch_labels),
            sil_metric=silhouette_metric, r2_metric=r2_metric,
            permutations=permutations, standardize=standardize
        )

        # 6) AFTER: 보정된 X에서 "original bio" 기준 평가
        print("  6) AFTER evaluation (original bio labels)...")
        after_bio = _eval_block(
            X_corr, np.asarray(original_bio_labels),
            sil_metric=silhouette_metric, r2_metric=r2_metric,
            permutations=permutations, standardize=standardize
        )
        print("[LatentGEEPipeline] Strict evaluation finished")
        return {
            "before_batch": before_batch,
            "after_batch":  after_batch,
            "after_bio":    after_bio,
            "X_corr":       X_corr,
        }


# =========================
# Helper: retrain → residualize → decode → save
# Optuna 결과를 최종 적용해서 “공식 corrected 데이터셋” 을 뽑을 때 필요한 함수
# 그냥 파이프라인 내부에서 학습하고 바로 평가만 한다면 → 불필요 (중복 기능).
# 최종 논문/실험에서 “best_params로 retrain 후 결과 저장” 프로세스를 쓸 거라면 그대로 두는 게 맞음
# =========================
def retrain_encode_residual_decode_save(
    X: np.ndarray,
    best_params: Dict[str, Any],
    eval_cfg: EvalConfig,
    save_path: str = "X_corrected.npy",
):
    # 1) 모델 구성
    model = VAE(
        input_dim=X.shape[1],
        latent_dim=best_params.get("latent_dim", 16),
        n_layers=best_params.get("n_layers", 2),
        base_dim=best_params.get("base_dim", 128),
        strategy=best_params.get("strategy", "constant"),
        dropout_rate=best_params.get("dropout", 0.0),
        activation=best_params.get("activation", "relu"),
    )

    # 2) 학습 (ZILN NLL + KL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ = train_vae(
        model,
        torch.tensor(X, dtype=torch.float32),
        epochs=best_params.get("epochs", 50),
        lr=best_params.get("lr", 1e-3),
        device=device,
        batch_size=best_params.get("batch_size", 256),
    )

    # 3) μ 인코딩
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(torch.tensor(X, dtype=torch.float32, device=next(model.parameters()).device))
    mu_np = mu.detach().cpu().numpy().astype("float32")

    # 4) 라벨 추정 → residualize
    labels, _ = evaluate_latentgee_u(
        model=model,
        X_tensor=torch.tensor(X, dtype=torch.float32),
        min_cluster_size=eval_cfg.hdb_min_cluster_size,
        min_samples=eval_cfg.hdb_min_samples,
        allow_noise=True,
        metric=eval_cfg.hdb_metric,
    )
    z_tilde = gee_latent_residual(mu_np, labels)  # np.ndarray (N,k)

    # 5) 디코드 + 저장
    x_corrected = decode_batch_corrected_latent(model, z_tilde, save_path=save_path)
    return x_corrected, labels
