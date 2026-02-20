import numpy as np
import torch
from typing import Dict, Any, Optional
from latentgee.config.schema import ModelConfig, TrainConfig, EvalConfig

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