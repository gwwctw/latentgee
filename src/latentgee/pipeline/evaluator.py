import os
import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
import warnings

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
from .utils import gee_latent_residual
from typing import Tuple
from datetime import date



def evaluate_batch_effect_pipeline(
# def evaluate_latentgee(
    model: torch.nn.Module,
    X_tensor: torch.Tensor,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    allow_noise: bool = True,          # True면 노이즈 포함, False면 -1 제외
    metric: str = "braycurtis",
) -> Tuple[np.ndarray, float]:
    """
    Encoder의 μ → HDBSCAN으로 pseudo-batch → GEE residual → silhouette 반환.
    labels: np.ndarray (길이 N, -1 = noise)
    score : float (noise 포함/제외는 allow_noise에 따름; 실패 시 -1.0)
    """
    # 1) 인코딩
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        mu, _ = model.encode(X_tensor.to(device, non_blocking=(device.type == "cuda")))
    z_np = mu.detach().cpu().numpy().astype("float32")

    # (선택) 표준화: HDBSCAN/실루엣 안정화
    z_np = StandardScaler().fit_transform(z_np)

    # 2) HDBSCAN (문자열 "None" 방지)
    if isinstance(min_samples, str) and min_samples.lower() == "none":
        min_samples = None

    hdb_kwargs = dict(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
    )
    try:
        # cosine은 BallTree 미지원 → generic로 시도
        if metric == "cosine":
            hdb = HDBSCAN(algorithm="generic", **hdb_kwargs)
        else:
            hdb = HDBSCAN(**hdb_kwargs)
        labels = hdb.fit_predict(z_np)
    except ValueError:
        warnings.warn(
            f"[WARN] HDBSCAN metric '{metric}' not supported → fallback to 'euclidean'",
            RuntimeWarning,
        )
        hdb = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = hdb.fit_predict(z_np)

    # 3) 사용 샘플 선택 (allow_noise 의미 정정)
    if allow_noise:
        mask = np.ones_like(labels, dtype=bool)   # 노이즈 포함
    else:
        mask = (labels != -1)                     # 노이즈 제외

    n_valid_clusters = np.unique(labels[mask]).size
    n_valid_points   = int(mask.sum())

    # 4) HDBSCAN 결과가 부적절하면 KMeans 폴백(= best-k 탐색 대신 k=2라도 OK)
    #    → best-k를 쓰고 싶으면 별도 함수 호출로 교체 가능
    if n_valid_clusters < 2 or n_valid_points < 3:
        # KMeans로 강제 라벨링 (원하면 find_best_k_silhouette로 교체)
        if z_np.shape[0] >= 2:
            km = KMeans(n_clusters=2, n_init="auto", random_state=42).fit(z_np)
            labels = km.labels_
            # KMeans 라벨엔 노이즈 개념 없음 → 전부 사용
            z_used, lbl_used = z_np, labels
        else:
            return labels, -1.0
    else:
        # 마스크에 따라 사용 데이터 결정
        z_used, lbl_used = z_np[mask], labels[mask]

    # 5) GEE residual → silhouette
    z_resid = gee_latent_residual(z_used, lbl_used)  # -1 라벨은 위에서 제거됨
    try:
        sil = silhouette_score(z_resid, lbl_used, metric=metric)
    except Exception:
        # 거리 메트릭 이슈 → euclidean 폴백
        try:
            sil = silhouette_score(z_resid, lbl_used, metric="euclidean")
        except Exception:
            sil = -1.0

    return labels, float(sil)
