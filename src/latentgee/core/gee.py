import yaml
import numpy as np
import pandas as pd
import optuna
import os
import warnings


def gee_residual(
    z_tensor: torch.Tensor,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    noise_handling: str = "global",  # {"global", "self"}
):
    """
    ▣ 역할
        1. HDBSCAN → pseudo-batch 라벨 생성
        2. 배치(클러스터)마다 각 latent 차원의 평균을 빼서
           batch-free 잔차 \tilde{z} 반환

    ▣ 매개변수
        z_tensor (N,k) : μ 또는 z (torch.float32/64)
        min_cluster_size / min_samples : HDBSCAN 하이퍼파라미터
        noise_handling :
            "global"  → noise(-1) 샘플은 전체 평균을 빼줌
            "self"    → noise 샘플은 그대로 두거나 개별 보정

    ▣ 반환값
        z_tilde   (N,k, torch.Tensor)  : batch 효과 제거된 latent
        labels    (N,   np.ndarray)    : HDBSCAN cluster 라벨 (-1 = noise)
    """
    if z_tensor.requires_grad:
        z_tensor = z_tensor.detach()

    z_cpu = z_tensor.cpu().numpy()

    # ── 1. Pseudo-batch 탐색 ─────────────────────────────
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",                     # cosine이면 아래 4-2 참고
        core_dist_n_jobs=os.cpu_count(),    # 코어 모두 사용
        approx_min_span_tree=True,          # 근사 MST로 속도↑
        algorithm="best"
    )
    labels = hdb.fit_predict(z_cpu)          # shape (N,)

    # ── 2. 배치별 평균 제거 (= GEE intercept) ───────────
    z_tilde = z_tensor.clone()
    labels_unique = np.unique(labels)

    for g in labels_unique:
        idx_np = (labels == g)
        idx = torch.from_numpy(idx_np).to(device=z_tensor.device, dtype=torch.bool)  # ← 핵심
        if g == -1:
            if noise_handling == "global":
                mean_vec = z_tensor.mean(dim=0, keepdim=True)
                z_tilde[idx] -= mean_vec
            elif noise_handling == "self":
                pass
        else:
            mean_vec = z_tensor[idx].mean(dim=0, keepdim=True)
            z_tilde[idx] -= mean_vec

    return z_tilde, labels

# --- 1. 함수 정의: prevalence cutoff에 따른 zero 비율 계산 ---
def compute_zero_proportion_by_prevalence(otu_df, cutoffs):
    n_samples = otu_df.shape[0]
    proportions = []

    for cutoff in cutoffs:
        # prevalence 계산: 각 OTU의 nonzero 비율
        prevalence = (otu_df > 0).sum(axis=0) / n_samples

        # cutoff 미만인 OTU 제거
        filtered_df = otu_df.loc[:, prevalence >= cutoff]

        # 전체 zero 비율 계산
        zero_count = (filtered_df == 0).sum().sum()
        total_count = filtered_df.size
        zero_proportion = zero_count / total_count

        proportions.append(zero_proportion)

    return proportions
