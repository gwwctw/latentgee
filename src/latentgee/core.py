# LatentGEE-U Implementation

import yaml
import numpy as np
import pandas as pd
import optuna
import os
import warnings

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

from functools import partial

from hdbscan import HDBSCAN# <— density-based clustering
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, normalize


from skbio.stats.distance import DistanceMatrix, permanova
from skbio.stats.composition import clr, multiplicative_replacement

from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from scipy.spatial.distance import pdist, squareform


os.environ.setdefault("OMP_NUM_THREADS", "5")  # MKL KMeans 경고/메모리 이슈 회피

# 베스트 모델 저장/재현
def save_model(model, path="best_model.pt"):
    torch.save(model.state_dict(), path)

def load_model(model_class, path="best_model.pt", *args, device=None,**kwargs):
    model=model_class(*args, **kwargs)
    ckpt=torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt)
    if device: model.to(device)
    model.eval()
    return model

# Toy example with >= 1000 samples
def simulate_microbiome_data(n_samples=1000, n_features=50, n_batches=3):
    np.random.seed(42)
    batch_labels = np.random.choice(range(n_batches), size=n_samples)
    data = []
    for i in range(n_samples):
        base=np.random.lognormal(mean=1, sigma=1, size=n_features)
        if batch_labels[i]==1:
            base[:10] += 2.0
        elif batch_labels[i]==2:
            base[10:20]+=1.5
        mask = np.random.binomial(1, 0.3, size=n_features)
        data.append(base*mask)
    return np.array(data), batch_labels

def decode_batch_corrected_latent(model, z_resid, save_path="X_corrected.npy"):
    """
    Decode batch-corrected latent z to reconstructed X, and save to file.
    
    Args:
        model: trained VAE model (with .decode() method)
        z_resid: numpy array of batch-corrected latent representations (z~)
        save_path: file path to save the reconstructed X (as .npy)
    
    Returns:
        x_corrected: decoded reconstruction of shape (n_samples, n_features)
    """
    model.eval()
    device = next(model.parameters()).device
    z_tensor = torch.tensor(z_resid, dtype=torch.float32, device=device)
    with torch.no_grad():
        pi, mu_x, logσ_x = model.decode(z_tensor)
        sigma = logσ_x.exp()  # ← 변수명 통일
        x_hat = (1 - pi) * torch.exp(mu_x + 0.5 * sigma * sigma)  # E[X|z] for ZILN
        x_corrected = x_hat.detach().cpu().numpy()
    np.save(save_path, x_corrected)
    return x_corrected
# --------------------------------------------------

# Flexible MLP builder with dropout and activation selection
"""
레이어 개수, 레이어 크기 변화 패턴(반감, 증가, 고정), activation, dropout을 전부 hyperparameter화한 커스텀 MLP 생성기. 
Optuna에서 조합을 실험하기 위해 만든 FLEXIBLE한 multi-layer percentron builder.

"""
class FlexibleMLP(nn.Module):
    def __init__(self, 
                 layer_dims, 
                 dropout_rate=0.0, 
                 activation='relu'):
        super().__init__()
        layers = []
        act_fn = self.get_activation_fn(activation)
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(act_fn)
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(dropout_rate))
        self.net = nn.Sequential(*layers)
        
    def get_activation_fn(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, x):
        return self.net(x)

# Network architecture generator
def build_layer_dims(input_dim, output_dim, n_layers, base_dim, strategy='constant'):
    dims = [input_dim]
    current_dim = base_dim
    for _ in range(n_layers):
        dims.append(current_dim)
        if strategy == 'halve':
            current_dim = max(current_dim // 2, output_dim)
        elif strategy == 'double':
            current_dim = min(current_dim * 2, 1024)  # arbitrary upper limit
        elif strategy == 'constant':
            continue
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    dims.append(output_dim)
    return dims

# LatentGEE-U VAE using flexible encoder/decoder
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, n_layers, base_dim,
                 strategy='constant', dropout_rate=0.0, activation='relu'):
        super().__init__()
        # ---------- Encoder ----------
        enc_dims = build_layer_dims(input_dim, latent_dim, n_layers, base_dim, strategy)
        self.enc_net = FlexibleMLP(enc_dims, dropout_rate, activation)
        self.fc_mu     = nn.Linear(enc_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(enc_dims[-1], latent_dim)

        # ---------- Decoder ----------
        dec_dims = build_layer_dims(latent_dim, base_dim, n_layers, base_dim, strategy)
        self.dec_net   = FlexibleMLP(dec_dims, dropout_rate, activation)
        self.dec_pi    = nn.Linear(dec_dims[-1], input_dim)   # zero prob
        self.dec_mu    = nn.Linear(dec_dims[-1], input_dim)   # log-normal μ
        self.dec_logσ  = nn.Linear(dec_dims[-1], input_dim)   # log-normal σ

    def encode(self, x):
        h = self.enc_net(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h  = self.dec_net(z)
        pi = torch.sigmoid(self.dec_pi(h))          # (0 ~ 1)
        mu = self.dec_mu(h)                         # 실수
        logσ = self.dec_logσ(h)                     # 실수
        return pi, mu, logσ    # 3-tuple 반환

    def forward(self, x):
        mu_z, logvar_z = self.encode(x)
        z   = self.reparameterize(mu_z, logvar_z)
        pi, mu_x, logσ_x = self.decode(z)
        return (pi, mu_x, logσ_x), mu_z, logvar_z, z
# --------------------------------------------------
# 2. ZILN 음의 log-우도 함수
# --------------------------------------------------
def ziln_nll(x, pi, mu, logσ, eps=1e-8):
    """
    x      : (N, D)  원 데이터 (양수 또는 0)
    pi     : P(x==0)  (N, D)
    mu     : 로그-정규 평균
    logσ   : 로그-정규 log-std
    """
    # ① x == 0 부분
    device = x.device  # 모든 연산을 이 device에서 진행

    # 안전한 log(2π) 계산 (GPU에서 작동하는 버전)
    log2pi = torch.log(torch.tensor(2.0 * np.pi, device=device))

    # ① x == 0 부분
    nll_zero = -(x == 0).float() * torch.log(pi + eps)

    # ② x > 0 부분 (LogNormal pdf)
    pos_mask = (x > 0).float()
    log_x    = torch.log(x + eps)
    sigma= logσ.exp()
    log_pdf  = -0.5 * (((log_x - mu) / sigma) ** 2) - logσ - 0.5 * log2pi
    nll_pos  = -pos_mask * (torch.log(1 - pi + eps) + log_pdf)

    return (nll_zero + nll_pos).mean() # mean over samples & features

def gee_latent_residual(z_np, pseudo_batch):
    df = pd.DataFrame(z_np, columns=[f"z{i}" for i in range(z_np.shape[1])])
    df["cluster"] = pseudo_batch
    residuals = []
    for col in df.columns[:-1]:
        model = GEE.from_formula(f"{col} ~ cluster", groups="cluster", data=df, family=Gaussian(), cov_struct=Exchangeable())
        result = model.fit()
        resid = df[col] - result.fittedvalues
        residuals.append(resid)
    return np.vstack(residuals).T


def find_best_k_silhouette(
    z_np: np.ndarray,
    max_k: int = 10,
    metric: str = "euclidean",
    sample_size: int | None = 500,     # None이면 전체로 정확도↑(느림)
    random_state: int = 42,
    min_cluster_size: int = 1,         # 각 클러스터 최소 샘플 수 제약
    use_minibatch: bool = False,       # N이 클 때 True
) -> Tuple[int, float, np.ndarray]:
    """
    z_np (N,D)에 대해 k=2..max_k KMeans를 시도하고 silhouette가 가장 큰 k를 고름.
    반환: (best_k, best_score, best_labels)
    """
    assert z_np.ndim == 2 and z_np.shape[0] >= 2

    X = z_np.astype(np.float32, copy=False)

    # 1) 표준화(+ cosine이면 L2 정규화로 유클리드~코사인 등가)
    X = StandardScaler().fit_transform(X)
    if metric == "cosine":
        # 코사인 유사도 ~ 단위구에서의 유클리드 거리
        X = normalize(X, norm="l2")

    # 2) 샘플 선택(근사용)
    if sample_size is not None and X.shape[0] > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X_eval = X[idx]
        eval_idx = idx
    else:
        X_eval = X
        eval_idx = None

    best_k, best_score, best_labels = 2, -1.0, None

    # 3) 모델 클래스 선택
    KM = MiniBatchKMeans if use_minibatch else KMeans

    for k in range(2, max_k + 1):
        km = KM(n_clusters=k, random_state=random_state, n_init="auto")
        labels_all = km.fit_predict(X)

        # 최소 클러스터 크기 제약
        if min_cluster_size > 1:
            ok = True
            for c in range(k):
                if np.sum(labels_all == c) < min_cluster_size:
                    ok = False; break
            if not ok:
                continue  # 이 k는 건너뜀

        # silhouette 계산(부분 샘플에 대해)
        if eval_idx is not None:
            score = silhouette_score(X_eval, labels_all[eval_idx], metric=metric)
        else:
            score = silhouette_score(X, labels_all, metric=metric)

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels_all

    # 유효한 k가 없으면 k=2로 강제
    if best_labels is None:
        km = KM(n_clusters=2, random_state=random_state, n_init="auto")
        best_labels = km.fit_predict(X)
        if eval_idx is not None:
            best_score = silhouette_score(X_eval, best_labels[eval_idx], metric=metric)
        else:
            best_score = silhouette_score(X, best_labels, metric=metric)
        best_k = 2

    return best_k, float(best_score), best_labels

from typing import Tuple

def evaluate_latentgee(
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

# --------------------------------------------------
# 3. train_vae() 에서 ZILN 손실 사용
# --------------------------------------------------
def train_vae(model, data_tensor, epochs=50, lr=1e-3,  device="cpu", batch_size=256):
    # 0) 타입 체크
    # Ensure model is an instance of nn.Module and data_tensor is a torch.Tensor before using .to(device)
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module instance")
    if not isinstance(data_tensor, torch.Tensor):
        raise TypeError("data_tensor must be a torch.Tensor instance")
    
    # 1) 디바이스/AMP 설정
    device_str = device if isinstance(device, str) else device.type
    model = model.to(device_str)
    use_amp = (device_str == "cuda")  # GPU에서만 AMP 사용
    # 2) DataLoader는 **CPU 텐서**로 만든다 (pin_memory=True 사용 가능)
    # 데이터가 CPU에서 GPU로 오기까지의 I/O 병목을 없애 GPU 대기시간 제거
    
    num_workers = min(8, max(2, os.cpu_count() - 2))   # CPU코어-2, 최대 8 정도
    data_cpu = data_tensor.detach().to("cpu")
    ds = TensorDataset(data_cpu)  # 항상 CPU 텐서                      # ★ 중요: data_cpu 사용
    dl = DataLoader(ds, 
                    batch_size=batch_size, 
                    shuffle=True,
                    num_workers=num_workers,
                    prefetch_factor=4,
                    persistent_workers=True,
                    pin_memory=(device_str == "cuda"),
                    drop_last=True)  # 모든 스텝에서 입력 shape를 동일하게 유지하고 싶다면

    # 3) 옵티마이저/스케일러
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(device="cuda", enabled=use_amp)

    last_loss = None

    for ep in range(epochs):
        model.train()
        for (xb,) in dl:
            # 4) 배치를 GPU로 전송 (pin_memory=True면 non_blocking=True로 이득)
            xb = xb.to(device_str, non_blocking=use_amp)

            with autocast(device_type="cuda", enabled=use_amp):
                (pi, mu_x, logσ_x), mu_z, logvar_z, _ = model(xb)
                recon_nll = ziln_nll(xb, pi, mu_x, logσ_x)  # 사용자 정의 NLL
                kl = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
                loss: torch.Tensor = recon_nll + kl

            if torch.isnan(loss):
                # 튜닝에서 이 trial을 최악으로 처리하고 끝내려면 큰 값 반환
                return float("inf")

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            last_loss = loss

        # if (ep + 1) % 10 == 0:
        #     print(f"[{ep+1}/{epochs}] ZILN-NLL {recon_nll.item():.4f}  KL {kl.item():.4f}")

    # 5) 마지막 loss를 float로 반환
    if last_loss is None:
        # 데이터가 비어있지 않다면 여긴 오면 안 됨
        return float("inf")
    return float(last_loss.detach().cpu().item())


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

def permanova_r2(
    X: np.ndarray,
    grouping,
    metric:str = "braycurtis",
    ids=None,
    permutations:int = 999,
    pseudocount:float = 1e-6,
):
    """
    Calculate PERMANOVA (pseudo-F, p-value) and R² like vegan::adonis2.
    
    Parameters
    ----------
    X : (n_samples, n_features) array_like
        Feature (count/proportion) matrix.
    grouping : 1-D array_like
        Group labels, length == n_samples.
    metric : str, default 'braycurtis'
        Distance metric for `pdist`.  
        Special case: 'aitchison' ⇒ clr-Euclidean.
    ids : list[str] or None
        Sample IDs for the distance matrix.
    permutations : int
        Number of permutations for PERMANOVA
    pseudocount : float
        Added to zeros before clr, if metric == 'aitchison'.
    """
    X = np.asarray(X.astype(float))
    grouping = np.asarray(grouping)

    # ---------- sample IDs ----------
    if ids is None:
        ids = [f"S{i+1}" for i in range(X.shape[0])]

    # ---------- distance matrix ----------
    if metric.lower() == "aitchison":
        # 1) zero replacement → clr transform
        X_clr = clr(multiplicative_replacement(X, pseudocount))
        # 2) Euclidean distance in clr space == Aitchison distance
        dist = squareform(pdist(X_clr, metric="euclidean"))
    else:
        dist = squareform(pdist(X, metric=metric))

    dm = DistanceMatrix(dist, ids=ids)

    # ---------- R² ----------
    d2 = dm.data ** 2
    n = d2.shape[0]
    sst = d2[np.triu_indices(n, 1)].sum() / n

    ssw = 0.0
    for g in np.unique(grouping):
        idx = np.where(grouping == g)[0]
        if len(idx) < 2:
            continue
        d2_g = d2[np.ix_(idx, idx)]
        ssw += d2_g[np.triu_indices(len(idx), 1)].sum() / len(idx)

    ssb = sst - ssw
    r2 = ssb / sst

    # ---------- PERMANOVA ----------
    res = permanova(dm, grouping, permutations=permutations)

    return res, r2

import pandas as pd, numpy as np, subprocess, tempfile, pathlib, shutil

def adonis2_permanova_r2_via_rscript(X: np.ndarray, grouping, metric="braycurtis", permutations=999):
    X = pd.DataFrame(np.asarray(X, float))
    meta = pd.DataFrame({"group": np.asarray(grouping)})
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        Xf, Mf, Of, Rf = td/"X.csv", td/"meta.csv", td/"out.csv", td/"run.R"
        X.to_csv(Xf, index=False); meta.to_csv(Mf, index=False)
        Rf.write_text(f"""
        suppressMessages(library(vegan)); suppressMessages(library(compositions))
        X <- read.csv("{Xf}", check.names=FALSE); META <- read.csv("{Mf}")
        META$group <- factor(META$group)
        if ("{metric}"=="braycurtis") {{
          d <- vegdist(X, method="bray")
        }} else if ("{metric}"=="aitchison") {{
          X <- as.matrix(X) + 1e-6; X <- X/rowSums(X); X <- clr(X); d <- dist(X, method="euclidean")
        }} else stop("metric not supported")
        fit <- adonis2(d ~ group, data=META, permutations={permutations})
        write.csv(as.data.frame(fit), file="{Of}", row.names=TRUE)
        """, encoding="utf-8")
        subprocess.run(["Rscript", str(Rf)], check=True)
        res = pd.read_csv(Of, index_col=0)
        term_rows = [i for i in res.index if i not in ("Residual","Total")]
        r2 = float(res.loc[term_rows[0], "R2"])
        return res, r2
    
    