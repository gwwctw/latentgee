# LatentGEE-U Implementation

import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import optuna

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from hdbscan import HDBSCAN            # <— density-based clustering
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler

from skbio.stats.distance import DistanceMatrix, permanova
from skbio.stats.composition import clr, multiplicative_replacement

from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from scipy.spatial.distance import pdist, squareform


# 베스트 모델 저장/재현
def save_model(model, path="best_model.pt"):
    torch.save(model.state_dict(), path)

def load_model(model_class, path="best_model.pt", *args, **kwargs):
    model=model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
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
    z_tensor = torch.tensor(z_resid, dtype=torch.float32)
    with torch.no_grad():
        x_corrected = model.decode(z_tensor).cpu().numpy()
    np.save(save_path, x_corrected)
    return x_corrected


# Flexible MLP builder with dropout and activation selection
class FlexibleMLP(nn.Module):
    def __init__(self, layer_dims, dropout_rate=0.0, activation='relu'):
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
    nll_zero = -(x == 0).float() * torch.log(pi + eps)

    # ② x > 0 부분 (LogNormal pdf)
    pos_mask = (x > 0).float()
    log_x    = torch.log(x + eps)
    log_pdf  = -0.5 * (((log_x - mu) / logσ.exp()) ** 2) \
               - logσ - 0.5 * np.log(2 * np.pi)           # log-pdf(로그정규)
    nll_pos  = -pos_mask * (torch.log(1 - pi + eps) + log_pdf)

    return (nll_zero + nll_pos).mean()      # mean over samples & features

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


def find_best_k_silhouette(z_np, max_k=10):
    best_score = -1
    best_k = 2
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(z_np)
        labels = kmeans.labels_
        score = silhouette_score(z_np, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

    파이
# Silhouette-based evaluation function
def evaluate_latentgee_u(
        model: nn.Module,
        X_tensor: torch.Tensor,
        min_cluster_size: int = 10,
        min_samples: int | None = None,
        allow_noise: bool = True,
        metric: str = "braycurtis", 
    ) -> tuple[np.ndarray, float]:
    """
    • Encoder의 μ(=latent z) → HDBSCAN 으로 pseudo-batch 추정
    • GEE residual → silhouette score 반환
    Returns
    -------
    labels  : np.ndarray  (length = n_samples)  −1 = noise
    score   : float       silhouette score (noise 제외)
    """
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X_tensor)
        z_np = mu.cpu().numpy()

    # ---- HDBSCAN clustering ----
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,          # None ⇒ heuristic = min_cluster_size
        metric=metric,
        cluster_selection_method="eom"
    )
    labels = hdb.fit_predict(z_np)        # noise ⇒ −1

    # 실루엣 / GEE 는 noise 제외
    mask = labels != -1
    n_valid = np.unique(labels[mask]).size
    if n_valid < 2 or mask.sum() < 3:
        raise optuna.TrialPruned()   # or return -1

    z_used, lbl_used = z_np[mask], labels[mask]
    # GEE residual
    z_resid = gee_latent_residual(z_used, lbl_used)

    # 1) silhouette
    sil = silhouette_score(z_resid, lbl_used)

    # # 2) PERMANOVA R²
    # D  = pairwise_distances(z_resid, metric=metric)
    # ids = [f"s{i}" for i in range(len(z_resid))]
    # dm = DistanceMatrix(D, ids=ids)
    # perm = permanova(dm, grouping=lbl_used, permutations=999)
    # stat = perm['test statistic']
    # r2   = 1.0 - stat

    # return labels, sil, r2
    return labels, sil
    
# --------------------------------------------------
# 3. train_vae() 에서 ZILN 손실 사용
# --------------------------------------------------
def train_vae(model, data_tensor, epochs=50, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        (pi, mu_x, logσ_x), mu_z, logvar_z, _ = model(data_tensor)

        recon_nll = ziln_nll(data_tensor, pi, mu_x, logσ_x) # negative log likelihood 
        kl = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
        loss = recon_nll + kl

        if torch.isnan(loss):
            raise optuna.TrialPruned()

        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 10 == 0:
            print(f"[{ep+1}/{epochs}] ZILN-NLL {recon_nll:.4f}  KL {kl:.4f}")
    return loss.item()


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
        cluster_selection_method="eom",
    )
    labels = hdb.fit_predict(z_cpu)          # shape (N,)

    # ── 2. 배치별 평균 제거 (= GEE intercept) ───────────
    z_tilde = z_tensor.clone()
    labels_unique = np.unique(labels)

    for g in labels_unique:
        idx = (labels == g)
        if g == -1:            # noise 처리
            if noise_handling == "global":
                mean_vec = z_tensor.mean(dim=0, keepdim=True)
                z_tilde[idx] -= mean_vec
            elif noise_handling == "self":
                pass           # 그대로 두거나 원하는 방식으로
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
        Number of permutations for PERMANOVA.
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
