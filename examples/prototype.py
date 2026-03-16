
import sys
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import optuna
import sklearn
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from pathlib import Path
from datetime import datetime

import hdbscan
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

print("torch ==", getattr(torch, "__version__", None))
print("numpy ==", getattr(np, "__version__", None))
print("scikit-learn ==", getattr(sklearn, "__version__", None))
print("optuna ==", getattr(optuna, "__version__", None))
print("hdbscan ==", hdbscan.__version__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 베스트 모델 저장/재현
def save_model(model, path="best_model.pt"):
    torch.save(model.state_dict(), path)

def load_model(model_class, path="best_model.pt", *args, **kwargs):
    model=model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def save_dated_filename(base_name = "best_model", ext = ".pt", folder = "."):
    
    today = datetime.today().strftime("%Y-%m-%d")
    folder = Path(folder)

    filename = f"{base_name}_{today}{ext}"
    path = folder / filename

    if not path.exists():
        return path

    i = 1
    while True:
        filename = f"{base_name}_{today}({i}){ext}"
        path = folder / filename
        if not path.exists():
            return path
        i += 1

def choose_base_dim(input_dim: int, strategy: str, n_layers: int) -> int:
    """
    • 'halve'  : input_dim // 2
    • 'double' : input_dim // 4  (폭 좁혀 시작)
    • default  : input_dim // 4
    """
    if strategy == "halve":
        return max(input_dim // 2, 8)
    elif strategy == "double":
        return max(input_dim // 4, 8)
    else:                   # 'constant'
        return max(input_dim // 4, 8)

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
        enc_dims = build_layer_dims(input_dim, base_dim, n_layers, latent_dim, strategy)
        self.enc_net = FlexibleMLP(enc_dims, dropout_rate, activation)
        self.fc_mu     = nn.Linear(enc_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(enc_dims[-1], latent_dim)

        # ---------- Decoder ----------
        dec_dims = build_layer_dims(latent_dim, base_dim, n_layers, input_dim, strategy)
        self.dec_net   = FlexibleMLP(dec_dims, dropout_rate, activation)
        self.dec_pi    = nn.Linear(dec_dims[-1], input_dim)   # zero prob
        self.dec_mu    = nn.Linear(dec_dims[-1], input_dim)   # log-normal μ
        self.dec_log_sigma  = nn.Linear(dec_dims[-1], input_dim)   # log-normal _sigma

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
        log_sigma = self.dec_log_sigma(h)                     # 실수
        return pi, mu, log_sigma    # 3-tuple 반환

    def forward(self, x):
        mu_z, logvar_z = self.encode(x)
        z   = self.reparameterize(mu_z, logvar_z)
        pi, mu_x, log_sigma_x = self.decode(z)
        return (pi, mu_x, log_sigma_x), mu_z, logvar_z, z
# --------------------------------------------------
# 2. ZILN 음의 log-우도 함수
# --------------------------------------------------
def ziln_nll(x, pi, mu, log_sigma, eps=1e-8):
    """
    x      : (N, D)  원 데이터 (양수 또는 0)
    pi     : P(x==0)  (N, D)
    mu     : 로그-정규 평균
    log_sigma   : 로그-정규 log-std
    """
    # ① x == 0 부분
    nll_zero = -(x == 0).float() * torch.log(pi + eps)

    # ② x > 0 부분 (LogNormal pdf)
    pos_mask = (x > 0).float()
    log_x    = torch.log(x + eps)
    log_pdf  = -0.5 * (((log_x - mu) / log_sigma.exp()) ** 2) \
               - log_sigma - 0.5 * np.log(2 * np.pi)           # log-pdf(로그정규)
    nll_pos  = -pos_mask * (torch.log(1 - pi + eps) + log_pdf)

    return (nll_zero + nll_pos).mean()      # mean over samples & features

# --------------------------------------------------
# 3. Pseudo-batch clustering 함수
# --------------------------------------------------
def pseudo_clustering(z_tensor, 
                      min_cluster_size: int = 10,
                      min_samples: int | None = None,
                      metric="euclidean",
                      cluster_selection_method="eom"):
    """
    ▣ 역할
        1. HDBSCAN → pseudo-batch 라벨 생성
    """
    if z_tensor.requires_grad:
        z_tensor = z_tensor.detach()
        z_cpu = z_tensor.detach().cpu().numpy()

    # ── 1. Pseudo-batch 탐색 ─────────────────────────────
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method
    )
    labels = hdb.fit_predict(z_cpu)          # shape (N,)    
    return labels

# --------------------------------------------------
# 4. gee latent residualizaiton
# --------------------------------------------------
def gee_latent_residual(z_np, 
                        pseudo_batch_labels, 
                        covariates_df=None):

    df = pd.DataFrame(z_np, columns=[f"z{i}" for i in range(z_np.shape[1])])
    df["cluster"] = pseudo_batch_labels
    cov_names = []
    if covariates_df is not None:
        if isinstance(covariates_df, pd.DataFrame):
            df = pd.concat([df, covariates_df], axis=1)
            cov_names = list(covariates_df.columns)

        elif isinstance(covariates_df, np.ndarray):
            for i in range(covariates_df.shape[1]):
                name = f"cov{i}"
                df[name] = covariates_df[:, i]
                cov_names.append(name)

    residuals = []
    latent_cols = [c for c in df.columns if c.startswith("z")]

    for col in latent_cols:
        if covariates_df is None:
            formula = f"{col} ~ 1"
        else:
            formula = f"{col} ~ {' + '.join(cov_names)}"

        model = GEE.from_formula(
            formula,
            groups="cluster",
            data=df,
            family=Gaussian(),
            cov_struct=Exchangeable()
        )

        result = model.fit()
        resid = df[col] - result.fittedvalues
        residuals.append(resid.values)

    return np.vstack(residuals).T

def evaluate_latentgee(
        model: nn.Module,
        X_tensor: torch.Tensor,
        covariates_df: pd.DataFrame | None = None,
        min_cluster_size: int = 10,
        min_samples: int | None = None,
        metric: str = "euclidean",
    ) -> tuple[np.ndarray, float]:

    """
    LatentGEE evaluation

    Steps
    -----
    1. Encode data to latent space
    2. HDBSCAN clustering → pseudo-batch
    3. GEE residualization
    4. silhouette score calculation
    """

    model.eval()

    with torch.no_grad():
        mu, logvar = model.encode(X_tensor)
        z = model.reparameterize(mu, logvar)
        z_np = z.cpu().numpy()

    # ---- pseudo-batch clustering ----
    labels = pseudo_clustering(
        z,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric = metric
    )

    # ---- remove noise samples ----
    mask = labels != -1

    n_valid = np.unique(labels[mask]).size
    if n_valid < 2 or mask.sum() < 3:
        raise optuna.TrialPruned()

    z_used = z_np[mask]
    lbl_used = labels[mask]
    noise_ratio = (labels == -1).mean()

    # ---- covariate subset ----
    if covariates_df is not None:
        cov_used = covariates_df.iloc[mask]
    else:
        cov_used = None

    # ---- GEE residualization ----
    z_resid = gee_latent_residual(
        z_used,
        lbl_used,
        covariates_df=cov_used
    )

    # ---- silhouette score ----
    sil = silhouette_score(z_resid, lbl_used, metric=metric)

    return labels, sil, noise_ratio
    
# --------------------------------------------------
# 3. train_vae() 에서 ZILN 손실 사용
# --------------------------------------------------
def train_vae(model, data_tensor, epochs=50, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        (pi, mu_x, log_sigma_x), mu_z, logvar_z, _ = model(data_tensor)

        recon_nll = ziln_nll(data_tensor, pi, mu_x, log_sigma_x) # negative log likelihood 
        kl = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
        loss = recon_nll + kl

        if torch.isnan(loss):
            raise optuna.TrialPruned()

        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 10 == 0:
            print(f"[{ep+1}/{epochs}] ZILN-NLL {recon_nll:.4f}  KL {kl:.4f}")
    return loss.item()

# ---------------------------------------------------------
# 2. Optuna objective
def objective(trial: optuna.Trial,
              config: dict,
              X_tensor: torch.Tensor,
              log_file: str = "optuna_trial_log.csv") -> float:
    input_dim = X_tensor.shape[1]

    # ── ① 파라미터 샘플링 ─────────────────────────
    strategy      = trial.suggest_categorical("strategy",
                        config["search_space"]["model"]["strategy"])
    n_layers      = trial.suggest_int("n_layers",
                        *config["search_space"]["model"]["n_layers"])
    base_dim = trial.suggest_categorical("base_dim",
                                         config["search_space"]["model"]["base_dim"])
    latent_dim    = trial.suggest_int("latent_dim",
                        *config["search_space"]["model"]["latent_dim"])
    activation    = trial.suggest_categorical("activation",
                        config["search_space"]["model"]["activation"])
    dropout_rate  = trial.suggest_float("dropout_rate",
                        *config["search_space"]["model"]["dropout_rate"])
    epochs        = trial.suggest_categorical("epochs",
                        config["search_space"]["training"]["epochs"])
    
    lr_low, lr_high = sorted(map(float,config["search_space"]["training"]["learning_rate"]["loguniform"]))
    learning_rate = trial.suggest_float("learning_rate", lr_low, lr_high, log=True)

    # HDBSCAN
    mcs_low, mcs_high   = config["search_space"]["clustering"]["min_cluster_size"]
    min_cluster_size    = trial.suggest_int("min_cluster_size", mcs_low, mcs_high)
    min_samples_token   = trial.suggest_categorical("min_samples_token",config["search_space"]["clustering"]["min_samples"])
    min_samples         = None if (min_samples_token in ["None", None]) else int(min_samples_token)
    metric              = trial.suggest_categorical("metric", config["search_space"]["clustering"]["metric"])


    
    # ── ② 모델 구성 & 학습 ──────────────────────── 

    model = VAE(input_dim=input_dim,
                latent_dim=latent_dim,
                n_layers=n_layers,
                base_dim=base_dim,
                strategy=strategy,
                dropout_rate=dropout_rate,
                activation=activation)
    try:
        last_loss = train_vae(model, X_tensor, epochs=epochs, lr=learning_rate)
        last_loss = float(last_loss)           # Tensor → python scalar
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            raise optuna.TrialPruned()
        else:
            raise

    # ── ③ 평가 ───────────────────────────────────
    try:
        labels, score, noise_ratio = evaluate_latentgee(
            model, X_tensor,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric
        )
    except ValueError:                         # 실루엣 계산 실패
        raise optuna.TrialPruned()

    if np.isnan(score) or np.isinf(score):
        raise optuna.TrialPruned()

    # ── ④ 로그 ───────────────────────────────────
    n_clusters = len(np.unique(labels[labels != -1]))
    res_df = pd.DataFrame({
        "trial_number":  [trial.number],
        "base_dim":      [base_dim],
        "n_layers":      [n_layers],
        "latent_dim":    [latent_dim],
        "activation":    [activation],
        "strategy":      [strategy],
        "dropout_rate":  [dropout_rate],
        "epochs":        [epochs],
        "learning_rate": [learning_rate],
        "loss":          [last_loss],
        "silhouette":    [score],
        "n_clusters":    [n_clusters],
        "noise_ratio":   [noise_ratio]
    })
    mode, header = ("w", True) if trial.number == 0 else ("a", False)
    res_df.to_csv(log_file, mode=mode, index=False, header=header)

    print(f"Trial {trial.number:3d} | sil={score:+.4f} | k={n_clusters}")

    # ── ⑤ Optuna가 최대화할 스코어 반환 ───────────
    return score

def load_hivrc(file_path):
    data_path = Path(f"{file_path}/insight.merged_otus.txt")
    meta_path = Path(f"{file_path}/SupplementaryMaterial.xlsx")
    
    dat = pd.read_csv(data_path, sep="\t", encoding = "utf-8")
    dat_meta = pd.read_excel(meta_path, header = 1, usecols = "B:F")

    dat_cols = dat_meta["SeqID"].astype(str).to_list()
    dat_cols.insert(0, 'Resphera Insight (Raw Counts)')
    dat_T = dat[dat_cols].T
    dat_T.reset_index(inplace = True)
    dat_T.columns = dat_T.iloc[0].astype(str)
    dat_T = dat_T.iloc[1:]
    dat_T.columns.values[0] = "SeqID"
    dat_merged = pd.merge(dat_T, dat_meta[["SeqID", "Study.1", "Study", "hivstatus", "Age"]], on = "SeqID", how = "inner")
        
    X = dat_T.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    # X_tensor = torch.tensor(X.values, dtype = torch.float32)
    
    return X, dat_merged

def main():
    
    # ── seed 고정
    set_seed()
    # ── Device 설정 ────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---- 1. YAML 로드 ----
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
        
    # ---- 2. data load ----
    X_raw, dat_merged = load_hivrc(file_path = "F:/졸업 후 연구/LatentGEE/Data")
    X_tensor = torch.tensor(X_raw.values, dtype=torch.float32).to(device)
    
    # ---- 3. Optuna 실행 ----
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        lambda tr: objective(tr, cfg, X_tensor),   # ← 기존 objective 그대로
        n_trials=1000,
    )
        
    # ---- 4. Best hyper-parameter
    best_trial   = study.best_trial
    best_params  = best_trial.params
    
    print("Best hyperparameters")
    print(best_params)

    # ── 4-1. 파라미터 꺼내기 ───────────────────────────
    input_dim    = X_tensor.shape[1]
    base_dim = best_params["base_dim"]
    latent_dim   = best_params["latent_dim"]
    n_layers     = best_params["n_layers"]
    strategy     = best_params["strategy"]
    dropout_rate = best_params["dropout_rate"]
    activation   = best_params["activation"]
    epochs       = best_params["epochs"]          # 30 · 50 · 80 · 100 중 하나
    lr           = best_params["learning_rate"]
    min_cs       = best_params["min_cluster_size"]
    min_samples  = None                           # search space가 "None" 뿐이므로

    # ── 4-2. 모델 인스턴스 ─────────────────────────────
    best_model = VAE(
        input_dim   = input_dim,
        latent_dim  = latent_dim,
        n_layers    = n_layers,
        base_dim    = base_dim,
        strategy    = strategy,
        dropout_rate= dropout_rate,
        activation  = activation,
    ).to(device)
    train_vae(best_model, X_tensor, epochs=epochs, lr=lr)
    
    # ── 4-3. encode → reparameterize → decode
    best_model.eval()
    with torch.no_grad():
        # a) get μ, logvar
        mu_z, logvar_z = best_model.encode(X_tensor)
        # b) reparameterize
        z = best_model.reparameterize(mu_z, logvar_z)
        # c) decode
        pi, mu_x, log_sigma_x = best_model.decode(z)
        sigma_x = torch.exp(log_sigma_x)
        # d) compute E[X] under ZILN: (1−π) · exp(μ + ½_sigma²)
        X_recon = (1 - pi) * torch.exp(mu_x + 0.5 * sigma_x**2)

    # move back to CPU / numpy
    recon_np = X_recon.cpu().numpy()

    # 4) wrap in a DataFrame with original sample / feature labels
    recon_df = pd.DataFrame(
        recon_np,
        index=X_raw.index,      # same sample order
        columns=X_raw.columns   # same OTU names
    )

    # 5) save to disk
    save_recon_path = save_dated_filename("X_reconstructed_ZILN", ".csv")
    recon_df.to_csv(save_recon_path) 

    print(f"Reconstruction complete — written to {save_recon_path}")
    
    # ─── best model 저장 ───────────────────────────────────────────
    save_path = save_dated_filename("best_model", ".pt")
    torch.save(best_model.state_dict(), save_path)
    print(f"✓ 모델 가중치를 {save_path} 로 저장했습니다.")
  
    
if __name__ == "__main__":
    main()