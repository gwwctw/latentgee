import os
import yaml
import numpy as np
import pandas as pd
import optuna
import sklearn
import random
import sys
import logging
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

import hdbscan
from hdbscan import HDBSCAN            # <— density-based clustering

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, TensorDataset

from skbio.stats.distance import DistanceMatrix, permanova

from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from skbio.stats.distance import permanova, DistanceMatrix
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
#from matplotlib.patches import Ellipse
#from scipy.spatial.distance import pdist, squareform

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record) # emit()은 로그 메시지가 기록될 때마다 자동으로 호출되는 메서드
        self.flush()  # 매 로그마다 즉시 파일에 씀

def setup_logger(log_dir=".", cutoff=0.1):
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    today = datetime.today().strftime("%Y-%m-%d")
    log_path = log_dir / f"latentgee_prototype_cutoff_{cutoff}_{today}.log"
    
    i = 1
    while log_path.exists():
        log_path = log_dir / f"latentgee_prototype_cutoff_{cutoff}_{today}({i}).log"
        i += 1

    logger = logging.getLogger("latentgee_prototype")
    logger.setLevel(logging.DEBUG)

    # FlushFileHandler 사용
    file_handler = FlushFileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"로그 저장 경로: {log_path}")
    return logger

# 베스트 모델 저장/재현
def save_model(model, path="best_model.pt"):
    torch.save(model.state_dict(), path)

def make_save_best_callback(logger, log_dir, cutoff=0.1):
    def callback(study, trial):
        if study.best_trial.number == trial.number:
            today = datetime.today().strftime("%Y-%m-%d")
            best_params = trial.params
            logger.info(f"New best trial {trial.number} | r2={trial.value:.4f}")
            best_path = Path(log_dir) / f"best_params_latest_trial{trial.number}_cutoff_{cutoff}_{today}.json"
            with open(best_path, "w") as f:
                json.dump(best_params, f, indent=2)
    return callback


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

def apply_init(model, init_type):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            elif init_type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
# Flexible MLP builder with dropout and activation selection
class FlexibleMLP(nn.Module):
    def __init__(self, layer_dims, dropout_rate=0.0, activation='relu', norm='none'):
        super().__init__()
        layers = []
        act_fn = self.get_activation_fn(activation)
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                # norm 추가
                if norm == 'batchnorm':
                    layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
                elif norm == 'layernorm':
                    layers.append(nn.LayerNorm(layer_dims[i + 1]))
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
def build_layer_dims(input_dim, base_dim, output_dim, n_layers,  strategy='constant'):
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
                 strategy='constant', dropout_rate=0.0, activation='relu',
                 norm='none'):   # ← 추가
        super().__init__()
        # ---------- Encoder ----------
        enc_dims = build_layer_dims(input_dim, base_dim, latent_dim,  n_layers, strategy)
        self.enc_net = FlexibleMLP(enc_dims[:-1], dropout_rate, activation, norm)
        self.fc_mu     = nn.Linear(enc_dims[-2], latent_dim)
        self.fc_logvar = nn.Linear(enc_dims[-2], latent_dim)

        # ---------- Decoder ----------
        dec_dims = build_layer_dims(latent_dim, base_dim, input_dim, n_layers, strategy)
        self.dec_net   = FlexibleMLP(dec_dims[:-1], dropout_rate, activation, norm)
        self.dec_pi    = nn.Linear(dec_dims[-2], input_dim)   # zero prob
        self.dec_mu    = nn.Linear(dec_dims[-2], input_dim)   # log-normal μ
        self.dec_log_sigma  = nn.Linear(dec_dims[-2], input_dim)   # log-normal _sigma

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
    
    z_cpu = z_tensor.detach().cpu().numpy()

    # ── 1. Pseudo-batch 탐색 ─────────────────────────────
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method
    )
    labels = hdb.fit_predict(z_cpu)          # shape (N,)    
    return labels

# --------------------------------------------------
# 4. gee latent residualizaiton
# --------------------------------------------------
def gee_latent_residual(z_np, 
                        pseudo_batch_labels, 
                        covariates_df: pd.DataFrame | None = None):

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
            formula = f"{col} ~ C(cluster)"
        else:
            formula = f"{col} ~ C(cluster) + {' + '.join(cov_names)}"

        model = GEE.from_formula(
            formula,
            groups="cluster",
            data=df,
            family=Gaussian(),
            cov_struct=Exchangeable()
        )

        result = model.fit()
        if not result.converged:
            raise ValueError("GEE did not converge")
        if np.isnan(result.fittedvalues).any():
            raise ValueError("GEE fittedvalues contain NaN")
        
        resid = df[col].values - result.fittedvalues.values
        
        residuals.append(resid)

    return np.vstack(residuals).T

def compute_permanova_r2(z_resid, lbl_used):
    dist = squareform(pdist(z_resid, metric="euclidean"))
    dm = DistanceMatrix(dist)
    result = permanova(dm, lbl_used, permutations=99)  # 빠르게 99
    n = len(lbl_used)
    # R² 계산
    d2 = dist ** 2
    sst = d2[np.triu_indices(n, 1)].sum() / n
    ssw = 0.0
    for g in np.unique(lbl_used):
        idx = np.where(lbl_used == g)[0]
        if len(idx) < 2:
            continue
        d2_g = d2[np.ix_(idx, idx)]
        ssw += d2_g[np.triu_indices(len(idx), 1)].sum() / len(idx)
    r2 = (sst - ssw) / sst
    return float(r2)

def evaluate_latentgee(
        model: nn.Module,
        X_tensor: torch.Tensor,
        covariates_df: pd.DataFrame | None = None,
        min_cluster_size: int = 10,
        min_samples: int | None = None,
        study_labels=None,
        cluster_selection_method: str = "eom"
    ) -> tuple[np.ndarray, float]:

    """
    LatentGEE evaluation

    Steps
    -----
    1. Encode data to latent space
    2. HDBSCAN clustering → pseudo-batch
    3. GEE residualization
    4. permanova r2 calculation
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
        metric = "euclidean",
        cluster_selection_method = cluster_selection_method,
    )

    # ---- remove noise samples ----
    mask = labels != -1
    n_valid = np.unique(labels[mask]).size
    if n_valid < 2 or mask.sum() < 3:
        raise ValueError("no. of valid cluster < 2")
    z_used = z_np[mask]
    lbl_used = labels[mask]
    noise_ratio = (labels == -1).mean()

    # ---- covariate subset ----
    if covariates_df is not None:
        cov_used = covariates_df[mask].reset_index(drop=True)
    else:
        cov_used = None

    # ---- GEE residualization ----
    z_resid = gee_latent_residual(
        z_used,
        lbl_used,
        covariates_df=cov_used
    )

    # ---- permanova r2 score ----
    if np.isnan(z_resid).any():
        raise ValueError("z_resid contains NaN")

    if study_labels is not None:
        study_used = np.array(study_labels).flatten()[mask]
        score = compute_permanova_r2(z_resid, study_used)
    else:
        score = compute_permanova_r2(z_resid, lbl_used)
        
    return labels, score, noise_ratio
    
# --------------------------------------------------
# 3. train_vae() 에서 ZILN 손실 사용
# --------------------------------------------------
def train_vae(model, data_tensor,
              beta_kl=0.1,
              epochs=50,
              lr=1e-3,
              batch_size=64,
              weight_decay=0.0,
              grad_clip_norm=0.5,
              kl_warmup_ratio=0.5,
              logger=None):   # ← epoch 수 대신 비율로 변경

    kl_warmup_epochs = int(epochs * kl_warmup_ratio)  # 전체 epoch의 50%를 warmup으로
    
    dataset = TensorDataset(data_tensor)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(epochs):
        model.train()
        epoch_recon = 0.0
        epoch_kl    = 0.0
        n_batches   = 0

        # KL warmup
        if kl_warmup_epochs > 0 and ep < kl_warmup_epochs:
            beta_t = beta_kl * (ep + 1) / kl_warmup_epochs
        else:
            beta_t = beta_kl

        for (x_batch,) in loader:
            (pi, mu_x, log_sigma_x), mu_z, logvar_z, _ = model(x_batch)
            recon_nll = ziln_nll(x_batch, pi, mu_x, log_sigma_x)
            kl = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
            loss = recon_nll + beta_t * kl

            if torch.isnan(loss):
                raise ValueError("NaN loss detected")

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            opt.step()

            epoch_recon += recon_nll.item()
            epoch_kl    += kl.item()
            n_batches   += 1

        if (ep + 1) % 10 == 0:
            if logger :
                logger.info(f"[{ep+1}/{epochs}] "
                    f"kl_warmup_epochs {kl_warmup_epochs}   "
                    f"ZILN-NLL {epoch_recon/n_batches:.4f}  "
                    f"KL {epoch_kl/n_batches:.4f}  "
                    f"beta_t {beta_t:.4f}")
            else:
                print(f"[{ep+1}/{epochs}] "
                    f"kl_warmup_epochs {kl_warmup_epochs}   "
                    f"ZILN-NLL {epoch_recon/n_batches:.4f}  "
                    f"KL {epoch_kl/n_batches:.4f}  "
                    f"beta_t {beta_t:.4f}")

    return loss.item()

# ---------------------------------------------------------
# 2. Optuna objective
def objective(trial: optuna.Trial,
              config: dict,
              X_tensor_cache:  dict,
              real_batch: np.array,
              trial_res_file,
              cutoff_list: list,
              covariates_df: pd.DataFrame | None = None,
              logger = None) -> float:
    

    # ── ① 파라미터 샘플링 ─────────────────────────
    cutoff        = trial.suggest_categorical("cutoff", cutoff_list)
    init          = trial.suggest_categorical("init",
                        config["search_space"]["model"]["init"])
    beta_kl       = trial.suggest_float("beta_kl",
                        *config["search_space"]["model"]["beta_kl"])
    norm          = trial.suggest_categorical("norm",
                        config["search_space"]["model"]["norm"])
    strategy      = trial.suggest_categorical("strategy",
                        config["search_space"]["model"]["strategy"])
    n_layers      = trial.suggest_int("n_layers",
                        *config["search_space"]["model"]["n_layers"])
    base_dim      = trial.suggest_categorical("base_dim",
                        config["search_space"]["model"]["base_dim"])
    latent_dim    = trial.suggest_int("latent_dim",
                        *config["search_space"]["model"]["latent_dim"])
    activation    = trial.suggest_categorical("activation",
                        config["search_space"]["model"]["activation"])
    dropout_rate  = trial.suggest_categorical("dropout_rate",
                        config["search_space"]["model"]["dropout_rate"])
    
    epochs        = trial.suggest_categorical("epochs",
                        config["search_space"]["training"]["epochs"])
    batch_size    = trial.suggest_categorical("batch_size",
                        config["search_space"]["training"]["batch_size"])
    weight_decay  = trial.suggest_float("weight_decay",
                        config["search_space"]["training"]["weight_decay"]["loguniform"][0],
                        config["search_space"]["training"]["weight_decay"]["loguniform"][1],
                        log=True)
    learning_rate = trial.suggest_float("learning_rate", 
                        config["search_space"]["training"]["learning_rate"]["loguniform"][0],
                        config["search_space"]["training"]["learning_rate"]["loguniform"][1],
                        log=True)
    
    grad_clip_norm  = trial.suggest_float("grad_clip_norm",
                        *config["search_space"]["training"]["grad_clip_norm"])
    kl_warmup_ratio = trial.suggest_float("kl_warmup_ratio",
                        *config["search_space"]["model"]["kl_warmup_ratio"])

    csm             = trial.suggest_categorical("cluster_selection_method",
                        config["search_space"]["clustering"]["cluster_selection_method"])
    # HDBSCAN
    mcs_low, mcs_high   = config["search_space"]["clustering"]["min_cluster_size"]
    min_cluster_size    = trial.suggest_int("min_cluster_size", mcs_low, mcs_high)
    min_samples_token   = trial.suggest_categorical("min_samples_token",config["search_space"]["clustering"]["min_samples"])
    
    min_samples         = None if min_samples_token in (None, "null") else int(min_samples_token)
    # metric              = trial.suggest_categorical("metric", config["search_space"]["clustering"]["metric"])


    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = X_tensor_cache[cutoff].to(device)
    input_dim = X_tensor.shape[1]
    
    # ── ② 모델 구성 & 학습 ──────────────────────── 
    model = VAE(input_dim=input_dim,
                latent_dim=latent_dim,
                n_layers=n_layers,
                base_dim=base_dim,
                strategy=strategy,
                dropout_rate=dropout_rate,
                activation=activation, norm=norm).to(device)
    
    apply_init(model, init)  # ← 추가
    
    last_loss = float("nan") #초기화
    
    
    try:
        last_loss = train_vae(model, 
                              X_tensor, 
                              beta_kl=beta_kl,
                              epochs=epochs, 
                              lr=learning_rate, 
                              batch_size=batch_size, 
                              weight_decay=weight_decay, 
                              grad_clip_norm=grad_clip_norm,
                              kl_warmup_ratio=kl_warmup_ratio,
                              logger=logger)
        last_loss = float(last_loss)
    except RuntimeError as e:
        if "out of memory" in str(e): # OOM — 하드웨어 한계라 재시도 의미 없음
            raise optuna.TrialPruned()
        else:
            raise
    except ValueError as e:
        if "NaN" in str(e): # NaN loss — 이 파라미터 조합은 학습 자체가 불가능
            raise optuna.TrialPruned()
        else:
            if logger:
                logger.info(f"Trial {trial.number} | train ValueError: {e}")
            raise
        

    # ── ③ 평가 ───────────────────────────────────
    try:
        labels, score, noise_ratio = evaluate_latentgee(
            model, X_tensor,
            covariates_df= covariates_df,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            study_labels=real_batch,
            cluster_selection_method=csm
        )
    except ValueError as e:                        # 클러스터 못 찾음 → 1.0 반환
        if logger:
            logger.info(f"Trial {trial.number} | eval ValueError: {e}")
        return 1.0

    if np.isnan(score) or np.isinf(score):    # score 이상값 → 1.0 반환
        return 1.0

    # ── ④ 로그 ───────────────────────────────────
    n_clusters = len(np.unique(labels[labels != -1]))
    res_df = pd.DataFrame({
        "cutoff": [cutoff],
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
        "permanova":    [score],
        "n_clusters":    [n_clusters],
        "noise_ratio":   [noise_ratio],
        "beta_kl":       [beta_kl],          # 추가
        "weight_decay":  [weight_decay],  # 추가
        "grad_clip_norm":     [grad_clip_norm],     # 추가
        "csm":           [csm],           # 추가
        "kl_warmup_ratio" : [kl_warmup_ratio],
        "norm" : [norm]
    })
    
    file_exists = os.path.exists(trial_res_file)
    mode   = "a" if file_exists else "w"
    header = not file_exists
        
    res_df.to_csv(trial_res_file, mode=mode, index=False, header=header)

    logger.info(f"Trial {trial.number:3d} | r2={score:+.4f} | k={n_clusters}")
    
    del model
    torch.cuda.empty_cache()
    

    # ── ⑤ Optuna가 최대화할 스코어 반환 ───────────
    return score

def load_hivrc(file_path, cutoff=0.1, normalize=True):
    data_path = Path(f"{file_path}/insight.merged_otus.txt")
    meta_path = Path(f"{file_path}/SupplementaryMaterial.xlsx")

    dat = pd.read_csv(data_path, sep="\t", encoding = "utf-8")
    dat_meta = pd.read_excel(meta_path, header = 1, usecols = "B:L")

    dat_cols = dat_meta["SeqID"].astype(str).to_list()
    dat_cols.insert(0, 'Resphera Insight (Raw Counts)')
    dat_T = dat[dat_cols].T
    dat_T.reset_index(inplace = True)
    dat_T.columns = dat_T.iloc[0].astype(str)
    dat_T = dat_T.iloc[1:]
    dat_T.columns.values[0] = "SeqID"
    dat_merged = pd.merge(dat_T, dat_meta, on = "SeqID", how = "inner")

    X = dat_merged.iloc[:, 1:(dat_T.shape[1])].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    dat_cov = dat_merged[['hivstatus', 'Age', 'gender']].reset_index(drop=True)
    
    dat_cov['Age'] = dat_cov['Age'].fillna(dat_cov['Age'].median())
    # msm=1인 경우 gender=1로 채우기
    mask_msm = dat_cov['gender'].isna() & (dat_merged['msm'] == 1)
    dat_cov.loc[mask_msm, 'gender'] = 1

    # 나머지 NaN은 최빈값으로
    dat_cov['gender'] = dat_cov['gender'].fillna(dat_cov['gender'].mode()[0])
    
    dat_batch_lbl = dat_merged['Study'].reset_index(drop=True)
    
    # prevalence filtering
    prevalence = (X > 0).sum(axis=0) / X.shape[0]
    X = X.loc[:, prevalence > cutoff]
    # relative abundance
    if normalize:
        X = X.div(X.sum(axis=1), axis=0)
        
    assert len(X) == len(dat_cov), f"샘플 수 불일치: X={len(X)}, dat_cov={len(dat_cov)}"
    

    return X, dat_cov, dat_batch_lbl
    

def main():
    # WORKDIR = "C:/Users/KOBIC/Documents/latentgee/examples"
    WORKDIR = "/DATA/WGS_study/YSL/projects/latentgee/examples"    
    LOGDIR = f"{WORKDIR}/logs"
    RESDIR = f"{WORKDIR}/results"
    

    # ── seed 고정
    set_seed()
    # ── Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    
    # ---- 1. YAML 로드 ----    
    with open(f"{WORKDIR}/config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # ── logfile 생성
    logger = setup_logger(log_dir=LOGDIR, cutoff = "multi") # ── 로거 설정
    logger.info("LatentGEE prototype 시작")  # 이 줄도 추가하면 좋아요
    trial_res_file = save_dated_filename(f"{LOGDIR}/optuna_trials", ".csv")
        
    # ── 모듈 패키지 기록
    logger.info(f"python == {sys.version.split()[0]}")
    logger.info(f"torch == {getattr(torch, '__version__', None)}")
    logger.info(f"numpy == {getattr(np, '__version__', None)}")
    logger.info(f"scikit-learn == {getattr(sklearn, '__version__', None)}")
    logger.info(f"optuna == {getattr(optuna, '__version__', None)}")
    try:
        import importlib.metadata
        logger.info(f"hdbscan == {importlib.metadata.version('hdbscan')}")
    except importlib.metadata.PackageNotFoundError:
        logger.info("hdbscan == (version unknown)")
        
    
        
    # ---- 2. data cashing & load ----
    # DATA_PATH = "F:/졸업 후 연구/LatentGEE/Data"
    DATA_PATH = "/DATA/WGS_study/YSL/projects/Data"   
    cutoff_list = cfg["data"]["zero_prevalence_cutoff"] 
    _ , dat_cov, real_batch = load_hivrc(file_path=DATA_PATH, cutoff=0.1, normalize=True)
    covariates_df = dat_cov
    study_label = real_batch
    
    X_tensor_cache = {}
    for c in cutoff_list:
        X_raw_c, _, _ = load_hivrc(file_path=DATA_PATH, cutoff=c)
        X_tensor_cache[c] = torch.tensor(X_raw_c.values, dtype=torch.float32)
        
    # ---- 3. Optuna 실행 ----
    tuning_cfg = cfg["tuning"]
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=tuning_cfg["pruner_startup_trials"],
        n_warmup_steps  =tuning_cfg["pruner_warmup_steps"],
        interval_steps  =tuning_cfg["pruner_interval_steps"],
    ) if tuning_cfg["pruner"] else optuna.pruners.NopPruner()

    study = optuna.create_study(
        direction  ="minimize",
        sampler    =optuna.samplers.TPESampler(seed=tuning_cfg["seed"]),
        pruner     =pruner,
        study_name =tuning_cfg["study_name"],
    )
    study.optimize(
        lambda tr: objective(tr, cfg, X_tensor_cache, real_batch=study_label,
                             trial_res_file=trial_res_file, cutoff_list=cutoff_list,
                             covariates_df=covariates_df, logger=logger),
        n_trials=tuning_cfg["n_trials"],
        callbacks=[make_save_best_callback(logger, LOGDIR, cutoff_list)]
    )
        
    # ---- 4. Best hyper-parameter ----
    best_trial  = study.best_trial
    best_params = best_trial.params
        
    logger.info("Best hyperparameters")
    logger.info(best_params)

    # ── 4-1. 파라미터 꺼내기
    best_cutoff = best_params["cutoff"]   
    base_dim       = best_params["base_dim"]
    latent_dim     = best_params["latent_dim"]
    n_layers       = best_params["n_layers"]
    strategy       = best_params["strategy"]
    dropout_rate   = best_params["dropout_rate"]
    activation     = best_params["activation"]
    epochs         = best_params["epochs"]
    batch_size     = best_params["batch_size"]
    lr             = best_params["learning_rate"]
    min_cs         = best_params["min_cluster_size"]
    beta_kl        = best_params["beta_kl"]
    weight_decay   = best_params["weight_decay"]
    grad_clip_norm = best_params["grad_clip_norm"]
    kl_warmup_ratio= best_params["kl_warmup_ratio"]
    norm           = best_params["norm"]
    init           = best_params["init"]
    
    min_samples    = None

    # ── 4-2. 모델 인스턴스 & 학습
    X_raw_best, _, _ = load_hivrc(file_path=DATA_PATH, cutoff=best_cutoff)
    X_tensor_best = torch.tensor(X_raw_best.values, dtype=torch.float32).to(device)
    input_dim = X_tensor_best.shape[1]
    
    best_model = VAE(
        input_dim   = input_dim,
        latent_dim  = latent_dim,
        n_layers    = n_layers,
        base_dim    = base_dim,
        strategy    = strategy,
        dropout_rate= dropout_rate,
        activation  = activation,
        norm        = norm,
    ).to(device)
    apply_init(best_model, init)
    train_vae(best_model, X_tensor_best,
              beta_kl       =beta_kl,
              epochs        =epochs,
              lr            =lr,
              batch_size    =batch_size,
              weight_decay  =weight_decay,
              grad_clip_norm=grad_clip_norm,
              kl_warmup_ratio=kl_warmup_ratio,
              logger=logger)
    

    # ── 4-3. encode → z_tilde (GEE residualization) → decode
    best_model.eval()

    with torch.no_grad():
        # a) encode
        mu_z, logvar_z = best_model.encode(X_tensor_best)
        # b) reparameterize
        z = best_model.reparameterize(mu_z, logvar_z)

    z_np = z.cpu().numpy()

    # c) pseudo-batch clustering
    labels = pseudo_clustering(
        z,
        min_cluster_size=min_cs,
        min_samples     =min_samples,
        metric          ="euclidean"
    )

    # d) noise 샘플 제거
    noise_mask = labels == -1
    valid_mask = labels != -1
    
    z_used   = z_np[valid_mask]
    lbl_used = labels[valid_mask]
    
    noise_ratio = (labels == -1).mean()
    n_clusters  = len(np.unique(lbl_used))
    logger.info(f"  pseudo-batch 클러스터 수 : {n_clusters}")
    logger.info(f"  노이즈 비율              : {noise_ratio:.2%}")

    if n_clusters < 2:
        raise ValueError("클러스터가 2개 미만 — min_cluster_size 또는 학습 파라미터 재검토 필요")
    
    # e) GEE residualization → z_tilde (valid 샘플)
    cov_used = covariates_df[valid_mask].reset_index(drop=True) if covariates_df is not None else None
    z_tilde_valid = gee_latent_residual(z_used, lbl_used, covariates_df=cov_used)
    
    z_tilde_all = np.zeros_like(z_np)          # ← 추가
    z_tilde_all[valid_mask] = z_tilde_valid    # ← 추가

    # f) noise 샘플 보정 (가장 가까운 클러스터 평균 빼기)
    if noise_mask.sum() > 0:
        # 각 클러스터 중심 계산
        cluster_means = {}
        for c in np.unique(lbl_used):
            cluster_means[c] = z_np[valid_mask][lbl_used == c].mean(axis=0)
        
        centers = np.vstack(list(cluster_means.values()))
        
        # noise 샘플과 각 클러스터 중심 거리 계산
        dists = pairwise_distances(z_np[noise_mask], centers)
        nearest = np.argmin(dists, axis=1)
        
        # 가장 가까운 클러스터 평균 빼기
        z_noise_corrected = z_np[noise_mask] - centers[nearest]
        z_tilde_all[noise_mask] = z_noise_corrected

    # g) decode (전체 샘플)
    z_tilde_tensor = torch.tensor(z_tilde_all, dtype=torch.float32).to(device)

    with torch.no_grad():
        pi, mu_x, log_sigma_x = best_model.decode(z_tilde_tensor)
        sigma_x = torch.exp(log_sigma_x)
        X_corrected = (1 - pi) * torch.exp(mu_x + 0.5 * sigma_x ** 2)

    # h) 저장 (전체 샘플 유지)
    corrected_df = pd.DataFrame(
        X_corrected.cpu().numpy(),
        index  = X_raw_best.index,
        columns= X_raw_best.columns
    )

    save_corrected_path = save_dated_filename(f"{RESDIR}/X_corrected_LatentGEE_prototype_cutoff{best_cutoff}", ".csv")
    corrected_df.to_csv(save_corrected_path)
    logger.info(f"Batch correction complete — written to {save_corrected_path}")

    # ── 4-5. best model 저장
    save_path = save_dated_filename(f"{RESDIR}/best_model", ".pt")
    torch.save(best_model.state_dict(), save_path)
    logger.info(f"✓ 모델 가중치를 {save_path} 로 저장했습니다.")


if __name__ == "__main__":
    main()