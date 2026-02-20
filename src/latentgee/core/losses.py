import yaml
import numpy as np
import pandas as pd
import optuna
import os

from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.cov_struct import Exchangeable


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
