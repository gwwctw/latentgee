import os
import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
import warnings

from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.cov_struct import Exchangeable


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

