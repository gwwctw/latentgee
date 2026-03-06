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


def gee_latent_residual(Z: np.ndarray, 
                        pseudo_batch: np.ndarray, 
                        covariates: Optional[list] = None):
    df = pd.DataFrame(Z, columns=[f"z{i}" for i in range(Z.shape[1])])
    df["pseudo_batch"] = pseudo_batch
    residuals = []
    if covariates is None:        
        indep = f"pseudo_batch"
    else:
        if length(covariates) == 1:
            indep = f"{covariates} + pseudo_batch"
        
        else:
            cov = " + ".join(covariates)
            indep = f"{cov} + pseudo_batch"

    for col in df.columns[:-1]:
        
        model = GEE.from_formula(f"{col} ~ {indep}", groups="pseudo_batch", data=df, family=Gaussian(), cov_struct=Exchangeable())
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

