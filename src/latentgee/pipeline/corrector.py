# NEW

# src/latentgee/pipeline/corrector.py
from __future__ import annotations

from typing import Optional
import numpy as np

from latentgee.core.latent_correction import gee_latent_residual


class BaseCorrector:
    def correct(self, 
                Z: np.ndarray,
                groups: np.ndarray,
                covariates: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError
    

class GEECorrector(BaseCorrector):
    """
    Latent residualization using GEE.
    covariates are adjusted ONLY here (as you wanted).
    """
    
    def correct(self, Z:np.ndarray, groups:np.ndarray, covariates: Optional[np.ndarray] = None) -> np.ndarray:
        Z = np.asarray()
        groups = np.asarray(groups)
        
        return gee_latent_residual(Z, groups, covariates=covariates).astype("float32")