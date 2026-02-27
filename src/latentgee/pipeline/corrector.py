# NEW
from __future__ import annotations

from typing import Optional
import numpy as np

from latentgee.core.gee import gee_residual


class BaseCorrector:
    def correct(self, Z: np.ndarray, groups: np.ndarray, covariates: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

class GEECorrector(BaseCorrector):
    """
    Latent residualization using GEE
    """
    
    def correct(self: Z:np.ndarray)





class LatentCorrector:

    def correct(
        self,
        Z: np.ndarray,
        groups: np.ndarray,
        covariates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        
        Z_resid = gee_latent_residual(
            Z,
            groups,
            covariates=covariates
        )
        return Z_resid