
# NEW
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