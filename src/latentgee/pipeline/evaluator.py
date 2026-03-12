# src/latentgee/pipeline/evaluator.py
import numpy as np
from typing import Tuple, Dict

from sklearn.preprocessing import StandardScaler

from latentgee.config.schemas import EvalConfig
from latentgee.utils.matrics import safe_silhouette, permanova_r2

try:
    import hdbscan
except ImportError:  # optional dependency guard
    hdbscan = None


class BaseEvaluator:
    def pseudo_batch(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def batch_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        raise NotImplementedError


class HDBSCANBatchEvaluator(BaseEvaluator):
    def __init__(self, eval_cfg: EvalConfig, standardize_latent: bool = True):
        self.cfg = eval_cfg
        self.standardize_latent = standardize_latent

    def pseudo_batch(self, Z: np.ndarray) -> np.ndarray:
        if hdbscan is None:
            raise ImportError("hdbscan is not installed. Please install: pip install hdbscan")

        Z_in = Z.astype("float64", copy=False)
        if self.standardize_latent:
            Z_in = StandardScaler().fit_transform(Z_in)

        kwargs = dict(
            min_cluster_size=self.cfg.hdb_min_cluster_size,
            min_samples=self.cfg.hdb_min_samples,
            metric=self.cfg.hdb_metric,
            cluster_selection_method="eom",
        )

        # cosine sometimes needs algorithm="generic"
        if self.cfg.hdb_metric == "cosine":
            clusterer = hdbscan.HDBSCAN(algorithm="generic", **kwargs)
        else:
            clusterer = hdbscan.HDBSCAN(**kwargs)

        labels = clusterer.fit_predict(Z_in)
        return labels.astype(int)

    def batch_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        X_in = X.astype("float64", copy=False)
        X_std = StandardScaler().fit_transform(X_in)

        sil = safe_silhouette(X_std, labels, metric=self.cfg.hdb_metric, allow_noise=self.cfg.allow_noise)
        _, r2 = permanova_r2(
            X_std,
            grouping=np.asarray(labels),
            metric=self.cfg.permanova_metric,
            permutations=self.cfg.permanova_permutations,
        )
        return {"silhouette": float(sil), "r2": float(r2)}
    