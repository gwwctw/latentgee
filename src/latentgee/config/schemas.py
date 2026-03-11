# src/latentgee/pipeline/runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from latentgee.config.schemas import TrainConfig, EvalConfig
from latentgee.core.model import BaseLatentModel
from latentgee.pipeline.datamodule import DataModule
from latentgee.pipeline.evaluator import BaseEvaluator
from latentgee.pipeline.corrector import BaseCorrector


@dataclass
class RunOutputs:
    X_corr: np.ndarray
    Z: np.ndarray
    Z_corr: np.ndarray
    pseudo_labels: np.ndarray
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]


class LatentGEEPipeline:
    def __init__(
        self,
        model: BaseLatentModel,
        evaluator: BaseEvaluator,
        corrector: BaseCorrector,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
    ):
        self.model = model
        self.evaluator = evaluator
        self.corrector = corrector
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self._is_fit = False

    def fit(self, X: np.ndarray) -> float:
        dm = DataModule(X, self.train_cfg)
        loss = self.model.fit(dm.train_loader(), self.train_cfg)
        self._is_fit = True
        return float(loss)

    def fit_transform(
        self,
        X: np.ndarray,
        *,
        covariates: Optional[np.ndarray] = None,
        batch_labels_for_after: Optional[np.ndarray] = None,
    ) -> RunOutputs:
        # 1) train
        dm = DataModule(X, self.train_cfg, covariates=covariates)
        _ = self.model.fit(dm.train_loader(), self.train_cfg)
        self._is_fit = True

        X_np = dm.X_numpy()

        # 2) encode
        Z = self.model.encode(X_np)

        # 3) pseudo-batch
        pseudo = self.evaluator.pseudo_batch(Z)

        # 4) metrics before (pseudo 기준)
        metrics_before = self.evaluator.batch_metrics(X_np, pseudo)

        # 5) correct latent (covariates only here)
        Z_corr = self.corrector.correct(Z, groups=pseudo, covariates=covariates)

        # 6) decode
        X_corr = self.model.decode(Z_corr)

        # 7) metrics after (선택: 실제 batch 라벨이 있으면 그걸로 평가)
        if batch_labels_for_after is None:
            metrics_after = self.evaluator.batch_metrics(X_corr, pseudo)
        else:
            metrics_after = self.evaluator.batch_metrics(X_corr, np.asarray(batch_labels_for_after))

        return RunOutputs(
            X_corr=X_corr,
            Z=Z,
            Z_corr=Z_corr,
            pseudo_labels=pseudo,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
        )