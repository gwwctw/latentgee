

# -----------------------
# Optuna Tuner
# -----------------------
class OptunaTuner:
    def __init__(self, X: np.ndarray, base_model_cfg: ModelConfig, train_cfg: TrainConfig, eval_cfg: EvalConfig):
        self.X = X
        self.base_model_cfg = base_model_cfg
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg

    def _suggest(self, trial: optuna.Trial) -> ModelConfig:
        # 예시: latent_dim, base_dim, n_layers, dropout, activation 탐색
        latent_dim = trial.suggest_int("latent_dim", 8, 64, step=8)
        base_dim   = trial.suggest_categorical("base_dim", [64, 128, 256, 512])
        n_layers   = trial.suggest_int("n_layers", 1, 3)
        dropout    = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
        activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "gelu"])
        strategy   = trial.suggest_categorical("strategy", ["constant", "halve", "double"])
        return ModelConfig(
            input_dim=self.base_model_cfg.input_dim,
            latent_dim=latent_dim,
            n_layers=n_layers,
            base_dim=base_dim,
            strategy=strategy,
            dropout=dropout,
            activation=activation,
        )
    
    def objective(self, trial: optuna.Trial) -> float:
        # 단일 목표: pseudo-batch silhouette (↓)
        model_cfg = self._suggest(trial)
        dm = DataModule(self.X, self.train_cfg)
        loader = dm.make_loader()

        model = LatentGEEModule(model_cfg)
        _ = model.fit(loader, self.train_cfg)

        evaluator = BatchEffectEvaluator(self.eval_cfg)
        _, sil = evaluator.silhouette_from_model(model, self.X)

        trial.report(float(sil), step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return float(sil)

    def tune(self, tune_cfg: TuningConfig) -> Tuple[optuna.Study, Dict[str, Any]]:
        pruner = optuna.pruners.MedianPruner() if tune_cfg.pruner else optuna.pruners.NopPruner()
        study = optuna.create_study(direction="minimize", pruner=pruner)  # silhouette ↓
        study.optimize(self.objective, n_trials=tune_cfg.n_trials, timeout=tune_cfg.timeout)
        return study, study.best_params
"""
    def objective_multi(self, trial: optuna.Trial, y_bio: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Multi-objective: (sil_batch ↓, r2_batch ↓, sil_bio ↑, r2_bio ↑)
        """
        model_cfg = self._suggest(trial)
        dm = DataModule(self.X, self.train_cfg)
        loader = dm.make_loader()

        model = LatentGEEModule(model_cfg)
        _ = model.fit(loader, self.train_cfg)

        # batch metrics
        evaluator = BatchEffectEvaluator(self.eval_cfg)
        labels_batch, sil_batch = evaluator.silhouette_from_model(model, self.X)

        X_std = StandardScaler().fit_transform(self.X)
        _, r2_batch = permanova_r2(
            X_std, grouping=labels_batch,
            metric=self.eval_cfg.permanova_metric,
            permutations=self.eval_cfg.permanova_permutations,
        )

        # bio metrics (encoder μ 기준)
        with torch.no_grad():
            mu = model.encode_mu(torch.tensor(self.X, dtype=torch.float32, device=self.train_cfg.device))
        Z = mu.detach().cpu().numpy()
        Z_std = StandardScaler().fit_transform(Z)

        sil_bio = silhouette_score(Z_std, y_bio, metric="euclidean")
        _, r2_bio = permanova_r2(
            Z_std, grouping=y_bio,
            metric=self.eval_cfg.permanova_metric,
            permutations=self.eval_cfg.permanova_permutations,
        )

        return float(sil_batch), float(r2_batch), float(sil_bio), float(r2_bio)
"""