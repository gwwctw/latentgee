import optuna


# ====== 1) YAML 헬퍼 ======    
def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
  
def _as_none(x):
    return None if (isinstance(x, str) and x.lower() == "none") else x

def suggest_auto(trial: optuna.Trial, name: str, spec):
    """YAML spec을 보고 Optuna의 suggest_*를 자동 선택."""
    if isinstance(spec, dict):
        if "loguniform" in spec:
            low, high = spec["loguniform"]
            return trial.suggest_float(name, float(low), float(high))
        raise ValueError(f"Unsupported dict spec for {name}: {spec}")
    if isinstance(spec, (list, tuple)):
        vals = [_as_none(v) for v in spec]
        if len(vals) == 2 and all(isinstance(v, numbers.Number) for v in vals):
            low, high = vals
            if float(low).is_integer() and float(high).is_integer():
                return trial.suggest_int(name, int(low), int(high))
            else:
                return trial.suggest_float(name, float(low), float(high))
        return trial.suggest_categorical(name, vals)
    return spec  # 단일 고정값

class LatentGEEDataModule:
    def __init__(
        self,
        X: np.ndarray,
        train_cfg: TrainConfig,
        covariates: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ):
        assert isinstance(X, np.ndarray)

        self.X = torch.from_numpy(X).float()
        self.train_cfg = train_cfg

        self.covariates = (
            torch.from_numpy(covariates).float()
            if covariates is not None
            else None
        )

        self.groups = (
            torch.from_numpy(groups).long()
            if groups is not None
            else None
        )
    def train_loader(self) -> DataLoader:
        ds = TensorDataset(self.X)
        return DataLoader(
            ds,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=self.train_cfg.num_workers,
            pin_memory=(self.train_cfg.device == "cuda"),
            drop_last=True,
        )
    
    def get_gee_data(self):
        assert self.groups is not None, "Pseudo-batch labels required for GEE"

        return {
            "X": self.X.cpu().numpy(),
            "groups": self.groups.cpu().numpy(),
            "covariates": (
                self.covariates.cpu().numpy()
                if self.covariates is not None
                else None
            ),
        }