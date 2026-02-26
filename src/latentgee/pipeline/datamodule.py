

from torch.utils.data import DataLoader


# numpy 데이터를 받아서 PyTorch 학습용 DataLoader를 만들어주는 래퍼
class DataModule:
    def __init__(
        self,
        X: np.ndarray,
        train_cfg: TrainConfig,
        covariates: Optional[np.ndarray] = None,
    ):
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be numpy array")

        self.X = torch.from_numpy(X).float()
        self.train_cfg = train_cfg

        self.covariates = (
            torch.from_numpy(covariates)
            if covariates is not None
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

    def inference_loader(self) -> DataLoader:
        ds = TensorDataset(self.X)
        return DataLoader(
            ds,
            batch_size=self.train_cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )