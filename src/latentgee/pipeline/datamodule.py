class DataModule:
    def __init__(self, X: np.ndarray, train_cfg: TrainConfig):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        self.X = torch.tensor(X, dtype=torch.float32)
        self.train_cfg = train_cfg

    def make_loader(self) -> DataLoader:
        ds = TensorDataset(self.X.cpu())
        return DataLoader(
            ds,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=self.train_cfg.num_workers,
            prefetch_factor=4 if self.train_cfg.num_workers > 0 else None,
            persistent_workers=(self.train_cfg.num_workers > 0),
            pin_memory=(self.train_cfg.device == "cuda"),
            drop_last=True,
        )