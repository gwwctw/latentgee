# src/latentgee/pipeline/datamodule.py
from typing import Optional
import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset

from latentgee.config.schemas import TrainConfig



class DataModule:
    def __init__(self, X: np.ndarray, train_cfg: TrainConfig, covariates: Optional[np.ndarray] = None):
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        self.train_cfg = train_cfg

        self.X = torch.from_numpy(X).float()  # keep on CPU; move in training loop
        self.covariates = covariates  # keep as numpy (GEE stage only)

    def train_loader(self) -> DataLoader:
        ds = TensorDataset(self.X)
        return DataLoader(
            ds,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=self.train_cfg.num_workers,
            pin_memory=(self.train_cfg.device == "cuda"),
            drop_last=True,
            persistent_workers=(self.train_cfg.num_workers > 0),
            prefetch_factor=4 if self.train_cfg.num_workers > 0 else None,
        )

    def inference_loader(self) -> DataLoader:
        ds = TensorDataset(self.X)
        return DataLoader(ds, batch_size=self.train_cfg.batch_size, shuffle=False, num_workers=0)

    def X_numpy(self) -> np.ndarray:
        return self.X.detach().cpu().numpy().astype("float32")
    


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