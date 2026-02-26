from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from latentgee.config.schema import ModelConfig, TrainConfig
from latentgee.core.vae import VAE
from latentgee.core.losses import ziln_nll

class BaseLatentModel(nn.Module):
    """stable internal interface. Keep this API stable for long-term maintenance."""
    
    def fit(self, loader: DataLoader, train_cfg: TrainConfig) -> float:
        raise NotImplementedError
    
    @torch.no_grad()
    def encode(self, X:np.ndarray) -> np.ndarray:
        """Return latent mean μ (N, k) as numpy."""
        raise NotImplementedError
    
    @torch.no_grad()
    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Return latent mean μ (N, k) as numpy."""
        raise NotImplementedError
    

class LatentGEEModule(BaseLatentModel):
    """
    Model wrapper only: owns VAE + provides fit/encode/decode.
    Correction/clustering lives outside (pipeline layer).
    """
    
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.vae = VAE(
            input_dim = model_cfg.input_dim,
            latent_dim = model_cfg.latent_dim,
            n_layers = model_cfg.n_layers,
            base_dim = model_cfg.base_dim,
            strategy = model_cfg.strategy,
            dropout_rate = model_cfg.dropout,
            activation = model_cfg.activation,
        )
    
    def forward(self, x:torch.Tensor):
        return self.vae(x)
    
    def fit(self, loader: DataLoader, train_cfg: TrainConfig) -> float:
        device = torch.device(train_cfg.device)
        self.to(device)
        self.train()
        
        opt = torch.optim.Adam(self.parameters(), lr=train_cfg.lr)
        use_amp = bool(train_cfg.amp) and (device.type == "cuda")
        scaler = GradScaler(device="cuda", enabled = use_amp)
        
        last_loss = float("inf")
        for _ep in range(train_cfg.epochs):
            for (xb, ) in loader:
                xb = xb.to(device, non_blocking=use_amp)