# src/latentgee/core/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from latentgee.config.schemas import ModelConfig, TrainConfig
from latentgee.core.vae import VAE
from latentgee.core.losses import ziln_nll


class BaseLatentModel(nn.Module):
    """Stable internal interface. Keep this API stable for long-term maintenance."""

    def fit(self, loader: DataLoader, train_cfg: TrainConfig) -> float:
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Return latent mean μ (N, k) as numpy."""
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Decode latent Z (N, k) -> X_hat (N, D) as numpy."""
        raise NotImplementedError


class LatentGEEModule(BaseLatentModel):
    """
    Model wrapper only: owns VAE + provides fit/encode/decode.
    Correction/clustering lives outside (pipeline layer).
    """

    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.vae = VAE(
            input_dim=model_cfg.input_dim,
            latent_dim=model_cfg.latent_dim,
            n_layers=model_cfg.n_layers,
            base_dim=model_cfg.base_dim,
            strategy=model_cfg.strategy,
            dropout_rate=model_cfg.dropout,
            activation=model_cfg.activation,
        )

    def forward(self, x: torch.Tensor):
        return self.vae(x)

    def fit(self, loader: DataLoader, train_cfg: TrainConfig) -> float:
        device = torch.device(train_cfg.device)
        self.to(device)
        self.train()

        opt = torch.optim.Adam(self.parameters(), lr=train_cfg.lr)
        use_amp = bool(train_cfg.amp) and (device.type == "cuda")
        scaler = GradScaler(device="cuda", enabled=use_amp)

        last_loss = float("inf")

        for _ep in range(train_cfg.epochs):
            for (xb,) in loader:
                xb = xb.to(device, non_blocking=use_amp)

                with autocast(device_type="cuda", enabled=use_amp):
                    (pi, mu_x, log_sigma_x), mu_z, logvar_z, _ = self.vae(xb)
                    recon = ziln_nll(xb, pi, mu_x, log_sigma_x)
                    kl = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
                    loss = recon + kl

                if torch.isnan(loss):
                    return float("inf")

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                last_loss = float(loss.detach().cpu().item())

        return last_loss

    @torch.no_grad()
    def encode(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        X_tensor = torch.from_numpy(X).float().to(device, non_blocking=(device.type == "cuda"))
        mu, _ = self.vae.encode(X_tensor)
        return mu.detach().cpu().numpy().astype("float32")

    @torch.no_grad()
    def decode(self, Z: np.ndarray) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        z_tensor = torch.from_numpy(Z).float().to(device)

        pi, mu_x, log_sigma_x = self.vae.decode(z_tensor)
        sigma = log_sigma_x.exp()
        x_hat = (1 - pi) * torch.exp(mu_x + 0.5 * sigma * sigma)  # E[X|z] (ZILN)
        return x_hat.detach().cpu().numpy().astype("float32")