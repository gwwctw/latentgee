import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from latentgee.config import ModelConfig, TrainConfig


# =========================
# LatentGEEModule (Model + Train loop)
# =========================
class LatentGEEModule(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        # VAE 객체 생성 (LatentGEE.py에 있는 VAE 클래스) 
        self.vae = VAE(
            input_dim=model_cfg.input_dim,
            latent_dim=model_cfg.latent_dim,
            n_layers=model_cfg.n_layers,
            base_dim=model_cfg.base_dim,
            strategy=model_cfg.strategy,
            dropout_rate=model_cfg.dropout,
            activation=model_cfg.activation,
        )

    def forward(self, x):
        """
        VAE forward 호출 (encode -> reparameterize -> decode)
        """
        return self.vae(x)
    
    
    def fit(self, X_tensor:torch.Tensor, train_cfg: TrainConfig) -> float: 
        loader = make_dataloader(
            X_tensor,
            batch_size = train_cfg.batch_size,
            suffle = True,
            num_workers = getattr(train_cfg, "num_workers", 0),
            pin_memory = (train_cfg.device == "cuda"),
        )
        
        return self.fit(loader, train_cfg)
            
    """
    def fit(self, loader: DataLoader, train_cfg: TrainConfig) -> float:
        
        # 주어진 DataLoader로 VAE 학습 수행 (ZILN NLL + KL 손실).
        # 마지막 배치의 loss(float)를 반환.
        
        device = train_cfg.device
        self.to(device)
        self.train()

        opt = torch.optim.Adam(self.parameters(), lr=train_cfg.lr)
        use_amp = train_cfg.amp and (device == "cuda")
        scaler = GradScaler(device="cuda", enabled=use_amp)

        last_loss = float("inf")

        for ep in range(train_cfg.epochs):
            for (xb,) in loader:
                xb = xb.to(device, non_blocking=use_amp)
                with autocast(device_type="cuda", enabled=use_amp):
                    (pi, mu_x, logsigma_x), mu_z, logvar_z, _ = self.vae(xb)
                    recon_nll = ziln_nll(xb, pi, mu_x, logsigma_x)
                    kl = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
                    loss = recon_nll + kl

                if torch.isnan(loss):
                    return float("inf")

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                last_loss = float(loss.detach().cpu().item())

        return last_loss
    """
    
    
    @torch.no_grad()
    def encode_mu(self, X_tensor: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
        """입력 X를 μ(latent mean)으로 인코딩."""
        self.eval()
        dev = device or next(self.parameters()).device
        mu, _ = self.vae.encode(X_tensor.to(dev, non_blocking=(dev.type == "cuda")))
        return mu.detach().cpu()

    @torch.no_grad()
    def decode_from_latent(self, z: np.ndarray) -> np.ndarray:
        """latent z에서 X를 디코딩 (ZILN 기대값으로 복원)."""
        self.eval()
        device = next(self.parameters()).device
        z_tensor = torch.tensor(z, dtype=torch.float32, device=device)
        pi, mu_x, log_sigma_x = self.vae.decode(z_tensor)
        sigma = log_sigma_x.exp()
        x_hat = (1 - pi) * torch.exp(mu_x + 0.5 * sigma * sigma)
        return x_hat.detach().cpu().numpy()
