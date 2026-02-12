import yaml
import numpy as np
import pandas as pd
import optuna
import os

# LatentGEE-U VAE using flexible encoder/decoder
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, n_layers, base_dim,
                 strategy='constant', dropout_rate=0.0, activation='relu'):
        super().__init__()
        # ---------- Encoder ----------
        enc_dims = build_layer_dims(input_dim, latent_dim, n_layers, base_dim, strategy)
        self.enc_net = FlexibleMLP(enc_dims, dropout_rate, activation)
        self.fc_mu     = nn.Linear(enc_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(enc_dims[-1], latent_dim)

        # ---------- Decoder ----------
        dec_dims = build_layer_dims(latent_dim, base_dim, n_layers, base_dim, strategy)
        self.dec_net   = FlexibleMLP(dec_dims, dropout_rate, activation)
        self.dec_pi    = nn.Linear(dec_dims[-1], input_dim)   # zero prob
        self.dec_mu    = nn.Linear(dec_dims[-1], input_dim)   # log-normal μ
        self.dec_logσ  = nn.Linear(dec_dims[-1], input_dim)   # log-normal σ

    def encode(self, x):
        h = self.enc_net(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h  = self.dec_net(z)
        pi = torch.sigmoid(self.dec_pi(h))          # (0 ~ 1)
        mu = self.dec_mu(h)                         # 실수
        logσ = self.dec_logσ(h)                     # 실수
        return pi, mu, logσ    # 3-tuple 반환

    def forward(self, x):
        mu_z, logvar_z = self.encode(x)
        z   = self.reparameterize(mu_z, logvar_z)
        pi, mu_x, logσ_x = self.decode(z)
        return (pi, mu_x, logσ_x), mu_z, logvar_z, z

