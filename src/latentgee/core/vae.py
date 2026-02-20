import numpy as np
import optuna
import torch
import torch.nn as nn

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

# Flexible MLP builder with dropout and activation selection
"""
레이어 개수, 레이어 크기 변화 패턴(반감, 증가, 고정), activation, dropout을 전부 hyperparameter화한 커스텀 MLP 생성기. 
Optuna에서 조합을 실험하기 위해 만든 FLEXIBLE한 multi-layer percentron builder.

"""
class FlexibleMLP(nn.Module):
    def __init__(self, 
                 layer_dims, 
                 dropout_rate=0.0, 
                 activation='relu'):
        super().__init__()
        layers = []
        act_fn = self.get_activation_fn(activation)
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(act_fn)
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(dropout_rate))
        self.net = nn.Sequential(*layers)
        
    def get_activation_fn(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, x):
        return self.net(x)

# Network architecture generator
def build_layer_dims(input_dim, output_dim, n_layers, base_dim, strategy='constant'):
    dims = [input_dim]
    current_dim = base_dim
    for _ in range(n_layers):
        dims.append(current_dim)
        if strategy == 'halve':
            current_dim = max(current_dim // 2, output_dim)
        elif strategy == 'double':
            current_dim = min(current_dim * 2, 1024)  # arbitrary upper limit
        elif strategy == 'constant':
            continue
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    dims.append(output_dim)
    return dims

