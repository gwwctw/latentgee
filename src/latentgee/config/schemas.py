# src/latentgee/config/schemas.py
from dataclasses import dataclass, field
from typing import Optional
import os
import torch


@dataclass
class ModelConfig:
    input_dim: int
    latent_dim: int = 16
    n_layers: int = 2
    base_dim: int = 128
    strategy: str = "constant"
    dropout: float = 0.0
    activation: str = "relu"


@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 256
    num_workers: int = max(0, (os.cpu_count() or 4) - 2)
    amp: bool = field(default_factory=lambda: torch.cuda.is_available())


@dataclass
class EvalConfig:
    hdb_min_cluster_size: int = 10
    hdb_min_samples: Optional[int] = None
    hdb_metric: str = "braycurtis"
    allow_noise: bool = True
    permanova_metric: str = "braycurtis"
    permanova_permutations: int = 999
    
@dataclass
class TuningConfig:
    n_trials: int = 50
    timeout : Optional[int] = None
    
    pruner: bool = True
    pruner_type: str = "median"
    pruner_startup_trials: int = 5
    pruner_warmup_steps: int = 0
    pruner_interval_steps: int = 1
    
    sampler: str = "tpe"
    seed: int = 42
    
    study_name: str = "latentgee_optuna"