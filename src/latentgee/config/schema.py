
# =========================
# Configs
# =========================
@dataclass
class DataConfig:
    zero_prevalence_cutoff: list[float]
    standardize_latent: list[bool]
    tensor_path: str

@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 256
    num_workers: int = max(2, (os.cpu_count() or 4) - 2)
    amp: bool = field(default_factory=lambda: torch.cuda.is_available())

@dataclass
class ModelConfig:
    input_dim: int
    latent_dim: int = 16
    n_layers: int = 2
    base_dim: int = 128
    strategy: str = "constant"   # {'constant','halve','double'}
    dropout: float = 0.0
    activation: str = "relu"

@dataclass
class EvalConfig:
    # HDBSCAN pseudo-batch
    hdb_min_cluster_size: int = 10
    hdb_min_samples: Optional[int] = None
    hdb_metric: str = "braycurtis"  # e.g. 'euclidean','cosine','braycurtis' (하이픈 X)
    allow_noise: bool = True
    # PERMANOVA
    permanova_metric: str = "braycurtis"
    permanova_permutations: int = 999

@dataclass
class TuningConfig:
    n_trials: int = 25
    timeout: Optional[int] = None
    pruner: bool = True
