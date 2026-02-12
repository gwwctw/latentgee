


# =========================
# 모델 탐색 공간
# =========================
@dataclass
class ModelSearchSpace:
    n_layers: list[int]
    latent_dim: list[int]
    base_dim: list[int]
    strategy: list[str]
    dropout_rate: list[float]
    beta_kl: list[float]
    kl_warmup_epochs: list[int]
    norm: list[str]
    init: list[str]

# =========================
# 학습 탐색 공간
# =========================
@dataclass
class TrainingSearchSpace:
    epochs: list[int]
    batch_size: list[int]
    learning_rate: dict
    weight_decay: dict
    optimizer: list[str]
    amp: list[bool]
    grad_clip_norm: list[float]
    scheduler: list[str]
    warmup_epochs: list[int]
# =========================
# 클러스터링 탐색 공간
# =========================
@dataclass
class ClusteringSearchSpace:
    min_cluster_size: list[int]
    min_samples: list[Any]     # "None"이 문자열로 들어오므로 Any
    metric: list[str]
    cluster_selection_method: list[str]
    allow_single_cluster: list[bool]