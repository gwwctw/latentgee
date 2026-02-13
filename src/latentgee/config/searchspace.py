


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
    
    
    
def suggest_auto(trial: optuna.Trial, name: str, spec):
    """YAML spec을 보고 Optuna의 suggest_*를 자동 선택."""
    if isinstance(spec, dict):
        if "loguniform" in spec:
            low, high = spec["loguniform"]
            return trial.suggest_float(name, float(low), float(high))
        raise ValueError(f"Unsupported dict spec for {name}: {spec}")
    if isinstance(spec, (list, tuple)):
        vals = [_as_none(v) for v in spec]
        if len(vals) == 2 and all(isinstance(v, numbers.Number) for v in vals):
            low, high = vals
            if float(low).is_integer() and float(high).is_integer():
                return trial.suggest_int(name, int(low), int(high))
            else:
                return trial.suggest_float(name, float(low), float(high))
        return trial.suggest_categorical(name, vals)
    return spec  # 단일 고정값