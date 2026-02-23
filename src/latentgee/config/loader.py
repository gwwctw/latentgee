
# ====== 1) YAML 헬퍼 ======    
def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
  
def _as_none(x):
    return None if (isinstance(x, str) and x.lower() == "none") else x

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
