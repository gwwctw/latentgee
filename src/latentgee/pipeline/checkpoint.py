import yaml
import numpy as np
import pandas as pd
import optuna
import os
import warnings


# 베스트 모델 저장/재현
def save_model(model, path="best_model.pt"):
    torch.save(model.state_dict(), path)

def load_model(model_class, path="best_model.pt", *args, device=None,**kwargs):
    model=model_class(*args, **kwargs)
    ckpt=torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt)
    if device: model.to(device)
    model.eval()
    return model