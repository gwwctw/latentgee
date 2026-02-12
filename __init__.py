"""
LatentGEE
~~~~
Unsupervised batch-effect correction powered by VAE+LatentGEE
"""


from importlib.metadata import version as _v

__all__=[
    "LatentGEE",
    "DataModule",
    "TrainerPipeline",
    "load_dataset",
]

# 버전 자동 추출 (pyproject.toml의 [project] version 사용)
__version__=_v("latentgee-u")

# 하위 모듈에서 공통으로 많이 쓰는 심벌 re-export
from .models.latent_gee import LatentGEE
from .data.datamodule import DataModule, load_dataset
from .pipeline import TrainerPipeline
