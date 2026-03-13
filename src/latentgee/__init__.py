# config
from .config.loader import load_cfg, _as_none, suggest_auto, LatentGEEDataModule
from .config.schemas import ModelConfig, TrainConfig, EvalConfig, TuningConfig
from .config.searchspace import ModelSearchSpace, TrainingSearchSpace, ClusteringSearchSpace

# core
from .core.gee import gee_residual
from .core.latent_correction import gee_latent_residual, decode_batch_corrected_latent
from .core.losses import ziln_nll
from .core.model import BaseLatentModel, LatentGEEModule
from .core.training import train_vae 
from .core.vae import FlexibleMLP, VAE

# pipeline
from .pipeline.checkpoint import save_model, load_model
from .pipeline.corrector import GEECorrector, BaseCorrector
from .pipeline.datamodule import DataModule
from .pipeline.evaluator import BaseEvaluator, HDBSCANBatchEvaluator
from .pipeline.pipeline import LatentGEEPipeline, RunOutputs
from .pipeline.tuner import OptunaTuner

# utils
from .utils.dataset_cache import get_dataset_for_cutoff
from .utils.matrics import safe_silhouette, permanova_r2, adonis2_permanova_r2_via_rscript, evaluate_latentgee, BatchEffectEvaluator

__all__ = [
    # config
    "load_cfg", "_as_none", "suggest_auto",
    "LatentGEEDataModule", "ModelConfig", "TrainConfig", "EvalConfig", "TuningConfig",
    "ModelSearchSpace", "TrainingSearchSpace", "ClusteringSearchSpace",
    
    # core
    "gee_residual"
    "gee_latent_residual", "decode_batch_corrected_latent",
    "ziln_nll",
    "BaseLatentModel", "LatentGEEModule",
    "train_vae",
    "FlexibleMLP", "VAE"
    
    # pipieline
    "save_model", "load_model",
    "GEECorrector", "BaseCorrector",
    "DataModule",
    "BaseEvaluator", "HDBSCANBatchEvaluator",
    "LatentGEEPipeline", "RunOutputs",
    "OptunaTuner",
    
    # utils
    "get_dataset_for_cutoff",
    "safe_silhouette", "permanova_r2", "adonis2_permanova_r2_via_rscript", "evaluate_latentgee", "BatchEffectEvaluator"
    ]
