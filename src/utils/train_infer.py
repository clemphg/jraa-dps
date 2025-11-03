import os
import yaml
import shutil
import numpy as np
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.utils.data.normalization import NormRange, NormSqrt
from src.models.unet_wavelet_diffusion import UNetModel
from src.utils.data.activity_attenuation_dataset import ActivityAttenuationDataset

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(data_cfg, model_cfg, train_cfg):
    """Combine config dicts into a single structure."""
    cfg = {"data": data_cfg, "model": model_cfg, **train_cfg}
    return cfg


def create_weights_logs_dirs(save_root: str, config_paths: list) -> Dict[str, str]:
    """Create output directories and copy config files for reproducibility."""
    weights_dir = os.path.join(save_root, "weights")
    logs_dir = os.path.join(save_root, "logs")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    for p in config_paths:
        shutil.copy(p, os.path.join(save_root, os.path.basename(p)))

    return {"weights": weights_dir, "logs": logs_dir}


def set_seed(seed: int = 123):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_reconstruction(filename: str, save_path: str, data: np.ndarray):
    """Save reconstruction result if not already existing."""
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, filename)
    if not os.path.exists(save_file):
        np.save(save_file, data)


# -------------------------------------------------------------------------
# Dataset Factory
# -------------------------------------------------------------------------

def build_dataloader(cfg: Dict[str, Any], split: str) -> DataLoader:
    """Construct the dataset and dataloader from config."""
    with open(cfg["id_patients_path"], "rb") as f:
        id_patients = torch.load(f) if cfg["id_patients_path"].endswith(".pt") else __import__("pickle").load(f)
    id_patients_split = id_patients[split]

    norm_act = NormSqrt(
        sqrt_order=cfg["norm_act"]["sqrt_order"],
        img_min=cfg["norm_act"]["img_min"],
        img_max=cfg["norm_act"]["img_max"],
        new_min=cfg["norm_act"]["new_min"],
        new_max=cfg["norm_act"]["new_max"],
    )
    norm_atn = NormRange(
        img_min=cfg["norm_atn"]["img_min"],
        img_max=cfg["norm_atn"]["img_max"],
        new_min=cfg["norm_atn"]["new_min"],
        new_max=cfg["norm_atn"]["new_max"],
    )

    dataset_train = ActivityAttenuationDataset(
        path=cfg["path_data"],
        id_patients=id_patients_split,
        nb_slices=cfg["nb_slices"],
        to_tensor=True,
        norm_method_act=norm_act,
        norm_method_atn=norm_atn,
        clip_act=cfg["clip_act"],
        clip_atn=cfg["clip_atn"],
    )

    return DataLoader(
        dataset_train,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=cfg.get("num_workers", 6),
        pin_memory=True,
    )


# -------------------------------------------------------------------------
# Model Factory
# -------------------------------------------------------------------------

def build_model(cfg: Dict[str, Any], device: str) -> torch.nn.Module:
    """Initialize model from configuration dictionary."""
    model = UNetModel(**cfg["params"])
    model.to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.")
    return model
