"""
utils.py
========

Utility functions for logging, file management, and tensor operations used in
diffusion and reconstruction pipelines.

Includes:
    - extract(): broadcast-safe tensor value extraction by timestep.
    - get_logger(): configurable logger with console + file handlers.
    - find_last_weights(): find the latest checkpoint by epoch index.
    - find_idx_already_recon(): track reconstructed indices across methods.
"""

import os
import logging
from logging import StreamHandler, FileHandler
from typing import Dict, List, Tuple, Optional

import torch


# -----------------------------------------------------------------------------
# Tensor Utilities
# -----------------------------------------------------------------------------

def extract(v: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Extract coefficients at specified timesteps and reshape for broadcasting.

    Args:
        v (torch.Tensor): Source tensor, typically precomputed constants of shape [T, ...].
        t (torch.Tensor): Tensor of timestep indices of shape [B].
        x_shape (tuple): Target shape (usually matching x_t) for reshaping.

    Returns:
        torch.Tensor: Values gathered from `v` indexed by `t`, reshaped to
        [B, 1, 1, ...] for broadcast compatibility.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def get_logger(log_dir: str, log_filename: str = "training.log") -> logging.Logger:
    """Create a logger with both console and file handlers.

    Args:
        log_dir (str): Directory where the log file will be saved.
        log_filename (str): Log filename (default: 'training.log').

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger("TrainerLogger")

    # Avoid adding duplicate handlers on multiple calls
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        console_handler = StreamHandler()
        file_handler = FileHandler(log_path, mode="a")

        console_handler.setLevel(logging.INFO)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


# -----------------------------------------------------------------------------
# Checkpoint Management
# -----------------------------------------------------------------------------

def find_last_weights(folder_weights: str) -> Optional[str]:
    """Find the most recent model weights file by epoch number.

    Expects filenames containing an epoch index (e.g., 'model_epoch_100.pth').

    Args:
        folder_weights (str): Directory containing model checkpoint files.

    Returns:
        Optional[str]: Full path to the latest weights file,
        or None if no valid file is found.
    """
    max_epoch = -1
    last_filename = None

    for filename in os.listdir(folder_weights):
        try:
            epoch_num = int(filename.split("_")[-1].split(".")[0])
            if epoch_num > max_epoch:
                max_epoch = epoch_num
                last_filename = filename
        except (ValueError, IndexError):
            # Skip files not following the naming convention
            continue

    return os.path.join(folder_weights, last_filename) if last_filename else None


# -----------------------------------------------------------------------------
# Reconstruction Index Tracking
# -----------------------------------------------------------------------------

def find_idx_already_recon(out_folder: str) -> Tuple[Dict[str, List[int]], List[int]]:
    """Identify which image indices have been reconstructed across methods.

    Args:
        out_folder (str): Root directory containing per-method subfolders.

    Returns:
        Tuple[Dict[str, List[int]], List[int]]:
            - Dictionary mapping each (method_param) to its reconstructed indices.
            - List of indices reconstructed by *all* methods.
    """
    idx_already_recon: Dict[str, List[int]] = {}
    idx_already_recon_all_methods: List[int] = []

    for method in os.listdir(out_folder):
        if method.endswith(".pkl"):
            continue

        path_method = os.path.join(out_folder, method)
        if not os.path.isdir(path_method):
            continue

        for params in os.listdir(path_method):
            path_params = os.path.join(path_method, params)
            if not os.path.isdir(path_params):
                continue

            # Collect all indices from .pth or .npy filenames
            try:
                indices = [int(fn.split(".")[0]) for fn in os.listdir(path_params)]
                idx_already_recon[f"{method}_{params}"] = indices
            except ValueError:
                # Skip files with unexpected naming
                continue

    # Compute intersection of all index sets
    if idx_already_recon:
        indices_sets = [set(v) for v in idx_already_recon.values()]
        idx_already_recon_all_methods = sorted(list(set.intersection(*indices_sets)))

    return idx_already_recon, idx_already_recon_all_methods
