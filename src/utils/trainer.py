"""
utils/trainer.py
================

General-purpose helper utilities for training loops, including:
EMA updates, gradient clipping, and alpha-schedule extract helpers.
"""

import torch


def update_ema(target_params, source_params, rate: float = 0.99) -> None:
    """Exponential moving average update."""
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def clip_gradients(parameters, max_norm: float) -> None:
    """Clip gradients in-place to a maximum norm."""
    torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """Extract coefficients for batch indices `t` and reshape for broadcasting."""
    b = t.shape[0]
    out = a.gather(-1, t).reshape(b, *((1,) * (len(x_shape) - 1)))
    return out
