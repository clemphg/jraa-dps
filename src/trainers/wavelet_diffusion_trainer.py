"""
wavelet_diffusion_trainer.py
============================

Trainer for Gaussian diffusion in the 3D wavelet domain.
Implements the forward noising process and denoising target for model training.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from src.operators.wavelet import Wavelet
from src.trainers.trainer import Trainer
from src.utils.trainer import extract


class WaveletDiffusionTrainer(Trainer):
    """Trains a Gaussian diffusion model in the 3D wavelet domain."""

    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        model: nn.Module,
        loss: nn.Module,
        optimizer,
        scheduler=None,
        epochs: int = 1000,
        start_epoch: int = 0,
        grad_clip: Optional[float] = 1e-1,
        weights_dir: str = "./weights",
        log_dir: str = "./logs",
        beta1: float = 1e-4,
        betaT: float = 0.02,
        T: int = 1000,
        ema_rate: Optional[float] = None,
        display_progress: bool = False,
        device: str = "cuda",
    ):
        super().__init__(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model=model,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            start_epoch=start_epoch,
            grad_clip=grad_clip,
            weights_dir=weights_dir,
            log_dir=log_dir,
            ema_rate=ema_rate,
            display_progress=display_progress,
        )

        self.device = device
        self.T = T

        # Diffusion schedule
        self.betas = torch.linspace(beta1, betaT, T).float().to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_bar = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar).to(device)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar).to(device)

        # Wavelet transform
        self.wavelet_op = Wavelet()

    # ------------------------------------------------------------------ #
    # Core Diffusion Step
    # ------------------------------------------------------------------ #

    def one_step(self, packed) -> Tuple[torch.Tensor, torch.Tensor]:
        """One diffusion step: forward noising + prediction.

        Args:
            packed: A batch tensor `y0` (clean images).

        Returns:
            (x_pred, x_tgt): predicted and true wavelet-domain clean signals.
        """
        y0 = packed.to(self.device)
        x0 = self.wavelet_op.transform(y0, div_low_pass=3)

        b, c, d, h, w = x0.shape
        t = torch.randint(self.T, size=(b,)).to(self.device)

        noise_img = torch.randn_like(y0).to(self.device)
        noise = self.wavelet_op.transform(noise_img)

        xt = (
            extract(self.sqrt_alphas_bar, t, (b, c, d, h, w)) * x0
            + extract(self.sqrt_one_minus_alphas_bar, t, (b, c, d, h, w)) * noise
        )

        x_pred = self.model(xt, t)
        x_tgt = x0
        return x_pred, x_tgt
