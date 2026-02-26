"""
trainer_base.py
===============

Generic Trainer class that provides:
- Common training loop
- Gradient clipping
- EMA parameter tracking
- Checkpointing
- Logging and TensorBoard integration
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Callable
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils.utils import get_logger
from src.utils.trainer import update_ema, clip_gradients


class Trainer(ABC):
    """Abstract base class for training models."""

    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        model,
        loss,
        optimizer,
        scheduler=None,
        epochs: int = 100,
        start_epoch: int = 0,
        grad_clip: Optional[float] = 1e-1,
        weights_dir: str = "./weights",
        log_dir: str = "./logs",
        ema_rate: Optional[float] = None,
        display_progress: bool = False,
    ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.grad_clip = grad_clip
        self.weights_dir = weights_dir
        self.log_dir = log_dir
        self.ema_rate = ema_rate
        self.display_progress = display_progress

        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Logging
        self.logger = get_logger(log_dir=log_dir)
        self.tb_writer = SummaryWriter(log_dir=log_dir)

        # EMA tracking
        self.ema_params = (
            [p.clone().detach() for p in self.model.parameters()]
            if self.ema_rate is not None
            else None
        )

        # Gradient clipping function
        self.grad_clipper: Optional[Callable] = (
            (lambda params: clip_gradients(params, self.grad_clip))
            if self.grad_clip is not None
            else None
        )

    # ------------------------------------------------------------------ #
    # Core Methods
    # ------------------------------------------------------------------ #

    def update_ema(self) -> None:
        """Update EMA parameters from model weights."""
        if self.ema_rate is not None and self.ema_params is not None:
            update_ema(
                target_params=self.ema_params,
                source_params=self.model.parameters(),
                rate=self.ema_rate,
            )

    def save_model_weights(self, epoch: int) -> None:
        """Save model or EMA weights."""
        if self.ema_rate is not None and self.ema_params is not None:
            ema_state = {}
            for (name, _), p in zip(self.model.state_dict().items(), self.ema_params):
                ema_state[name] = p.clone()
            weights = ema_state
            model_path = os.path.join(self.weights_dir, f"ema_model_epoch_{epoch}.pth")
        else:
            weights = self.model.state_dict()
            model_path = os.path.join(self.weights_dir, f"model_epoch_{epoch}.pth")

        torch.save(weights, model_path)
        print(f"Model weights saved to {model_path}")

    # ------------------------------------------------------------------ #
    # Training Loop
    # ------------------------------------------------------------------ #

    def train(self) -> None:
        """Run standard train/val loop with checkpointing and TB logging."""
        best_val_loss = 1e6
        best_val_loss_epoch = -1

        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            cum_loss = 0.0
            print(f"[Epoch {epoch + 1}] Training started at {time.ctime()}...")

            for packed in tqdm(self.train_dataloader, desc=f"Train [{epoch+1}/{self.epochs}]", ncols=100, disable=not self.display_progress):
                x_pred, x_tgt = self.one_step(packed)
                loss = self.loss(x_pred, x_tgt)
                cum_loss += loss.item()

                loss.backward()
                if self.grad_clipper:
                    self.grad_clipper(self.model.parameters())
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.ema_rate:
                    self.update_ema()

            avg_train_loss = cum_loss / max(1, len(self.train_dataloader))

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for packed in tqdm(self.val_dataloader, desc=f"Validation [{epoch+1}/{self.epochs}]", ncols=100, disable=not self.display_progress):
                    x_pred, x_tgt = self.one_step(packed)
                    loss_val = self.loss(x_pred, x_tgt)
                    val_loss += loss_val.item()
            avg_val_loss = val_loss / max(1, len(self.val_dataloader))

            # Scheduler
            if self.scheduler:
                self.scheduler.step(avg_val_loss)

            # Save weights if best
            if avg_val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.5f} (epoch {best_val_loss_epoch+1}) to {avg_val_loss:.5f}.")
                best_val_loss = avg_val_loss
                best_val_loss_epoch = epoch + 1
                self.save_model_weights(epoch + 1)

            # Logging
            self.logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}"
            )
            self.tb_writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
            self.tb_writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)
            print()

    # ------------------------------------------------------------------ #
    # Abstract
    # ------------------------------------------------------------------ #

    @abstractmethod
    def one_step(self, packed):
        """Perform one optimization step; must return (x_pred, x_tgt)."""
        raise NotImplementedError
