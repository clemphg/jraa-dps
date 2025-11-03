"""
normalization.py
================

Normalization methods for PET image preprocessing.

Implements several normalization strategies through a common abstract interface:
    - NormRange:    Linear rescaling to a fixed range (e.g., [-1, 1])
    - NormSqrt:     Power-based normalization to compress dynamic range
    - NormZScore:   Standard z-score normalization

Each normalization class provides:
    - norm(img):   Normalize an image tensor or array.
    - denorm(img): Invert normalization back to the original scale.
"""

from abc import ABC, abstractmethod
from typing import Union

import torch


# -----------------------------------------------------------------------------
# Abstract Base
# -----------------------------------------------------------------------------

class NormMethod(ABC):
    """Abstract base class for normalization methods."""

    @abstractmethod
    def norm(self, img: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
        """Apply normalization to an image or tensor."""
        pass

    @abstractmethod
    def denorm(self, img: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
        """Revert normalization to the original value range."""
        pass


# -----------------------------------------------------------------------------
# Range Normalization
# -----------------------------------------------------------------------------

class NormRange(NormMethod):
    """Linear rescaling of input data to a specified range.

    Example:
        >>> normer = NormRange(img_min=0.0, img_max=1.0)
        >>> img_norm = normer.norm(img)
        >>> img_recon = normer.denorm(img_norm)
    """

    def __init__(
        self,
        img_min: float = 0.0,
        img_max: float = 0.025,
        new_min: float = -1.0,
        new_max: float = 1.0,
    ):
        """Initialize the normalization mapping.

        Default parameters correspond to μ-map values (in mm⁻¹) at 511 keV.

        Args:
            img_min (float): Minimum value of unnormalized images.
            img_max (float): Maximum value of unnormalized images.
            new_min (float): Minimum value of normalized range.
            new_max (float): Maximum value of normalized range.
        """
        self.img_min = img_min
        self.img_max = img_max
        self.new_min = new_min
        self.new_max = new_max

    def norm(self, img: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
        """Normalize to target range."""
        scale = (self.new_max - self.new_min) / (self.img_max - self.img_min)
        return (img - self.img_min) * scale + self.new_min

    def denorm(self, img: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
        """Invert normalization back to the original range."""
        scale = (self.img_max - self.img_min) / (self.new_max - self.new_min)
        return (img - self.new_min) * scale + self.img_min


# -----------------------------------------------------------------------------
# Power (Sqrt) Normalization
# -----------------------------------------------------------------------------

class NormSqrt(NormMethod):
    """Normalization with power compression (square-root–like scaling).

    Applies a fractional power to compress large values before rescaling
    to a desired target range. Often used for PET or CT data with heavy-tailed
    distributions.

    Example:
        >>> normer = NormSqrt(sqrt_order=2, img_min=0.0, img_max=1e5)
        >>> img_norm = normer.norm(img)
        >>> img_recon = normer.denorm(img_norm)
    """

    def __init__(
        self,
        sqrt_order: int = 2,
        img_min: float = 0.0,
        img_max: float = 1e5,
        new_min: float = -1.0,
        new_max: float = 9.0,
    ):
        """Initialize parameters.

        Args:
            sqrt_order (int): Power for the transformation (2 = square-root).
            img_min (float): Minimum value in the input range.
            img_max (float): Maximum value in the input range.
            new_min (float): Minimum normalized output.
            new_max (float): Maximum normalized output.
        """
        self.sqrt_order = sqrt_order
        self.img_min = img_min
        self.img_max = img_max
        self.new_min = new_min
        self.new_max = new_max

    def norm(self, img: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
        """Apply square-root–like normalization."""
        base = (img - self.img_min) / (self.img_max - self.img_min)
        scaled = base.clamp_min(0) ** (1 / self.sqrt_order)
        return scaled * (self.new_max - self.new_min) + self.new_min

    def denorm(self, img: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
        """Invert square-root normalization."""
        base = (img - self.new_min) / (self.new_max - self.new_min)
        scaled = base.clamp_min(0) ** self.sqrt_order
        return scaled * (self.img_max - self.img_min) + self.img_min


# -----------------------------------------------------------------------------
# Z-Score Normalization
# -----------------------------------------------------------------------------

class NormZScore(NormMethod):
    """Standard z-score normalization.

    Example:
        >>> normer = NormZScore(img_mean=0.01, img_std=0.005)
        >>> img_norm = normer.norm(img)
        >>> img_recon = normer.denorm(img_norm)
    """

    def __init__(self, img_mean: float, img_std: float):
        """Initialize z-score parameters.

        Args:
            img_mean (float): Dataset mean.
            img_std (float): Dataset standard deviation.
        """
        self.img_mean = img_mean
        self.img_std = img_std

    def norm(self, img: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
        """Normalize via (x - μ) / σ."""
        return (img - self.img_mean) / self.img_std

    def denorm(self, img: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
        """Invert z-score normalization."""
        return img * self.img_std + self.img_mean
