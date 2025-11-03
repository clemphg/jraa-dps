"""
Operator Base Class
===================

Defines an abstract base class for linear or nonlinear transformation
operators used in image reconstruction, signal processing, or neural
network frameworks.

Subclasses must implement both:
    - `transform()`: The forward transformation.
    - `transposed_transform()`: The adjoint (transpose) or inverse transformation.
"""

from abc import ABC, abstractmethod
import torch


class Operator(ABC):
    """Abstract base class for transformation operators."""

    @abstractmethod
    def transform(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Applies the forward transformation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor to transform.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        raise NotImplementedError("Subclasses must implement `transform` method.")

    @abstractmethod
    def transposed_transform(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Applies the transposed or adjoint transformation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor to apply the transposed transform.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Transposed (or adjoint) transformed tensor.
        """
        raise NotImplementedError("Subclasses must implement `transposed_transform` method.")
