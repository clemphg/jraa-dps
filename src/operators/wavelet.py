"""
3D Wavelet Transform Operators
==============================

Differentiable 3D discrete wavelet transform (DWT) and inverse discrete wavelet 
transform (IDWT) for volumetric data (e.g., 3D PET). Also defines a high-level 
`Wavelet` operator wrapper that integrates DWT/IDWT within the project's operator 
framework.

Based on: https://github.com/pfriedri/wdm-3d
"""

import math
import numpy as np
import pywt
import torch
from torch.autograd import Function
from src.operators.operator import Operator


# ---------------------------------------------------------------------------- #
# 3D Discrete Wavelet Transform
# ---------------------------------------------------------------------------- #

class DWTFunction3D(Function):
    """Differentiable 3D discrete wavelet transform (DWT)."""

    @staticmethod
    def forward(ctx, x, low_0, low_1, low_2, high_0, high_1, high_2):
        """Forward 3D DWT.

        Args:
            x (torch.Tensor): Input tensor [N, C, D, H, W].
            low_0, low_1, low_2 (torch.Tensor): Low-pass filter matrices.
            high_0, high_1, high_2 (torch.Tensor): High-pass filter matrices.

        Returns:
            tuple(torch.Tensor): Eight subbands (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH).
        """
        ctx.save_for_backward(low_0, low_1, low_2, high_0, high_1, high_2)

        L = torch.matmul(low_0, x)
        H = torch.matmul(high_0, x)

        LL = torch.matmul(L, low_1).transpose(2, 3)
        LH = torch.matmul(L, high_1).transpose(2, 3)
        HL = torch.matmul(H, low_1).transpose(2, 3)
        HH = torch.matmul(H, high_1).transpose(2, 3)

        LLL = torch.matmul(low_2, LL).transpose(2, 3)
        LLH = torch.matmul(low_2, LH).transpose(2, 3)
        LHL = torch.matmul(low_2, HL).transpose(2, 3)
        LHH = torch.matmul(low_2, HH).transpose(2, 3)
        HLL = torch.matmul(high_2, LL).transpose(2, 3)
        HLH = torch.matmul(high_2, LH).transpose(2, 3)
        HHL = torch.matmul(high_2, HL).transpose(2, 3)
        HHH = torch.matmul(high_2, HH).transpose(2, 3)

        return LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH

    @staticmethod
    def backward(ctx, *grads):
        """Backward pass for 3D DWT (computes adjoint transform)."""
        low_0, low_1, low_2, high_0, high_1, high_2 = ctx.saved_tensors
        (
            grad_LLL,
            grad_LLH,
            grad_LHL,
            grad_LHH,
            grad_HLL,
            grad_HLH,
            grad_HHL,
            grad_HHH,
        ) = grads

        def combine(low_matrix, high_matrix, grad_low, grad_high):
            return torch.add(
                torch.matmul(low_matrix.t(), grad_low.transpose(2, 3)),
                torch.matmul(high_matrix.t(), grad_high.transpose(2, 3)),
            ).transpose(2, 3)

        grad_LL = combine(low_2, high_2, grad_LLL, grad_HLL)
        grad_LH = combine(low_2, high_2, grad_LLH, grad_HLH)
        grad_HL = combine(low_2, high_2, grad_LHL, grad_HHL)
        grad_HH = combine(low_2, high_2, grad_LHH, grad_HHH)

        grad_L = torch.add(
            torch.matmul(grad_LL, low_1.t()), torch.matmul(grad_LH, high_1.t())
        )
        grad_H = torch.add(
            torch.matmul(grad_HL, low_1.t()), torch.matmul(grad_HH, high_1.t())
        )

        grad_input = torch.add(
            torch.matmul(low_0.t(), grad_L), torch.matmul(high_0.t(), grad_H)
        )
        return grad_input, None, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------- #
# 3D Inverse Discrete Wavelet Transform
# ---------------------------------------------------------------------------- #

class IDWTFunction3D(Function):
    """Differentiable 3D inverse discrete wavelet transform (IDWT)."""

    @staticmethod
    def forward(
        ctx,
        LLL,
        LLH,
        LHL,
        LHH,
        HLL,
        HLH,
        HHL,
        HHH,
        low_0,
        low_1,
        low_2,
        high_0,
        high_1,
        high_2,
    ):
        """Forward 3D IDWT (reconstruction).

        Returns:
            torch.Tensor: Reconstructed 3D volume [N, C, D, H, W].
        """
        ctx.save_for_backward(low_0, low_1, low_2, high_0, high_1, high_2)

        def combine(low_matrix, high_matrix, low_part, high_part):
            return torch.add(
                torch.matmul(low_matrix.t(), low_part.transpose(2, 3)),
                torch.matmul(high_matrix.t(), high_part.transpose(2, 3)),
            ).transpose(2, 3)

        LL = combine(low_2, high_2, LLL, HLL)
        LH = combine(low_2, high_2, LLH, HLH)
        HL = combine(low_2, high_2, LHL, HHL)
        HH = combine(low_2, high_2, LHH, HHH)

        L = torch.add(torch.matmul(LL, low_1.t()), torch.matmul(LH, high_1.t()))
        H = torch.add(torch.matmul(HL, low_1.t()), torch.matmul(HH, high_1.t()))

        output = torch.add(torch.matmul(low_0.t(), L), torch.matmul(high_0.t(), H))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass (computes DWT of gradient)."""
        low_0, low_1, low_2, high_0, high_1, high_2 = ctx.saved_tensors

        grad_L = torch.matmul(low_0, grad_output)
        grad_H = torch.matmul(high_0, grad_output)

        grad_LL = torch.matmul(grad_L, low_1).transpose(2, 3)
        grad_LH = torch.matmul(grad_L, high_1).transpose(2, 3)
        grad_HL = torch.matmul(grad_H, low_1).transpose(2, 3)
        grad_HH = torch.matmul(grad_H, high_1).transpose(2, 3)

        def part(matrix, grad):
            return torch.matmul(matrix, grad).transpose(2, 3)

        grad_LLL, grad_LLH, grad_LHL, grad_LHH = (
            part(low_2, grad_LL),
            part(low_2, grad_LH),
            part(low_2, grad_HL),
            part(low_2, grad_HH),
        )
        grad_HLL, grad_HLH, grad_HHL, grad_HHH = (
            part(high_2, grad_LL),
            part(high_2, grad_LH),
            part(high_2, grad_HL),
            part(high_2, grad_HH),
        )

        return (
            grad_LLL,
            grad_LLH,
            grad_LHL,
            grad_LHH,
            grad_HLL,
            grad_HLH,
            grad_HHL,
            grad_HHH,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------- #
# 3D DWT and IDWT Modules
# ---------------------------------------------------------------------------- #

class DWT3D(torch.nn.Module):
    """3D Discrete Wavelet Transform (DWT) module."""

    def __init__(self, wavename: str = "haar", device: str = "cuda"):
        """
        Args:
            wavename (str): Name of the wavelet (see `pywt.wavelist()`).
            device (str): Device to store transformation matrices on.
        """
        super().__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)

        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_half = self.band_length // 2
        self.device = device

    def _get_matrices(self, depth, height, width):
        """Builds low-pass and high-pass matrices for each axis."""
        def build_matrix(size, coeffs):
            L = size // 2
            matrix = np.zeros((L, size + self.band_length - 2))
            for i in range(L):
                matrix[i, i * 2 : i * 2 + self.band_length] = coeffs
            return matrix[:, self.band_half - 1 : -self.band_half + 1 or None]

        h0 = build_matrix(height, self.band_low)
        h1 = build_matrix(width, self.band_low).T
        h2 = build_matrix(depth, self.band_low)
        g0 = build_matrix(height, self.band_high)
        g1 = build_matrix(width, self.band_high).T
        g2 = build_matrix(depth, self.band_high)

        def to_tensor(arr):
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        self.low_0, self.low_1, self.low_2 = map(to_tensor, [h0, h1, h2])
        self.high_0, self.high_1, self.high_2 = map(to_tensor, [g0, g1, g2])

    def forward(self, x: torch.Tensor):
        """Applies 3D DWT.

        Args:
            x (torch.Tensor): Input tensor [N, C, D, H, W].

        Returns:
            tuple(torch.Tensor): Eight subbands (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH).
        """
        _, _, D, H, W = x.shape
        self._get_matrices(D, H, W)
        return DWTFunction3D.apply(
            x,
            self.low_0,
            self.low_1,
            self.low_2,
            self.high_0,
            self.high_1,
            self.high_2,
        )


class IDWT3D(torch.nn.Module):
    """3D Inverse Discrete Wavelet Transform (IDWT) module."""

    def __init__(self, wavename: str = "haar", device: str = "cuda"):
        super().__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = list(reversed(wavelet.dec_lo))
        self.band_high = list(reversed(wavelet.dec_hi))
        assert len(self.band_low) == len(self.band_high)

        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_half = self.band_length // 2
        self.device = device

    def _get_matrices(self, depth, height, width):
        """Builds inverse transform matrices for each axis."""
        def build_matrix(size, coeffs):
            L = size // 2
            matrix = np.zeros((L, size + self.band_length - 2))
            for i in range(L):
                matrix[i, i * 2 : i * 2 + self.band_length] = coeffs
            return matrix[:, self.band_half - 1 : -self.band_half + 1 or None]

        h0 = build_matrix(height, self.band_low)
        h1 = build_matrix(width, self.band_low).T
        h2 = build_matrix(depth, self.band_low)
        g0 = build_matrix(height, self.band_high)
        g1 = build_matrix(width, self.band_high).T
        g2 = build_matrix(depth, self.band_high)

        def to_tensor(arr):
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        self.low_0, self.low_1, self.low_2 = map(to_tensor, [h0, h1, h2])
        self.high_0, self.high_1, self.high_2 = map(to_tensor, [g0, g1, g2])

    def forward(self, *bands):
        """Reconstructs 3D volume from 8 subbands."""
        LLL = bands[0]
        _, _, D_half, H_half, W_half = LLL.shape
        D, H, W = D_half * 2, H_half * 2, W_half * 2
        self._get_matrices(D, H, W)
        return IDWTFunction3D.apply(
            *bands,
            self.low_0,
            self.low_1,
            self.low_2,
            self.high_0,
            self.high_1,
            self.high_2,
        )


# ---------------------------------------------------------------------------- #
# Unified Wavelet Operator Wrapper
# ---------------------------------------------------------------------------- #

class Wavelet(Operator):
    """Unified 3D Wavelet operator combining DWT and IDWT."""

    def __init__(
        self,
        dwt: DWT3D = None,
        idwt: IDWT3D = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.dwt = dwt or DWT3D(device=device)
        self.idwt = idwt or IDWT3D(device=device)

    def transform(self, x: torch.Tensor, div_low_pass: float = 1.0) -> torch.Tensor:
        """Applies 3D DWT and stacks all subbands along the channel dimension."""
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.dwt(x)
        stacked = torch.cat(
            [LLL / div_low_pass, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1
        )
        return stacked

    def transposed_transform(
        self, x: torch.Tensor, mult_low_pass: float = 1.0
    ) -> torch.Tensor:
        """Applies inverse 3D wavelet transform (IDWT)."""
        B, C, D, H, W = x.shape
        num_channels = C // 8
        components = []

        for c in range(num_channels):
            components.append(
                self.idwt(
                    x[:, c, :, :, :].view(B, 1, D, H, W) * mult_low_pass,
                    x[:, c + num_channels, :, :, :].view(B, 1, D, H, W),
                    x[:, c + 2 * num_channels, :, :, :].view(B, 1, D, H, W),
                    x[:, c + 3 * num_channels, :, :, :].view(B, 1, D, H, W),
                    x[:, c + 4 * num_channels, :, :, :].view(B, 1, D, H, W),
                    x[:, c + 5 * num_channels, :, :, :].view(B, 1, D, H, W),
                    x[:, c + 6 * num_channels, :, :, :].view(B, 1, D, H, W),
                    x[:, c + 7 * num_channels, :, :, :].view(B, 1, D, H, W),
                )
            )

        return torch.cat(components, dim=1)
