"""
Gaussian Scatter Model
======================

Defines a differentiable scatter correction model for PET sinograms.
The model implements a parameterized scatter distribution, with support for 
different projection geometries and Time-of-Flight (TOF) bins.

Scatter is modeled with a Gaussian for each projection with learnable scaling, center, and standard deviation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianScatter(nn.Module):
    """Gaussian-shaped scatter model with learnable amplitude, mean, and std.

    The scatter profile is parametrized in each projection as a Gaussian
    function along the radial axis, with learnable parameters:
    amplitude (a), mean (b), and width (c). The model supports optional TOF bins
    and includes smoothness regularization to enforce spatial continuity.
    """

    def __init__(
        self,
        num_rad: int,
        num_views: int,
        num_sinos: int,
        num_tofbins: int = 0,
        init_a: float = 1.0,
        init_b: float = 0.0,
        init_c: float = 1.0,
        float_init: bool = True,
        beta_softplus: float = 1.0,
        device: str = "cuda",
    ):
        """Initializes the Gaussian scatter model.

        Args:
            num_rad (int): Number of radial bins.
            num_views (int): Number of projection angles (views).
            num_sinos (int): Number of sinograms (axial slices).
            num_tofbins (int, optional): Number of Time-of-Flight bins. Defaults to 0.
            init_a (float, optional): Initial amplitude. Defaults to 1.0.
            init_b (float, optional): Initial mean (offset). Defaults to 0.0.
            init_c (float, optional): Initial standard deviation. Defaults to 1.0.
            float_init (bool, optional): If True, initializes parameters as nn.Parameters.
            beta_softplus (float, optional): Softplus slope parameter for stability.
            device (str, optional): Compute device ('cuda' or 'cpu'). Defaults to 'cuda'.
        """
        super().__init__()
        self.num_rad = num_rad
        self.num_views = num_views
        self.num_sinos = num_sinos
        self.num_tofbins = num_tofbins
        self.beta_softplus = beta_softplus
        self.device = device

        if float_init:
            self.raw_a = nn.Parameter(torch.full((num_views, num_sinos), init_a, device=device))
            self.raw_b = nn.Parameter(torch.full((num_views, num_sinos), init_b, device=device))
            self.raw_c = nn.Parameter(torch.full((num_views, num_sinos), init_c, device=device))
        else:
            self.raw_a, self.raw_b, self.raw_c = init_a, init_b, init_c

    def forward(self) -> torch.Tensor:
        """Generates a Gaussian scatter profile.

        Returns:
            torch.Tensor: Scatter tensor of shape [num_rad, num_views, num_sinos]
                or [num_rad, num_views, num_sinos, num_tofbins] if TOF is enabled.
        """
        a = F.softplus(self.raw_a, beta=self.beta_softplus)
        b = self.raw_b
        c = F.softplus(self.raw_c, beta=self.beta_softplus) + 1e-6

        # Create centered radial coordinate [num_rad, 1, 1]
        r = torch.arange(self.num_rad, device=a.device).float()
        r = (r - (self.num_rad - 1) / 2.0).view(self.num_rad, 1, 1)

        scatter = a.unsqueeze(0) * torch.exp(-0.5 * ((r - b.unsqueeze(0)) ** 2) / (c.unsqueeze(0) ** 2))
        if self.num_tofbins > 0:
            scatter = scatter.unsqueeze(-1).expand(-1, -1, -1, self.num_tofbins)
        return scatter

    def get_gradient(self):
        """Returns gradients of parameters a, b, and c.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Gradients of (a, b, c).
        """
        return (
            self.raw_a.grad.detach(),
            self.raw_b.grad.detach(),
            self.raw_c.grad.detach(),
        )

    def smoothness_loss(
        self,
        scatter: torch.Tensor | None = None,
        order: int = 1,
        neighbor_weights: list[float] | None = None,
    ) -> torch.Tensor:
        """Computes smoothness loss along projection (view) dimension.

        Encourages adjacent projections to produce similar scatter profiles.

        Args:
            scatter (torch.Tensor, optional): Precomputed scatter tensor.
                If None, computed internally. Shape [num_rad, num_views, num_sinos].
            order (int, optional): Order of neighbor comparison (e.g., 1 for nearest neighbor). Defaults to 1.
            neighbor_weights (list[float], optional): Custom weights for neighbor levels.

        Returns:
            torch.Tensor: Scalar smoothness loss value.
        """
        # Default neighbor weights if not provided
        if neighbor_weights is None or len(neighbor_weights) != order:
            neighbor_weights = [a / sum(range(order + 1)) for a in reversed(range(1, order + 1))]
        elif order == 0:
            neighbor_weights = [0]

        # Generate scatter internally if not provided
        if scatter is None:
            r = torch.arange(self.num_rad, device=self.device).float().view(self.num_rad, 1, 1)
            a = F.softplus(self.raw_a, beta=self.beta_softplus)
            b = self.raw_b
            c = F.softplus(self.raw_c, beta=self.beta_softplus) + 1e-6
            scatter = a.unsqueeze(0) * torch.exp(-0.5 * ((r - b.unsqueeze(0)) ** 2) / (c.unsqueeze(0) ** 2))

        # Compute weighted L2 differences across views
        smooth_loss = 0.0
        for k, w in enumerate(neighbor_weights, start=1):
            diffs = scatter - torch.roll(scatter, shifts=-k, dims=1)
            smooth_loss += w * (diffs ** 2).mean()
            del diffs

        return smooth_loss
