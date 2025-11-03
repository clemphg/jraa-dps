"""
JRAADPS — Joint Reconstruction and Adaptive Attenuation Diffusion Sampler
=========================================================================

Wavelet-domain diffusion sampler supporting optional *joint scatter estimation*.

This class extends `GaussianDiffusionSampler` to:
- Perform denoising diffusion in the wavelet domain.
- Optionally optimize a differentiable scatter model jointly.
- Apply a smoothness regularization term on the scatter field, scaled by a
  configurable `scatter_smoothness_weight`.

Example
-------
>>> sampler = JRAADPS(model, enable_joint_scatter=True, scatter_smoothness_weight=1e-3)
>>> sampler.inference(zT, grad_fun, scatter_model=my_scatter_model)
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.operators.wavelet import Wavelet
from src.samplers.diffusion_sampler import DiffusionSampler
from src.utils.utils import extract


class JRAADPS(DiffusionSampler):
    """Joint Reconstruction and Adaptive Attenuation Diffusion Sampler."""

    def __init__(
        self,
        model,
        beta1: float = 1e-4,
        betaT: float = 0.02,
        T: int = 1000,
        device: str = "cuda",
        enable_joint_scatter: bool = False,
        scatter_smoothness_weight: float = 1e-3,
        scatter_smoothness_order: float = 1e-3,
    ):
        """
        Initialize the JRAADPS sampler.

        Args:
            model (nn.Module): Diffusion model for wavelet-domain denoising.
            beta1 (float): Start of linear beta schedule.
            betaT (float): End of linear beta schedule.
            T (int): Number of diffusion timesteps.
            device (str): Compute device.
            enable_joint_scatter (bool): Whether to include scatter estimation.
            scatter_smoothness_weight (float): Scaling factor for smoothness regularization.
            scatter_smoothness_order (float): Number of neighbours over which to apply smoothness regularization.
        """
        super().__init__(model, beta1, betaT, T, device)
        self.wavelet_op = Wavelet()

        # scatter
        self.enable_joint_scatter = enable_joint_scatter
        self.scatter_smoothness_weight = scatter_smoothness_weight
        self.scatter_smoothness_order = scatter_smoothness_order


    def p_mean_variance(self, 
                        xt: torch.Tensor, 
                        t: int):
        """
        mean + log-variance of p_{theta}(x_{t-1} | xt)
        """
        model_log_var = torch.log(torch.cat([self.posterior_var[1:2], self.betas[1:]]))
        model_log_var = extract(model_log_var, t, xt.shape)

        z0 = self.model(xt, t)
        model_mean, _ = self.q_mean_variance(z0, xt, t)

        return model_mean, model_log_var, z0

    # --------------------------------------------------------------------- #
    # Utility: loss computation
    # --------------------------------------------------------------------- #

    def _compute_total_loss(self, grad_fun, x0_hat_lambda, x0_hat_mu, subset_idx, scatter_model):
        """Compute total loss including smoothness (if enabled)."""
        scatter = scatter_model() if self.enable_joint_scatter else None
        loss = grad_fun(x0_hat_lambda, x0_hat_mu, subset_idx, scatter)

        if self.enable_joint_scatter and self.scatter_smoothness_weight > 0.0:
            smooth_loss = scatter_model.smoothness_loss(scatter, order=self.scatter_smoothness_order)
            loss -= self.scatter_smoothness_weight * smooth_loss

        del scatter

        return loss

    # --------------------------------------------------------------------- #
    # Core Inference Loop
    # --------------------------------------------------------------------- #

    def inference(
        self,
        zT,
        grad_fun,
        zeta=1.0,
        xi=1.0,
        norm_grad=True,
        num_subsets=10,
        lr_scatter=1e-3,
        scatter_model=None,
        display_progress=False,
    ):
        """
        Perform denoising diffusion in wavelet domain, optionally with scatter estimation.

        Args:
            zT (torch.Tensor): Initial latent (noise) tensor.
            grad_fun (callable): Function returning (loss, divergence).
            zeta (float | list): Step size for lambda correction.
            xi (float | list): Step size for mu correction.
            norm_grad (bool): Whether to normalize gradients.
            num_subsets (int): Number of projection subsets (for OSEM-like updates).
            lr_scatter (float): Learning rate for scatter parameters.
            scatter_model (nn.Module, optional): Scatter model (required if joint estimation enabled).
            display_progress (bool): Whether to display progress.

        Returns:
            torch.Tensor: Final reconstructed estimate x₀.
        """
        zt = zT

        if self.enable_joint_scatter and scatter_model is not None:
            scatter_optimizer = torch.optim.Adam(
                [
                    {"params": scatter_model.raw_a, "lr": lr_scatter},
                    {"params": scatter_model.raw_b, "lr": lr_scatter},
                    {"params": scatter_model.raw_c, "lr": lr_scatter * 10},
                ]
            )

        # Measurements contribution schedules
        zetas = [zeta] * self.T if isinstance(zeta, (int, float)) else zeta
        xis = [xi] * self.T if isinstance(xi, (int, float)) else xi

        pbar = tqdm(reversed(range(self.T)), total=self.T, disable=not display_progress, ncols=100)
        for time_step in pbar:

            osem_subset = time_step % num_subsets # Angles subset for OSEM-like subset updates

            # DDPM update
            with torch.no_grad():
                t = zt.new_ones([zt.shape[0], ], dtype=torch.long) * time_step
                epsilon = torch.randn_like(zt) if time_step > 0 else 0
                mean, log_var, z0_hat = self.p_mean_variance(xt=zt, t=t)
                zt_prime = mean + torch.exp(0.5 * log_var) * epsilon

            # Compute loss (NPLL + scatter if relevant)
            with torch.enable_grad():
                zt.requires_grad_()

                if self.enable_joint_scatter and scatter_model is not None:
                    scatter_optimizer.zero_grad()

                z0_hat = self.model(zt, t)
                x0_hat = self.wavelet_op.transposed_transform(z0_hat, 3)
                x0_hat_lambda, x0_hat_mu = x0_hat[:, 0].unsqueeze(1), x0_hat[:, 1].unsqueeze(1)

                loss = self._compute_total_loss(grad_fun, x0_hat_lambda, x0_hat_mu, osem_subset, scatter_model)
                if display_progress:
                    pbar.set_postfix({'Loss': loss.item()})

                loss.backward()

                if self.enable_joint_scatter and scatter_model is not None:
                    scatter_optimizer.step()

            with torch.no_grad():

                grad = zt.grad
                grad_lambda = grad[:, 0::2]
                grad_mu = grad[:, 1::2]
                if norm_grad:
                    grad_lambda /= grad_lambda.norm() + 1e-7
                    grad_mu /= grad_mu.norm() + 1e-7

                # DPS update
                zt_prime[:, 0::2] -= zetas[time_step] * grad_lambda
                zt_prime[:, 1::2] -= xis[time_step] * grad_mu
                zt = zt_prime.clone().detach()

            del epsilon, mean, log_var, zt_prime
            del z0_hat, x0_hat, x0_hat_lambda, x0_hat_mu
            del loss, grad, grad_lambda, grad_mu
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return self.model(zt, t).detach()
