"""
Diffusion Sampler (Base Class)
=======================================

Defines the generic forward and reverse diffusion steps shared across
derived samplers (e.g. wavelet or joint scatter samplers).
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.utils import extract


class DiffusionSampler(ABC):
    """Abstract base class for diffusion samplers."""

    def __init__(
        self,
        model: nn.Module,
        beta1: float = 1e-4,
        betaT: float = 0.02,
        T: int = 1000,
        device: str = "cuda",
    ):
        super().__init__()

        self.model = model
        self.T = T
        self.device = device

        self.betas = torch.linspace(beta1, betaT, T).double().to(device)
        self.alphas = (1. - self.betas).to(device)
        self.alphas_bar = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_bar_prev = F.pad(self.alphas_bar, [1, 0], value=1)[:T].to(device)

        self.sqrt_recip_alphas_bar = torch.sqrt(1. /self.alphas_bar).to(device)
        self.sqrt_recipm1_alphas_bar = torch.sqrt(1. / self.alphas_bar - 1).to(device)

        self.posterior_var = (self.betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)).to(device)
        self.posterior_log_var_clipped = torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])).to(device)
        self.posterior_mean_coef1 = (torch.sqrt(self.alphas_bar_prev) * self.betas / (1. - self.alphas_bar)).to(device)
        self.posterior_mean_coef2 = (torch.sqrt(self.alphas) * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)).to(device)

    def q_mean_variance(self, x0, xt, t):
        """Return posterior mean and log-variance of q(x_{t-1} | x_t, x0)."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, xt.shape) * x0 +
            extract(self.posterior_mean_coef2, t, xt.shape) * xt
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, xt.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_x0_from_epsilon_theta(self, xt, t, eps_theta):
        """Estimate x0 from model noise prediction."""
        return (
            extract(self.sqrt_recip_alphas_bar, t, xt.shape) * xt -
            extract(self.sqrt_recipm1_alphas_bar, t, xt.shape) * eps_theta
        )

    def p_mean_variance(self, xt, t):
        """Return model posterior parameters p_theta(x_{t-1} | x_t)."""
        model_log_var = torch.log(torch.cat([self.posterior_var[1:2], self.betas[1:]]))
        model_log_var = extract(model_log_var, t, xt.shape)

        # mean param
        epsilon_theta = self.model(xt, t)

        # x0 approx x0_hat(xt, t)
        x0 = self.predict_x0_from_epsilon_theta(xt, t, eps_theta=epsilon_theta)
        model_mean, _ = self.q_mean_variance(x0, xt, t)
        return model_mean, model_log_var

    @abstractmethod
    def inference(self, xT: torch.Tensor):
        pass
