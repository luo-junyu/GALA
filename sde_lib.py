"""Abstract SDE classes, Reverse SDE, and VP SDEs."""

import abc
import torch
import numpy as np


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$"""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z, mask):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
            z: latent code
        Returns:
            log probability density
        """
        pass

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probability flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
            score_fn: A time-dependent score-based model that takes x and t and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """

        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, *args, **kwargs):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""

                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t, *args, **kwargs)
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            '''
            def sde_score(self, x, t, score):
                """Create the drift and diffusion functions for the reverse SDE/ODE, given score values."""
                drift, diffusion = sde_fn(x, t)
                if len(score.shape) == 4:
                    drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                elif len(score.shape) == 3:
                    drift = drift - diffusion[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                else:
                    raise ValueError
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion
            '''

            def discretize(self, x, t, *args, **kwargs):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t, *args, **kwargs) * \
                        (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

            '''
            def discretize_score(self, x, t, score):
                """Create discretized iteration rules for the reverse diffusion sampler, given score values."""
                f, G = discretize_fn(x, t)
                if len(score.shape) == 4:
                    rev_f = f - G[:, None, None, None] ** 2 * score * \
                        (0.5 if self.probability_flow else 1.)
                elif len(score.shape) == 3:
                    rev_f = f - G[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                else:
                    raise ValueError
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G
            '''

        return RSDE()


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
            beta_min: value of beta(0)
            beta_max: value of beta(1)
            N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        if len(x.shape) == 4:
            drift = -0.5 * beta_t[:, None, None, None] * x
        elif len(x.shape) == 3:
            drift = -0.5 * beta_t[:, None, None] * x
        else:
            raise NotImplementedError
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        if len(x.shape) == 4:
            mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        elif len(x.shape) == 3:
            mean = torch.exp(log_mean_coeff[:, None, None]) * x
        else:
            raise ValueError("The shape of x in marginal_prob is not correct.")
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    # def log_snr(self, t):
    #     log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    #     mean = torch.exp(log_mean_coeff)
    #     std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    #     log_snr = torch.log(mean / std)
    #     return log_snr, mean, std
    #
    # def log_snr_np(self, t):
    #     log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    #     mean = np.exp(log_mean_coeff)
    #     std = np.sqrt(1. - np.exp(2. * log_mean_coeff))
    #     log_snr = np.log(mean / std)
    #     return log_snr
    #
    # def lambda2t(self, lambda_ori):
    #     log_val = torch.log(torch.exp(-2. * lambda_ori) + 1.)
    #     t = 2. * log_val / (torch.sqrt(self.beta_0 ** 2 + 2. * (self.beta_1 - self.beta_0) * log_val) + self.beta_0)
    #     return t
    #
    # def lambda2t_np(self, lambda_ori):
    #     log_val = np.log(np.exp(-2. * lambda_ori) + 1.)
    #     t = 2. * log_val / (np.sqrt(self.beta_0 ** 2 + 2. * (self.beta_1 - self.beta_0) * log_val) + self.beta_0)
    #     return t

    def prior_sampling(self, shape):
        sample = torch.randn(*shape)
        if len(shape) == 4:
            sample = torch.tril(sample, -1)
            sample = sample + sample.transpose(-1, -2)

        return sample

    def prior_logp(self, z, mask):
        N = torch.sum(mask, dim=tuple(range(1, len(mask.shape))))
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum((z * mask) ** 2, dim=(1, 2, 3)) / 2.
        return logps

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        if len(x.shape) == 4:
            f = torch.sqrt(alpha)[:, None, None, None] * x - x
        elif len(x.shape) == 3:
            f = torch.sqrt(alpha)[:, None, None] * x - x
        else:
            NotImplementedError
        G = sqrt_beta
        return f, G

