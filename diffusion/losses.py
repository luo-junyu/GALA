"""All functions related to loss computation and optimization."""

import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
from sde_lib import VPSDE


def get_optimizer(config, params):
    """Return a flax optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!'
        )
    return optimizer


def optimization_manager(config):
    """Return an optimize_fn based on `config`."""

    def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimize with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    return optimize_fn


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
    """Create a loss function for training with arbitrary SDEs.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        train: `True` for training loss and `False` for evaluation loss.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise, sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps.
                    Otherwise, it requires ad-hoc interpolation to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses according
            to https://arxiv.org/abs/2101.09258; otherwise, use the weighting recommended in Score SDE paper.
        eps: A `float` number. The smallest time step to sample from.

    Returns:
        A loss function.
    """

    # reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        """Compute the loss function.

        Args:
            model: A score model.
            batch: A mini-batch of training data, including adjacency matrices and mask.

        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """
        adj, mask = batch
        score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
        t = torch.rand(adj.shape[0], device=adj.device) * (sde.T - eps) + eps

        z = torch.randn_like(adj)  # [B, C, N, N]
        z = torch.tril(z, -1)
        z = z + z.transpose(2, 3)

        mean, std = sde.marginal_prob(adj, t)
        mean = torch.tril(mean, -1)
        mean = mean + mean.transpose(2, 3)

        perturbed_data = mean + std[:, None, None, None] * z
        score = score_fn(perturbed_data, t, mask=mask)

        mask = torch.tril(mask, -1)
        mask = mask + mask.transpose(2, 3)
        mask = mask.reshape(mask.shape[0], -1)  # low triangular part of adj matrices

        if not likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = losses.reshape(losses.shape[0], -1)
            if reduce_mean:
                losses = torch.sum(losses * mask, dim=-1) / torch.sum(mask, dim=-1)
            else:
                losses = 0.5 * torch.sum(losses * mask, dim=-1)
            loss = losses.mean()
        else:
            g2 = sde.sde(torch.zeros_like(adj), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = losses.reshape(losses.shape[0], -1)
            if reduce_mean:
                losses = torch.sum(losses * mask, dim=-1) / torch.sum(mask, dim=-1)
            else:
                losses = 0.5 * torch.sum(losses * mask, dim=-1)
            loss = (losses * g2).mean()

        return loss

    return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
    """Create a one-step training/evaluation function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
             Tuple (`sde_lib.SDE`, `sde_lib.SDE`) that represents the forward node SDE and edge SDE.
        optimize_fn: An optimization function.
        reduce_mean: If `True`, average the loss across data dimensions.
            Otherwise, sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses according to
            https://arxiv.org/abs/2101.09258; otherwise, use the weighting recommended by score-sde.

    Returns:
        A one-step function for training or evaluation.
    """

    if continuous:
        if isinstance(sde, tuple):
            loss_fn = get_multi_sde_loss_fn(sde[0], sde[1], train, reduce_mean=reduce_mean, continuous=True,
                                            likelihood_weighting=likelihood_weighting)
        else:
            loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                                      continuous=True, likelihood_weighting=likelihood_weighting)
    else:
        assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
        elif isinstance(sde, tuple):
            raise ValueError("Discrete training for multi sde is not recommended.")
        else:
            raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

    def step_fn(state, batch):
        """Running one step of training or evaluation.

        For jax version: This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and
            jit-compiled together for faster execution.

        Args:
            state: A dictionary of training information, containing the score model, optimizer,
                EMA status, and number of optimization steps.
            batch: A mini-batch of training/evaluation data, including min-batch adjacency matrices and mask.

        Returns:
            loss: The average loss value of this state.
        """
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch)
                ema.restore(model.parameters())

        return loss

    return step_fn
