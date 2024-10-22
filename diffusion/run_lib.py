import os
import torch
import numpy as np
import random
import logging
import time
from absl import flags
from torch.utils import tensorboard
from torch_geometric.data import DataLoader, DenseDataLoader
import pickle

from models import pgsn
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
from evaluation import get_stats_eval, get_nn_eval
import sde_lib
import visualize
from utils import *
import tqdm


FLAGS = flags.FLAGS


def set_random_seed(config):
    seed = config.seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sde_train(config, workdir):
    """Runs the training pipeline.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints and TF summaries.
            If this contains checkpoint training will be resumed from the latest checkpoint.
    """

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directly
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(workdir, f'{config.data.name}-{config.data.train_index}' , "checkpoint.pth")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(os.path.dirname(checkpoint_meta_dir)):
        os.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build dataloader and iterators
    train_ds, n_node_pmf = datasets.get_dataset(config)
    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, num_workers=0)
    n_node_pmf = torch.from_numpy(n_node_pmf).to(config.device)

    train_iter = iter(train_loader)
    # create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting)

    # Build sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.training.eval_batch_size, config.data.num_channels,
                          config.data.max_node, config.data.max_node)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    num_train_steps = config.training.n_iters

    logging.info("Starting training loop at step %d." % (initial_step,))

    for step in tqdm.tqdm(range(initial_step, num_train_steps + 1)):
        try:
            graphs = next(train_iter)
        except :
            train_iter = train_loader.__iter__()
            continue
            # graphs = next(train_iter)
        adj, mask = dense_adj(graphs, config.data.max_node, scaler, config.data.dequantization)
        batch = (adj, mask)
        # Execute one training step
        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
            writer.add_scalar("training_loss", loss, step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq == 0:
            save_checkpoint(checkpoint_meta_dir, state)

def sde_evaluate(config, workdir, eval_folder="eval"):
    """Evaluate trained models.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints.
        eval_folder: The subfolder for storing evaluation results. Default to "eval".
    """

    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Build data pipeline
    train_ds, _, test_ds, n_node_pmf = datasets.get_dataset(config)
    n_node_pmf = torch.from_numpy(n_node_pmf).to(config.device)

    # Creat data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                               N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (config.eval.batch_size,
                          config.data.num_channels,
                          config.data.max_node, config.data.max_node)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
        eval_stats_fn = get_stats_eval(config)
        nn_eval_fn = get_nn_eval(config)

    # Begin evaluation
    begin_ckpt = config.eval.begin_ckpt
    logging.info("begin checkpoint: %d" % (begin_ckpt,))
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        # Wait if the target checkpoint doesn't exist yet
        waiting_message_printed = False
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        while not os.path.exists(ckpt_filename):
            if not waiting_message_printed:
                logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
                waiting_message_printed = True
            time.sleep(60)

        # Wait for 2 additional mins in case the file exists but is not ready for reading
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
        try:
            state = restore_checkpoint(ckpt_path, state, device=config.device)
        except:
            time.sleep(60)
            try:
                state = restore_checkpoint(ckpt_path, state, device=config.device)
            except:
                time.sleep(120)
                state = restore_checkpoint(ckpt_path, state, device=config.device)

        ema.copy_to(score_model.parameters())

        # Generate samples and compute MMD stats
        if config.eval.enable_sampling:
            num_sampling_rounds = int(np.ceil(config.eval.num_samples / config.eval.batch_size))
            all_samples = []
            for r in range(num_sampling_rounds):
                logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))
                sample, sample_steps, sample_nodes = sampling_fn(score_model, n_node_pmf)
                logging.info("sample steps: %d" % sample_steps)
                sample_list = adj2graph(sample, sample_nodes)
                all_samples += sample_list
            all_samples = all_samples[:config.eval.num_samples]

            # save the graphs
            sampler_name = config.sampling.method

            if config.eval.save_graph:
                graph_file = os.path.join(eval_dir, sampler_name + "_ckpt_{}.pkl".format(ckpt))
                with open(graph_file, "wb") as f:
                    pickle.dump(all_samples, f)

            # evaluate
            eval_results = eval_stats_fn(test_ds, all_samples)
            all_res = []
            for key, values in eval_results.items():
                all_res.append(values)
                logging.info("sampling -- ckpt: {}, {}: {:.6f}".format(ckpt, key, values))
            logging.info("sampling -- ckpt: {}, {}: {:.6f}".format(ckpt, "mean", np.mean(all_res)))
            # Draw and save the graph visualize figs
            this_sample_dir = os.path.join(eval_dir, sampler_name + "_ckpt_{}".format(ckpt))
            if not os.path.exists(this_sample_dir):
                os.makedirs(this_sample_dir)
            visualize.visualize_graphs(all_samples[:32], this_sample_dir, config, remove=False)

            # NN-based metric
            nn_eval_results = nn_eval_fn(test_ds, all_samples)
            for key, values in nn_eval_results.items():
                logging.info("sampling -- ckpt: {}, {} mean: {:.6f} std: {:.6f}".
                             format(ckpt, key, values[0], values[1]))


run_train_dict = {
    'sde': sde_train
}

run_eval_dict = {
    'sde': sde_evaluate
}


def train(config, workdir):
    run_train_dict[config.model_type](config, workdir)


def evaluate(config, workdir, eval_folder='eval'):
    run_eval_dict[config.model_type](config, workdir, eval_folder)

