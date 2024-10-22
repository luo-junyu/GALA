import torch
import random
import numpy as np
import logging
import time
import os
from torch_scatter import scatter
import torch.nn.functional as F


def compute_kernel_batch(x):
    batch_size = x.size(0)
    num_aug = x.size(1)
    dim = x.size(2)
    n_samples = batch_size * num_aug

    y = x.clone()
    x = x.unsqueeze(1).unsqueeze(3)  # (B, 1, n, 1, d)
    y = y.unsqueeze(0).unsqueeze(2)  # (1, B, 1, n, d)
    tiled_x = x.expand(batch_size, batch_size, num_aug, num_aug, dim)
    tiled_y = y.expand(batch_size, batch_size, num_aug, num_aug, dim)

    L2_distance = (tiled_x - tiled_y).pow(2).sum(-1)
    bandwidth = torch.sum(L2_distance.detach()) / (n_samples ** 2 - n_samples)

    return torch.exp(-L2_distance / bandwidth)


def compute_mmd_batch(x):
    batch_size = x.size(0)
    batch_kernel = compute_kernel_batch(x)  # B*B*n*n
    batch_kernel_mean = batch_kernel.reshape(batch_size, batch_size, -1).mean(2)  # B*B
    self_kernel = torch.diag(batch_kernel_mean)
    x_kernel = self_kernel.unsqueeze(1).expand(batch_size, batch_size)
    y_kernel = self_kernel.unsqueeze(0).expand(batch_size, batch_size)
    mmd = x_kernel + y_kernel - 2*batch_kernel_mean

    return mmd.detach()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def create_mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

def timeleft(start_time, epoch, total_epoch):
    time_per_epoch = (time.time() - start_time) / epoch
    time_left = time_per_epoch * (total_epoch - epoch)
    return time_left


def get_logger(args):
    create_mkdir(args.log_dir)
    log_path = os.path.join(args.log_dir, args.DS+'_'+args.log_file)
    print('logging into %s' % log_path)

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    # logger.info('#' * 20)
    # localtime = time.asctime(time.localtime(time.time()))
    # logger.info("%s" % localtime)

    # record arguments
    # args_str = ""
    # for k, v in sorted(vars(args).items()):
    #     args_str += "%s" % k + "=" + "%s" % v + "; "
    # logger.info(args_str)
    # print(args_str)
    # logger.info("args.DS: %s" % args.DS)

    return logger


@torch.no_grad()
def to_dense_adj(edge_index, batch=None, edge_attr=None, max_num_nodes=None):
    """Converts batched sparse adjacency matrices given by edge indices and
    edge attributes to a single dense batched adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)

    Returns:
        adj: [batch_size, max_num_nodes, max_num_nodes] Dense adjacency matrices.
        mask: Mask for dense adjacency matrices.
    """

    # breakpoint()


    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = num_nodes.max().item()

    elif idx1.max() >= max_num_nodes or idx2.max() >= max_num_nodes:
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = torch.ones(idx0.numel(), device=edge_index.device)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.size())[1:]
    adj = torch.zeros(size, dtype=edge_attr.dtype, device=edge_index.device)

    flattened_size = batch_size * max_num_nodes * max_num_nodes
    adj = adj.view([flattened_size] + list(adj.size())[3:])
    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    scatter(edge_attr, idx, dim=0, out=adj, reduce='add')
    adj = adj.view(size)

    node_idx = torch.arange(batch.size(0), dtype=torch.long, device=edge_index.device)
    node_idx = (node_idx - cum_nodes[batch]) + (batch * max_num_nodes)
    mask = torch.zeros(batch_size * max_num_nodes, dtype=adj.dtype, device=adj.device)
    mask[node_idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    mask = mask[:, None, :] * mask[:, :, None]

    return adj, mask


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        if not os.path.exists(os.path.dirname(ckpt_dir)):
            os.makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)

def save_model(ckpt_dir, model):
    saved_state = {
        'model': model.state_dict(),
    }
    torch.save(saved_state, ckpt_dir)


def dense_adj(graph_data, max_num_nodes, scaler=None, dequantization=False):
    """Convert PyG DataBatch to dense adjacency matrices.

    Args:
        graph_data: DataBatch object.
        max_num_nodes: The size of the output node dimension.
        scaler: Data normalizer.
        dequantization: uniform dequantization.

    Returns:
        adj: Dense adjacency matrices.
        mask: Mask for adjacency matrices.
    """

    graph_data = graph_data.to('cuda')
    adj, adj_mask = to_dense_adj(graph_data.edge_index, graph_data.batch, max_num_nodes=max_num_nodes)  # [B, N, N]
    if dequantization:
        noise = torch.rand_like(adj)
        noise = torch.tril(noise, -1)
        noise = noise + noise.transpose(1, 2)
        adj = (noise + adj) / 2.
    adj = scaler(adj[:, None, :, :])
    # set diag = 0 in adj_mask
    adj_mask = torch.tril(adj_mask, -1)
    adj_mask = adj_mask + adj_mask.transpose(1, 2)

    return adj, adj_mask[:, None, :, :]


def adj_graph(adj, mask, data):
    sample_nodes = mask.sum(dim=2)[:, 0, 0] + 1
    sample_nodes = sample_nodes.cpu().numpy().astype(int)

    adj[adj >= 0] = 1.
    adj[adj < 0] = 0.
    res_data = []

    for i in range(adj.shape[0]):
        adj_tmp = adj[i, 0]
        adj_tmp = torch.tril(adj_tmp, -1)
        adj_tmp = adj_tmp + adj_tmp.transpose(0, 1)

        adj_tmp = adj_tmp[:sample_nodes[i], :sample_nodes[i]]
        new_edge_index = torch.transpose(torch.nonzero(adj_tmp), 0, 1).long()
        old_edges = data[i].edge_index.shape[-1]
        new_edges = new_edge_index.shape[-1]
        import copy
        datai = copy.deepcopy(data[i])
        datai.edge_index = new_edge_index
        res_data.append(datai)
        print(f'shape change: {new_edges-old_edges} {old_edges} -> {new_edges}')

    from torch_geometric.data import Data, Batch
    return Batch.from_data_list(res_data)


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss
