import os
import logging
import torch
from torch_scatter import scatter


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


def adj2graph(adj, sample_nodes):
    """Covert the PyTorch tensor adjacency matrices to numpy array.

    Args:
        adj: [Batch_size, channel, Max_node, Max_node], assume channel=1
        sample_nodes: [Batch_size]
    """
    adj_list = []
    # discretization
    adj[adj >= 0.5] = 1.
    adj[adj < 0.5] = 0.
    for i in range(adj.shape[0]):
        adj_tmp = adj[i, 0]
        # symmetric
        adj_tmp = torch.tril(adj_tmp, -1)
        adj_tmp = adj_tmp + adj_tmp.transpose(0, 1)
        # truncate
        adj_tmp = adj_tmp.cpu().numpy()[:sample_nodes[i], :sample_nodes[i]]
        adj_list.append(adj_tmp)

    return adj_list
