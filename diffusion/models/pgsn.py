import torch.nn as nn
import torch
import functools
from torch_geometric.utils import dense_to_sparse

from . import utils, layers, gnns

get_act = layers.get_act
conv1x1 = layers.conv1x1


@utils.register_model(name='PGSN')
class PGSN(nn.Module):
    """Position enhanced graph score network."""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.act = act = get_act(config)

        # get model construction paras
        self.nf = nf = config.model.nf
        self.num_gnn_layers = num_gnn_layers = config.model.num_gnn_layers
        dropout = config.model.dropout
        self.embedding_type = embedding_type = config.model.embedding_type.lower()
        self.rw_depth = rw_depth = config.model.rw_depth
        self.edge_th = config.model.edge_th

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == 'positional':
            embed_dim = nf
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        # timestep embedding layers
        modules.append(nn.Linear(embed_dim, nf * 4))
        modules.append(nn.Linear(nf * 4, nf * 4))

        # graph size condition embedding
        self.size_cond = size_cond = config.model.size_cond
        if size_cond:
            self.size_onehot = functools.partial(nn.functional.one_hot, num_classes=config.data.max_node + 1)
            modules.append(nn.Linear(config.data.max_node + 1, nf * 4))
            modules.append(nn.Linear(nf * 4, nf * 4))

        channels = config.data.num_channels
        assert channels == 1, "Without edge features."

        # degree onehot
        self.degree_max = self.config.data.max_node // 2
        self.degree_onehot = functools.partial(
            nn.functional.one_hot,
            num_classes=self.degree_max + 1)

        # project edge features
        modules.append(conv1x1(channels, nf // 2))
        modules.append(conv1x1(rw_depth + 1, nf // 2))

        # project node features
        self.x_ch = nf
        self.pos_ch = nf // 2
        modules.append(nn.Linear(self.degree_max + 1, self.x_ch))
        modules.append(nn.Linear(rw_depth, self.pos_ch))

        # GNN
        modules.append(gnns.pos_gnn(act, self.x_ch, self.pos_ch, nf, config.data.max_node,
                                    config.model.graph_layer, num_gnn_layers,
                                    heads=config.model.heads, edge_dim=nf//2, temb_dim=nf * 4,
                                    dropout=dropout, attn_clamp=config.model.attn_clamp))

        # output
        modules.append(conv1x1(nf // 2, nf // 2))
        modules.append(conv1x1(nf // 2, channels))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond, *args, **kwargs):
        mask = kwargs['mask']
        modules = self.all_modules
        m_idx = 0

        # Sinusoidal positional embeddings
        timesteps = time_cond
        temb = layers.get_timestep_embedding(timesteps, self.nf)

        # time embedding
        temb = modules[m_idx](temb)
        m_idx += 1
        temb = modules[m_idx](self.act(temb))
        m_idx += 1

        if self.size_cond:
            with torch.no_grad():
                node_mask = utils.mask_adj2node(mask.squeeze(1))  # [B, N]
                num_node = torch.sum(node_mask, dim=-1)  # [B]
                num_node = self.size_onehot(num_node.to(torch.long)).to(torch.float)
            num_node_emb = modules[m_idx](num_node)
            m_idx += 1
            num_node_emb = modules[m_idx](self.act(num_node_emb))
            m_idx += 1
            temb = temb + num_node_emb

        if not self.config.data.centered:
            # rescale the input data to [-1, 1]
            x = x * 2. - 1.

        with torch.no_grad():
            # continuous-valued graph adjacency matrices
            cont_adj = ((x + 1.) / 2.).clone()
            cont_adj = (cont_adj * mask).squeeze(1)  # [B, N, N]
            cont_adj = cont_adj.clamp(min=0., max=1.)
            if self.edge_th > 0.:
                cont_adj[cont_adj < self.edge_th] = 0.

            # discretized graph adjacency matrices
            adj = x.squeeze(1).clone()  # [B, N, N]
            adj[adj >= 0.] = 1.
            adj[adj < 0.] = 0.
            adj = adj * mask.squeeze(1)

        # extract RWSE and Shortest-Path Distance
        x_pos, spd_onehot = utils.get_rw_feat(self.rw_depth, adj)

        # edge [B, N, N, F]
        dense_edge_ori = modules[m_idx](x).permute(0, 2, 3, 1)
        m_idx += 1
        dense_edge_spd = modules[m_idx](spd_onehot).permute(0, 2, 3, 1)
        m_idx += 1

        # Use Degree as node feature
        x_degree = torch.sum(cont_adj, dim=-1)  # [B, N]
        x_degree = x_degree.clamp(max=float(self.degree_max))
        x_degree = self.degree_onehot(x_degree.to(torch.long)).to(torch.float)  # [B, N, max_node]
        x_degree = modules[m_idx](x_degree)  # projection layer [B, N, nf]
        m_idx += 1

        # pos encoding
        x_pos = modules[m_idx](x_pos)
        m_idx += 1

        # Dense to sparse node [BxN, -1]
        x_degree = x_degree.reshape(-1, self.x_ch)
        x_pos = x_pos.reshape(-1, self.pos_ch)
        dense_index = cont_adj.nonzero(as_tuple=True)
        edge_index, _ = dense_to_sparse(cont_adj)

        # Run GNN layers
        h_dense_edge = modules[m_idx](x_degree, x_pos, edge_index, dense_edge_ori, dense_edge_spd, dense_index, temb)
        m_idx += 1

        # Output
        h = self.act(modules[m_idx](self.act(h_dense_edge)))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1

        # make edge estimation symmetric
        h = (h + h.transpose(2, 3)) / 2. * mask

        assert m_idx == len(modules)

        return h
