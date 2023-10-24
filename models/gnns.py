import torch.nn as nn
import torch
from .trans_layers import *


class pos_gnn(nn.Module):
    def __init__(self, act, x_ch, pos_ch, out_ch, max_node, graph_layer, n_layers=3, edge_dim=None, heads=4,
                 temb_dim=None, dropout=0.1, attn_clamp=False):
        super().__init__()
        self.out_ch = out_ch
        self.Dropout_0 = nn.Dropout(dropout)
        self.act = act
        self.max_node = max_node
        self.n_layers = n_layers

        if temb_dim is not None:
            self.Dense_node0 = nn.Linear(temb_dim, x_ch)
            self.Dense_node1 = nn.Linear(temb_dim, pos_ch)
            self.Dense_edge0 = nn.Linear(temb_dim, edge_dim)
            self.Dense_edge1 = nn.Linear(temb_dim, edge_dim)

        self.convs = nn.ModuleList()
        self.edge_convs = nn.ModuleList()
        self.edge_layer = nn.Linear(edge_dim * 2 + self.out_ch, edge_dim)

        for i in range(n_layers):
            if i == 0:
                self.convs.append(eval(graph_layer)(x_ch, pos_ch, self.out_ch//heads, heads, edge_dim=edge_dim*2,
                                                    act=act, attn_clamp=attn_clamp))
            else:
                self.convs.append(eval(graph_layer)
                                  (self.out_ch, pos_ch, self.out_ch//heads, heads, edge_dim=edge_dim*2, act=act,
                                   attn_clamp=attn_clamp))
            self.edge_convs.append(nn.Linear(self.out_ch, edge_dim*2))

    def forward(self, x_degree, x_pos, edge_index, dense_ori, dense_spd, dense_index, temb=None):
        """
        Args:
            x_degree: node degree feature [B*N, x_ch]
            x_pos: node rwpe feature [B*N, pos_ch]
            edge_index: [2, edge_length]
            dense_ori: edge feature [B, N, N, nf//2]
            dense_spd: edge shortest path distance feature [B, N, N, nf//2]
            dense_index
            temb: [B, temb_dim]
        """

        B, N, _, _ = dense_ori.shape

        if temb is not None:
            dense_ori = dense_ori + self.Dense_edge0(self.act(temb))[:, None, None, :]
            dense_spd = dense_spd + self.Dense_edge1(self.act(temb))[:, None, None, :]

            temb = temb.unsqueeze(1).repeat(1, self.max_node, 1)
            temb = temb.reshape(-1, temb.shape[-1])
            x_degree = x_degree + self.Dense_node0(self.act(temb))
            x_pos = x_pos + self.Dense_node1(self.act(temb))

        dense_edge = torch.cat([dense_ori, dense_spd], dim=-1)

        ori_edge_attr = dense_edge
        h = x_degree
        h_pos = x_pos

        for i_layer in range(self.n_layers):
            h_edge = dense_edge[dense_index]
            # update node feature
            h, h_pos = self.convs[i_layer](h, h_pos, edge_index, h_edge)
            h = self.Dropout_0(h)
            h_pos = self.Dropout_0(h_pos)

            # update dense edge feature
            h_dense_node = h.reshape(B, N, -1)
            cur_edge_attr = h_dense_node.unsqueeze(1) + h_dense_node.unsqueeze(2)  # [B, N, N, nf]
            dense_edge = (dense_edge + self.act(self.edge_convs[i_layer](cur_edge_attr))) / math.sqrt(2.)
            dense_edge = self.Dropout_0(dense_edge)

        # Concat edge attribute
        h_dense_edge = torch.cat([ori_edge_attr, dense_edge], dim=-1)
        h_dense_edge = self.edge_layer(h_dense_edge).permute(0, 3, 1, 2)

        return h_dense_edge
