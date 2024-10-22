import math
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import numpy as np


class PosTransLayer(MessagePassing):
    """Involving the edge feature and updating position feature. Multiply Msg."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, pos_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, act=None, attn_clamp: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(PosTransLayer, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.pos_channels = pos_channels
        self.in_channels = in_channels = x_channels + pos_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.attn_clamp = attn_clamp

        if act is None:
            self.act = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.act = act

        self.lin_key = Linear(in_channels, heads * out_channels)
        self.lin_query = Linear(in_channels, heads * out_channels)
        self.lin_value = Linear(in_channels, heads * out_channels)

        self.lin_edge0 = Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_edge1 = Linear(edge_dim, heads * out_channels, bias=False)

        self.lin_pos = Linear(heads * out_channels, pos_channels, bias=False)

        self.lin_skip = Linear(x_channels, heads * out_channels, bias=bias)
        self.norm1 = nn.GroupNorm(num_groups=min(heads * out_channels // 4, 32),
                                  num_channels=heads * out_channels, eps=1e-6)
        self.norm2 = nn.GroupNorm(num_groups=min(heads * out_channels // 4, 32),
                                  num_channels=heads * out_channels, eps=1e-6)
        # FFN
        self.FFN = nn.Sequential(Linear(heads * out_channels, heads * out_channels),
                                 self.act,
                                 Linear(heads * out_channels, heads * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_skip.reset_parameters()
        self.lin_edge0.reset_parameters()
        self.lin_edge1.reset_parameters()
        self.lin_pos.reset_parameters()

    def forward(self, x: OptTensor,
                pos: Tensor,
                edge_index: Adj,
                edge_attr: OptTensor = None
                ) -> Tuple[Tensor, Tensor]:
        """"""

        H, C = self.heads, self.out_channels

        x_feat = torch.cat([x, pos], -1)
        query = self.lin_query(x_feat).view(-1, H, C)
        key = self.lin_key(x_feat).view(-1, H, C)
        value = self.lin_value(x_feat).view(-1, H, C)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x, out_pos = self.propagate(edge_index, query=query, key=key, value=value, pos=pos, edge_attr=edge_attr,
                                        size=None)

        out_x = out_x.view(-1, self.heads * self.out_channels)

        # skip connection for x
        x_r = self.lin_skip(x)
        out_x = (out_x + x_r) / math.sqrt(2)
        out_x = self.norm1(out_x)

        # FFN
        out_x = (out_x + self.FFN(out_x)) / math.sqrt(2)
        out_x = self.norm2(out_x)

        # skip connection for pos
        out_pos = pos + torch.tanh(pos + out_pos)

        return out_x, out_pos

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                pos_j: Tensor,
                edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:

        edge_attn = self.lin_edge0(edge_attr).view(-1, self.heads, self.out_channels)
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / math.sqrt(self.out_channels)
        if self.attn_clamp:
            alpha = alpha.clamp(min=-5., max=5.)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # node feature message
        msg = value_j
        msg = msg * self.lin_edge1(edge_attr).view(-1, self.heads, self.out_channels)
        msg = msg * alpha.view(-1, self.heads, 1)

        # node position message
        pos_msg = pos_j * self.lin_pos(msg.reshape(-1, self.heads * self.out_channels))

        return msg, pos_msg

    def aggregate(self, inputs: Tuple[Tensor, Tensor], index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        if ptr is not None:
            raise NotImplementedError("Not implement Ptr in aggregate")
        else:
            return (scatter(inputs[0], index, 0, dim_size=dim_size, reduce=self.aggr),
                    scatter(inputs[1], index, 0, dim_size=dim_size, reduce="mean"))

    def update(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        return inputs

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
