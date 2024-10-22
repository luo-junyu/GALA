import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GINConv, global_add_pool, GATConv, SAGEConv, GCNConv, global_mean_pool, global_max_pool


class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, device, conv_type='SAGE', use_bn=0, JK='last',
                 global_pool='sum'):
        super(Encoder, self).__init__()

        self.use_bn = use_bn
        self.device = device

        self.num_gc_layers = num_gc_layers
        self.JK = JK
        self.global_pool = global_pool

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if conv_type == 'GIN':
                if i:
                    if use_bn:
                        nn = Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(), Linear(dim, dim))
                    else:
                        nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
                else:
                    if use_bn:
                        nn = Sequential(Linear(num_features, dim), BatchNorm1d(dim), ReLU(), Linear(dim, dim))
                    else:
                        nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
                conv = GINConv(nn)
            elif conv_type == 'GAT':
                if i:
                    conv = GATConv(dim, dim, heads=4, concat=False)
                else:
                    conv = GATConv(num_features, dim, heads=4, concat=False)
            elif conv_type == 'SAGE':
                if i:
                    conv = SAGEConv(dim, dim)
                else:
                    conv = SAGEConv(num_features, dim)
            elif conv_type == 'GCN':
                if i:
                    conv = GCNConv(dim, dim)
                else:
                    conv = GCNConv(num_features, dim)
            else:
                raise ValueError("Invalid conv_type: %s" % conv_type)

            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):

        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(self.device)

        xs = []
        xpool = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            if self.use_bn:
                x = self.bns[i](x)

            xs.append(x)
            if self.global_pool == 'sum':
                xpool.append(global_add_pool(x, batch))
            elif self.global_pool == 'mean':
                xpool.append(global_mean_pool(x, batch))
            elif self.global_pool == 'max':
                xpool.append(global_max_pool(x, batch))
            else:
                raise NotImplementedError

        if self.JK == 'last':
            features = xpool[-1]
        elif self.JK == "sum":
            features = 0
            for layer in range(self.num_gc_layers):
                features += xpool[layer]
        else:
            raise NotImplementedError

        return features

