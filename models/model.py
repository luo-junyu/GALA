import torch
import torch.nn as nn
from .gnn import Encoder
import torch.nn.functional as F


class GNN(nn.Module):
    def __init__(self, dataset_num_features, hidden_dim, num_gc_layers, num_class, args, device):
        super(GNN, self).__init__()

        self.device = device

        self.embedding_dim = hidden_dim
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers, device=device, conv_type=args.conv_type,
                               use_bn=args.use_bn, JK=args.JK, global_pool=args.global_pool)

        if args.use_bn:
            self.proj_head = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.BatchNorm1d(self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            )
            self.classifier = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.BatchNorm1d(self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embedding_dim, num_class)
            )
        else:
            self.proj_head = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            )
            self.classifier = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embedding_dim, num_class)
            )

        self.init_emb()

    def init_emb(self):
        # initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):

        if x is None:
            x = torch.ones(batch.shape[0]).to(self.device)

        y = self.encoder(x, edge_index, batch)

        y_proj = self.proj_head(y)

        return y, F.normalize(y_proj, dim=1)

    def loss_cal(self, x, x_aug):

        T = 0.2
        batch_size, _ = x.size()

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug)

        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss