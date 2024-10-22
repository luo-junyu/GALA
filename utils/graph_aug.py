import torch
import numpy as np


class AugTransform:
    def __init__(self, aug_type, aug_strength=0.1):
        self.aug = aug_type
        self.aug_strength = aug_strength

    def __call__(self, data):
        if self.aug == 'dnodes':
            data_aug = self.drop_nodes(data.clone())
        elif self.aug == 'pedges':
            data_aug = self.permute_edges(data.clone())
        elif self.aug == 'subgraph':
            data_aug = self.subgraph(data.clone())
        elif self.aug == 'mask_nodes':
            data_aug = self.mask_nodes(data.clone())
        elif self.aug == 'none':
            data_aug = data.clone()

        elif self.aug == 'random2':
            n = np.random.randint(2)
            if n == 0:
                data_aug = self.permute_edges(data.clone())
            elif n == 1:
                data_aug = self.mask_nodes(data.clone())
            else:
                print('sample error')
                assert False

        elif self.aug == 'random3':
            n = np.random.randint(3)
            if n == 0:
                data_aug = self.drop_nodes(data.clone())
            elif n == 1:
                data_aug = self.permute_edges(data.clone())
            elif n == 2:
                data_aug = self.subgraph(data.clone())
            else:
                print('sample error')
                assert False

        elif self.aug == 'without_pedges':
            n = np.random.randint(3)
            if n == 0:
                data_aug = self.drop_nodes(data.clone())
            elif n == 1:
                data_aug = self.mask_nodes(data.clone())
            elif n == 2:
                data_aug = self.subgraph(data.clone())
            else:
                print('sample error')
                assert False

        elif self.aug == 'without_subgraph':
            n = np.random.randint(3)
            if n == 0:
                data_aug = self.drop_nodes(data.clone())
            elif n == 1:
                data_aug = self.mask_nodes(data.clone())
            elif n == 2:
                data_aug = self.permute_edges(data.clone())
            else:
                print('sample error')
                assert False

        elif self.aug == 'random4':
            n = np.random.randint(4)
            if n == 0:
                data_aug = self.drop_nodes(data.clone())
            elif n == 1:
                data_aug = self.permute_edges(data.clone())
            elif n == 2:
                data_aug = self.subgraph(data.clone())
            elif n == 3:
                data_aug = self.mask_nodes(data.clone())
            else:
                print('sample error')
                assert False

        else:
            print('augmentation error')
            assert False

        return data_aug

    def drop_nodes(self, data):

        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        drop_num = int(node_num * self.aug_strength)

        idx_drop = np.random.choice(node_num, drop_num, replace=False)
        idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
        idx_dict = {idx_nondrop[n]: n for n in list(range(node_num - drop_num))}

        # data.x = data.x[idx_nondrop]
        edge_index = data.edge_index.numpy()

        adj = torch.zeros((node_num, node_num))
        adj[edge_index[0], edge_index[1]] = 1
        adj[idx_drop, :] = 0
        adj[:, idx_drop] = 0
        edge_index = adj.nonzero(as_tuple=False).t()

        data.edge_index = edge_index

        return data

    def permute_edges(self, data):

        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        permute_num = int(edge_num * self.aug_strength)

        edge_index = data.edge_index.transpose(0, 1).numpy()

        idx_add = np.random.choice(node_num, (permute_num, 2))
        edge_index = edge_index[np.random.choice(
            edge_num, edge_num - permute_num, replace=False)]
        data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

        return data

    def subgraph(self, data):

        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        sub_num = int(node_num * (1-self.aug_strength))

        edge_index = data.edge_index.numpy()

        idx_sub = [np.random.randint(node_num, size=1)[0]]
        idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])

        count = 0
        while len(idx_sub) <= sub_num:
            count = count + 1
            if count > node_num:
                break
            if len(idx_neigh) == 0:
                break
            sample_node = np.random.choice(list(idx_neigh))
            if sample_node in idx_sub:
                continue
            idx_sub.append(sample_node)
            idx_neigh.union(
                set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))

        idx_drop = [n for n in range(node_num) if n not in idx_sub]
        idx_nondrop = idx_sub
        idx_dict = {idx_nondrop[n]: n for n in list(range(len(idx_nondrop)))}

        edge_index = data.edge_index.numpy()

        adj = torch.zeros((node_num, node_num))
        adj[edge_index[0], edge_index[1]] = 1
        adj[idx_drop, :] = 0
        adj[:, idx_drop] = 0
        # edge_index = adj.nonzero().t()
        edge_index = adj.nonzero(as_tuple=False).t()

        data.edge_index = edge_index

        return data

    def mask_nodes(self, data):

        node_num, feat_dim = data.x.size()
        mask_num = int(node_num * self.aug_strength)

        idx_mask = np.random.choice(node_num, mask_num, replace=False)
        data.x[idx_mask] = torch.tensor(np.random.normal(
            loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)

        return data

