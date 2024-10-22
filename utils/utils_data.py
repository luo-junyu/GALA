from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
import torch_geometric.transforms as T
from torch.utils.data import Dataset
import torch
from .data_splits import get_domain_splits, get_splits_in_domain
import numpy as np
import random
import time
from torch_geometric.data import Data, Batch
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx, from_networkx
import community


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx

def get_dataset(DS, path, args):
    setup_seed(0)
   
    def graph_cluster(data):
        graph = to_networkx(data, to_undirected=True)
        partition = community.best_partition(graph)
        data.cluster = torch.tensor(list(partition.values()))
        return data

    def pre_process(data):
        data = graph_cluster(data)
        return data

    dataset = TUDataset(path, name=DS, use_node_attr=True, pre_transform=pre_process)

    # dataset.data.edge_attr = None

    print(f'Dataset: {DS}, Length: {len(dataset)}')

    source_split_index = args.source_index
    target_split_index = args.target_index
    split = args.data_split

    split_dataset = get_domain_splits(dataset, split)
    source_dataset = split_dataset[source_split_index]
    target_dataset = split_dataset[target_split_index]

    source_train_dataset, source_val_dataset = get_splits_in_domain(source_dataset)
    target_train_dataset, target_test_dataset = get_splits_in_domain(target_dataset)


    return dataset, (source_train_dataset, source_val_dataset, target_train_dataset, target_test_dataset)

def split_confident_data(model, dataset, device, args):
    """
    Split Target Dataset into Confident and Inconfident Dataset
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    confident_dataset = []
    inconfident_dataset = []
    correct_count = 0
    confident_percentage = args.confident_percentage

    for data in loader:
        data = data.to(device)
        _, feat_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
        prob = torch.softmax(model.classifier(feat_proj),dim=-1)

        # Filter with pseudo label confidence
        confident_id = prob.max(dim=-1)[0].topk(int(len(prob)*confident_percentage))[1]
        confident_label = prob.max(dim=-1)[1]
        confident_mask = torch.zeros(len(prob),dtype=torch.bool).to(device)
        confident_mask[confident_id] = True

        confident_dataset += data[confident_mask]
        inconfident_dataset += data[~(confident_mask)]
        correct_count += (prob.max(dim=-1)[1] == data.y).sum().item()

    return confident_dataset, inconfident_dataset
