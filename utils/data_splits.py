import torch

import torch

def get_splits_in_domain(dataset, ratio=0.8):
    data_idx = torch.randperm(len(dataset))
    dataset = dataset[data_idx]
    num_data = len(data_idx)
    num_train = int(num_data * ratio)
    train_index = data_idx[:num_train]
    test_index = data_idx[num_train:]

    train_dataset = dataset[train_index]
    test_dataset = dataset[test_index]

    return train_dataset, test_dataset


def get_domain_splits(dataset, split=4):
    node_density = []

    for i in range(len(dataset)):
        node_density.append(dataset[i].num_edges / (dataset[i].num_nodes * (dataset[i].num_nodes - 1)))
    node_density = torch.tensor(node_density)
    node_density, data_idx = torch.sort(node_density, descending=False)

    return_dataset = []
    for i in range(split):
        return_dataset.append(dataset[data_idx[i*(len(dataset) // split) : (i+1)*(len(dataset) // split)]])

    return return_dataset

def get_splits(dataset, args):
    """
    Test-Time Split
    split with node density
    """
    source_target_split = 0.5
    train_test_split = 0.8
    # reshuffle the dataset by node density
    node_density = []
    for i in range(len(dataset)):
        node_density.append(dataset[0].num_edges / (dataset[i].num_nodes * (dataset[i].num_nodes-1)))
    node_density = torch.tensor(node_density)
    node_density, data_idx = torch.sort(node_density, descending=False)
    # source target split
    source_index = data_idx[:int(len(dataset) * source_target_split)]
    target_index = data_idx[int(len(dataset) * source_target_split):]
    # split train val test
    num_source = len(source_index)
    num_source_train = int(num_source * train_test_split)
    num_target = len(target_index)
    num_target_train = int(num_target * train_test_split)
    
    source_train_index = source_index[:num_source_train]
    source_val_index = source_index[num_source_train:]
    target_train_index = target_index[:num_target_train]
    target_test_index = target_index[num_target_train:]

    source_train_dataset = dataset[source_train_index]
    source_val_dataset = dataset[source_val_index]
    target_train_dataset = dataset[target_train_index]
    target_test_dataset = dataset[target_test_index]

    # breakpoint()

    # print the dataset info
    print(f'[Dataset] Total: {len(dataset)}')
    print(f'[Source] Train: {len(source_train_dataset)}, Val: {len(source_val_dataset)}')
    print(f'[Target] Train: {len(target_train_dataset)}, Test: {len(target_test_dataset)}')

    return source_train_dataset, source_val_dataset, target_train_dataset, target_test_dataset