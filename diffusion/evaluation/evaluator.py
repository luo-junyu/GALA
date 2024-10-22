import networkx as nx
from .structure_evaluator import mmd_eval
from .gin_evaluator import nn_based_eval
from torch_geometric.utils import to_networkx
import torch
import torch.nn.functional as F
import dgl


def get_stats_eval(config):

    if config.eval.mmd_distance.lower() == 'rbf':
        method = [('degree', 1., 'argmax'), ('cluster', 0.1, 'argmax'),
                  ('spectral', 1., 'argmax')]
    else:
        raise ValueError

    def eval_stats_fn(test_dataset, pred_graph_list):
        pred_G = [nx.from_numpy_matrix(pred_adj) for pred_adj in pred_graph_list]
        sub_pred_G = []
        if config.eval.max_subgraph:
            for G in pred_G:
                CGs = [G.subgraph(c) for c in nx.connected_components(G)]
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
                sub_pred_G += [CGs[0]]
            pred_G = sub_pred_G

        test_G = [to_networkx(test_dataset[i], to_undirected=True, remove_self_loops=True)
                  for i in range(len(test_dataset))]
        results = mmd_eval(test_G, pred_G, method)
        return results

    return eval_stats_fn


def get_nn_eval(config):

    if hasattr(config.eval, "N_gin"):
        N_gin = config.eval.N_gin
    else:
        N_gin = 10

    def nn_eval_fn(test_dataset, pred_graph_list):
        pred_G = [nx.from_numpy_matrix(pred_adj) for pred_adj in pred_graph_list]
        sub_pred_G = []
        if config.eval.max_subgraph:
            for G in pred_G:
                CGs = [G.subgraph(c) for c in nx.connected_components(G)]
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
                sub_pred_G += [CGs[0]]
            pred_G = sub_pred_G
        test_G = [to_networkx(test_dataset[i], to_undirected=True, remove_self_loops=True)
                  for i in range(len(test_dataset))]

        results = nn_based_eval(test_G, pred_G, N_gin)
        return results

    return nn_eval_fn
