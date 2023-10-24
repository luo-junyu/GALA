"""Evaluation on random GIN features. Modified from https://github.com/uoguelph-mlrg/GGM-metrics"""

import torch
import numpy as np
import sklearn
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
import time
import dgl

from .gin import GIN


def load_feature_extractor(
        device, num_layers=3, hidden_dim=35, neighbor_pooling_type='sum',
        graph_pooling_type='sum', input_dim=1, edge_feat_dim=0,
        dont_concat=False, num_mlp_layers=2, output_dim=1,
        node_feat_loc='attr', edge_feat_loc='attr', init='orthogonal',
        **kwargs):

    model = GIN(num_layers=num_layers, hidden_dim=hidden_dim, neighbor_pooling_type=neighbor_pooling_type,
                graph_pooling_type=graph_pooling_type, input_dim=input_dim, edge_feat_dim=edge_feat_dim,
                num_mlp_layers=num_mlp_layers, output_dim=output_dim, init=init)

    model.node_feat_loc = node_feat_loc
    model.edge_feat_loc = edge_feat_loc

    model.eval()

    if dont_concat:
        model.forward = model.get_graph_embed_no_cat
    else:
        model.forward = model.get_graph_embed

    model.device = device
    return model.to(device)


def time_function(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        end = time.time()
        return results, end - start
    return wrapper


class GINMetric():
    def __init__(self, model):
        self.feat_extractor = model
        self.get_activations = self.get_activations_gin

    @time_function
    def get_activations_gin(self, generated_dataset, reference_dataset):
        return self._get_activations(generated_dataset, reference_dataset)

    def _get_activations(self, generated_dataset, reference_dataset):
        gen_activations = self.__get_activations_single_dataset(generated_dataset)
        ref_activations = self.__get_activations_single_dataset(reference_dataset)

        scaler = StandardScaler()
        scaler.fit(ref_activations)
        ref_activations = scaler.transform(ref_activations)
        gen_activations = scaler.transform(gen_activations)

        return gen_activations, ref_activations

    def __get_activations_single_dataset(self, dataset):

        node_feat_loc = self.feat_extractor.node_feat_loc
        edge_feat_loc = self.feat_extractor.edge_feat_loc

        ndata = [node_feat_loc] if node_feat_loc in dataset[0].ndata else '__ALL__'
        edata = [edge_feat_loc] if edge_feat_loc in dataset[0].edata else '__ALL__'
        graphs = dgl.batch(dataset, ndata=ndata, edata=edata).to(self.feat_extractor.device)

        if node_feat_loc not in graphs.ndata:  # Use degree as features
            feats = graphs.in_degrees() + graphs.out_degrees()
            feats = feats.unsqueeze(1).type(torch.float32)
        else:
            feats = graphs.ndata[node_feat_loc]

        graph_embeds = self.feat_extractor(graphs, feats)
        return graph_embeds.cpu().detach().numpy()

    def evaluate(self, *args, **kwargs):
        raise Exception('Must be implemented by child class')


class MMDEvaluation(GINMetric):
    def __init__(self, model, kernel='rbf', sigma='range', multiplier='mean'):
        super().__init__(model)

        if multiplier == 'mean':
            self.__get_sigma_mult_factor = self.__mean_pairwise_distance
        elif multiplier == 'median':
            self.__get_sigma_mult_factor = self.__median_pairwise_distance
        elif multiplier is None:
            self.__get_sigma_mult_factor = lambda *args, **kwargs: 1
        else:
            raise Exception(multiplier)

        if 'rbf' in kernel:
            if sigma == 'range':
                self.base_sigmas = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0])

                if multiplier == 'mean':
                    self.name = 'mmd_rbf'
                elif multiplier == 'median':
                    self.name = 'mmd_rbf_adaptive_median'
                else:
                    self.name = 'mmd_rbf_adaptive'
            elif sigma == 'one':
                self.base_sigmas = np.array([1])

                if multiplier == 'mean':
                    self.name = 'mmd_rbf_single_mean'
                elif multiplier == 'median':
                    self.name = 'mmd_rbf_single_median'
                else:
                    self.name = 'mmd_rbf_single'
            else:
                raise Exception(sigma)

            self.evaluate = self.calculate_MMD_rbf_quadratic

        elif 'linear' in kernel:
            self.evaluate = self.calculate_MMD_linear_kernel

        else:
            raise Exception()

    def __get_pairwise_distances(self, generated_dataset, reference_dataset):
        return sklearn.metrics.pairwise_distances(reference_dataset, generated_dataset, metric='euclidean', n_jobs=8)**2

    def __mean_pairwise_distance(self, dists_GR):
        return np.sqrt(dists_GR.mean())

    def __median_pairwise_distance(self, dists_GR):
        return np.sqrt(np.median(dists_GR))

    def get_sigmas(self, dists_GR):
        mult_factor = self.__get_sigma_mult_factor(dists_GR)
        return self.base_sigmas * mult_factor

    @time_function
    def calculate_MMD_rbf_quadratic(self, generated_dataset=None, reference_dataset=None):
        # https://github.com/djsutherland/opt-mmd/blob/master/two_sample/mmd.py

        if not isinstance(generated_dataset, torch.Tensor) and not isinstance(generated_dataset, np.ndarray):
            (generated_dataset, reference_dataset), _ = self.get_activations(generated_dataset, reference_dataset)

        GG = self.__get_pairwise_distances(generated_dataset, generated_dataset)
        GR = self.__get_pairwise_distances(generated_dataset, reference_dataset)
        RR = self.__get_pairwise_distances(reference_dataset, reference_dataset)

        max_mmd = 0
        sigmas = self.get_sigmas(GR)

        for sigma in sigmas:
            gamma = 1 / (2 * sigma**2)

            K_GR = np.exp(-gamma * GR)
            K_GG = np.exp(-gamma * GG)
            K_RR = np.exp(-gamma * RR)

            mmd = K_GG.mean() + K_RR.mean() - 2 * K_GR.mean()
            max_mmd = mmd if mmd > max_mmd else max_mmd

        return {self.name: max_mmd}

    @time_function
    def calculate_MMD_linear_kernel(self, generated_dataset=None, reference_dataset=None):
        # https://github.com/djsutherland/opt-mmd/blob/master/two_sample/mmd.py
        if not isinstance(generated_dataset, torch.Tensor) and not isinstance(generated_dataset, np.ndarray):
            (generated_dataset, reference_dataset), _ = self.get_activations(generated_dataset, reference_dataset)

        G_bar = generated_dataset.mean(axis=0)
        R_bar = reference_dataset.mean(axis=0)
        Z_bar = G_bar - R_bar
        mmd = Z_bar.dot(Z_bar)
        mmd = mmd if mmd >= 0 else 0
        return {'mmd_linear': mmd}


class prdcEvaluation(GINMetric):
    # From PRDC github: https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py#L54
    def __init__(self, *args, use_pr=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_pr = use_pr

    @time_function
    def evaluate(self, generated_dataset=None, reference_dataset=None, nearest_k=5):
        """ Computes precision, recall, density, and coverage given two manifolds. """

        if not isinstance(generated_dataset, torch.Tensor) and not isinstance(generated_dataset, np.ndarray):
            (generated_dataset, reference_dataset), _ = self.get_activations(generated_dataset, reference_dataset)

        real_nearest_neighbour_distances = self.__compute_nearest_neighbour_distances(reference_dataset, nearest_k)
        distance_real_fake = self.__compute_pairwise_distance(reference_dataset, generated_dataset)

        if self.use_pr:
            fake_nearest_neighbour_distances = self.__compute_nearest_neighbour_distances(generated_dataset, nearest_k)
            precision = (
                distance_real_fake <= np.expand_dims(real_nearest_neighbour_distances, axis=1)
            ).any(axis=0).mean()

            recall = (
                distance_real_fake <= np.expand_dims(fake_nearest_neighbour_distances, axis=0)
            ).any(axis=1).mean()

            f1_pr = 2 / ((1 / (precision + 1e-8)) + (1 / (recall + 1e-8)))
            result = dict(precision=precision, recall=recall, f1_pr=f1_pr)
        else:
            density = (1. / float(nearest_k)) * (
                    distance_real_fake <= np.expand_dims(real_nearest_neighbour_distances, axis=1)).sum(axis=0).mean()

            coverage = (distance_real_fake.min(axis=1) <= real_nearest_neighbour_distances).mean()

            f1_dc = 2 / ((1 / (density + 1e-8)) + (1 / (coverage + 1e-8)))
            result = dict(density=density, coverage=coverage, f1_dc=f1_dc)
        return result

    def __compute_pairwise_distance(self, data_x, data_y=None):
        """
        Args:
            data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
            data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
        Return:
            numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
        """
        if data_y is None:
            data_y = data_x
        dists = sklearn.metrics.pairwise_distances(data_x, data_y, metric='euclidean', n_jobs=8)
        return dists

    def __get_kth_value(self, unsorted, k, axis=-1):
        """
        Args:
            unsorted: numpy.ndarray of any dimensionality.
            k: int
        Return:
            kth values along the designated axis.
        """
        indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
        k_smallest = np.take_along_axis(unsorted, indices, axis=axis)
        kth_values = k_smallest.max(axis=axis)
        return kth_values

    def __compute_nearest_neighbour_distances(self, input_features, nearest_k):
        """
        Args:
            input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            nearest_k: int
        Return:
            Distances to kth nearest neighbours.
        """
        distances = self.__compute_pairwise_distance(input_features)
        radii = self.__get_kth_value(distances, k=nearest_k + 1, axis=-1)
        return radii


def nn_based_eval(graph_ref_list, graph_pred_list, N_gin=10):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    evaluators = []
    for _ in range(N_gin):
        gin = load_feature_extractor(device)
        evaluators.append(MMDEvaluation(model=gin, kernel='rbf', sigma='range', multiplier='mean'))
        evaluators.append(prdcEvaluation(model=gin, use_pr=True))
        evaluators.append(prdcEvaluation(model=gin, use_pr=False))

    ref_graphs = [dgl.from_networkx(g).to(device) for g in graph_ref_list]
    gen_graphs = [dgl.from_networkx(g).to(device) for g in graph_pred_list]

    metrics = {
        'mmd_rbf': [],
        'f1_pr': [],
        'f1_dc': []
    }
    for evaluator in evaluators:
        res, time = evaluator.evaluate(generated_dataset=gen_graphs, reference_dataset=ref_graphs)
        for key in list(res.keys()):
            if key in metrics:
                metrics[key].append(res[key])

    results = {
        'MMD_RBF': (np.mean(metrics['mmd_rbf']), np.std(metrics['mmd_rbf'])),
        'F1_PR': (np.mean(metrics['f1_pr']), np.std(metrics['f1_pr'])),
        'F1_DC': (np.mean(metrics['f1_dc']), np.std(metrics['f1_dc']))
    }
    return results
