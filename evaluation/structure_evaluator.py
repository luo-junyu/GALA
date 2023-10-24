"""MMD Evaluation on graph structure statistics. Modified from https://github.com/uoguelph-mlrg/GGM-metrics"""

import numpy as np
import networkx as nx
import numpy as np
# from scipy.linalg import toeplitz
# import pyemd
import concurrent.futures
from scipy.linalg import eigvalsh
from functools import partial


class Descriptor():
    def __init__(self, is_parallel=False, bins=100, kernel='rbf', sigma_type='single', **kwargs):
        self.is_parallel = is_parallel
        self.bins = bins
        self.max_workers = kwargs.get('max_workers')

        if kernel == 'rbf':
            self.distance = self.l2
            self.name += '_rbf'
        else:
            ValueError

        if sigma_type == 'argmax':
            log_sigmas = np.linspace(-5., 5., 50)
            # the first 30 sigma values is usually enough
            log_sigmas = log_sigmas[:30]
            self.sigmas = [np.exp(log_sigma) for log_sigma in log_sigmas]
        elif sigma_type == 'single':
            self.sigmas = kwargs['sigma']
        else:
            raise ValueError

    def evaluate(self, graph_ref_list, graph_pred_list):
        """Compute the distance between the distributions of two unordered sets of graphs.
        Args:
            graph_ref_list, graph_pred_list: two lists of networkx graphs to be evaluated.
        """

        graph_pred_list = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

        sample_pred = self.extract_features(graph_pred_list)
        sample_ref = self.extract_features(graph_ref_list)

        GG = self.disc(sample_pred, sample_pred, distance_scaling=self.distance_scaling)
        GR = self.disc(sample_pred, sample_ref, distance_scaling=self.distance_scaling)
        RR = self.disc(sample_ref, sample_ref, distance_scaling=self.distance_scaling)

        sigmas = self.sigmas
        max_mmd = 0
        mmd_dict = []
        for sigma in sigmas:
            gamma = 1 / (2 * sigma ** 2)

            K_GR = np.exp(-gamma * GR)
            K_GG = np.exp(-gamma * GG)
            K_RR = np.exp(-gamma * RR)

            mmd = K_GG.mean() + K_RR.mean() - (2 * K_GR.mean())
            mmd_dict.append((sigma, mmd))
            max_mmd = mmd if mmd > max_mmd else max_mmd

        # print(self.name, mmd_dict)

        return max_mmd

    def pad_histogram(self, x, y):
        # convert histogram values x and y to float, and pad them for equal length
        support_size = max(len(x), len(y))
        x = x.astype(np.float)
        y = y.astype(np.float)
        if len(x) < len(y):
            x = np.hstack((x, [0.] * (support_size - len(x))))
        elif len(y) < len(x):
            y = np.hstack((y, [0.] * (support_size - len(y))))

        return x, y

    # def emd(self, x, y, distance_scaling=1.0):
    #     support_size = max(len(x), len(y))
    #     x, y = self.pad_histogram(x, y)
    #
    #     d_mat = toeplitz(range(support_size)).astype(np.float)
    #     distance_mat = d_mat / distance_scaling
    #
    #     dist = pyemd.emd(x, y, distance_mat)
    #     return dist ** 2

    def l2(self, x, y, **kwargs):
        # gaussian rbf
        x, y = self.pad_histogram(x, y)
        dist = np.linalg.norm(x - y, 2)
        return dist ** 2

    def kernel_parallel_unpacked(self, x, samples2, kernel):
        dist = []
        for s2 in samples2:
            dist += [kernel(x, s2)]
        return dist

    def kernel_parallel_worker(self, t):
        return self.kernel_parallel_unpacked(*t)

    def disc(self, samples1, samples2, **kwargs):
        # Discrepancy between 2 samples
        tot_dist = []
        if not self.is_parallel:
            for s1 in samples1:
                for s2 in samples2:
                    tot_dist += [self.distance(s1, s2)]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for dist in executor.map(self.kernel_parallel_worker,
                                         [(s1, samples2, partial(self.distance, **kwargs)) for s1 in samples1]):
                    tot_dist += [dist]
        return np.array(tot_dist)


class degree(Descriptor):
    def __init__(self, *args, **kwargs):
        self.name = 'degree'
        self.sigmas = [kwargs.get('sigma', 1.0)]
        self.distance_scaling = 1.0
        super().__init__(*args, **kwargs)

    def extract_features(self, dataset):
        res = []
        if self.is_parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for deg_hist in executor.map(self.degree_worker, dataset):
                    res.append(deg_hist)
        else:
            for g in dataset:
                degree_hist = self.degree_worker(g)
                res.append(degree_hist)

        res = [s1 / np.sum(s1) for s1 in res]
        return res

    def degree_worker(self, G):
        return np.array(nx.degree_histogram(G))


class cluster(Descriptor):
    def __init__(self, *args, **kwargs):
        self.name = 'cluster'
        self.sigmas = [kwargs.get('sigma', [1.0 / 10])]
        super().__init__(*args, **kwargs)
        self.distance_scaling = self.bins

    def extract_features(self, dataset):
        res = []
        if self.is_parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for clustering_hist in executor.map(self.clustering_worker, [(G, self.bins) for G in dataset]):
                    res.append(clustering_hist)
        else:
            for g in dataset:
                clustering_hist = self.clustering_worker((g, self.bins))
                res.append(clustering_hist)

        res = [s1 / np.sum(s1) for s1 in res]
        return res

    def clustering_worker(self, param):
        G, bins = param
        clustering_coeffs_list = list(nx.clustering(G).values())
        hist, _ = np.histogram(
            clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
        return hist


class spectral(Descriptor):
    def __init__(self, *args, **kwargs):
        self.name = 'spectral'
        self.sigmas = [kwargs.get('sigma', 1.0)]
        self.distance_scaling = 1
        super().__init__(*args, **kwargs)

    def extract_features(self, dataset):
        res = []
        if self.is_parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for spectral_density in executor.map(self.spectral_worker, dataset):
                    res.append(spectral_density)
        else:
            for g in dataset:
                spectral_temp = self.spectral_worker(g)
                res.append(spectral_temp)
        return res

    def spectral_worker(self, G):
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
        spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
        spectral_pmf = spectral_pmf / spectral_pmf.sum()
        return spectral_pmf


def mmd_eval(graph_ref_list, graph_pred_list, methods):
    evaluators = []
    for (method, sigma, sigma_type) in methods:
        evaluators.append(eval(method)(sigma=sigma, sigma_type=sigma_type))

    results = {}
    for evaluator in evaluators:
        results[evaluator.name] = evaluator.evaluate(graph_ref_list, graph_pred_list)

    return results
