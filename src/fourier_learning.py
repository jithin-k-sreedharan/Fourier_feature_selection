"""
This is the NEWER version, that's in use from NeurIPS rebuttal period. The unusupervised algorithm
is changed in the this version to accommodate Mohsen's additions.

To Do: In the unsupervised algorithm, I still need to add the pre-processing (preranking and shuffle)
Mohsen used in his code.

Implementation of unuspervised and supervised Fourier feature selection algorithms
Algorithm 1 and Algorithm 2 in the paper
"""

from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np
from itertools import chain, combinations
import sys
import compute_fourier_coeff_supervised
import compute_norms_features_unsupervised
import math


# Generates the set of all subsets with the size of each subset as maximum k
def powerset(iterable, k):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, k + 1))


class OptionsUnsupervisedFourierFS:
    def __init__(self, max_depth, cluster_sizes, selection_thresholds, norm_epsilon, shuffle, preranking):
        self.max_depth = max_depth
        self.cluster_sizes = cluster_sizes
        self.selection_thresholds = selection_thresholds # same as n_redundant_threshold
        self.norm_epsilon = norm_epsilon
        self.shuffle = shuffle
        self.preranking = preranking


def UnsupervisedFourierFS_helper(X_nmlzd, depth, input_features, options):
    X_nmlzd_depth = X_nmlzd[:, input_features]
    d = len(input_features)
    n_clusters = math.ceil(d / options.cluster_sizes[depth])
    if n_clusters == 0:
        print("Error : n_clusters is zero!")
        sys.exit(2)
    clusters = np.linspace(0, d, n_clusters + 1, dtype=np.int)

    nonredundant_Features = []
    for i in range(1, len(clusters)):
        features_cluster = np.arange(clusters[i - 1], clusters[i])
        X_cluster = X_nmlzd_depth[:, features_cluster]
        sel_feats_norm2 = compute_norms_features_unsupervised.estimate_A(X_cluster,
                                                                         depth+1,
                                                                         options.norm_epsilon[depth])
        # import pdb; pdb.set_trace()
        sel_feats_norm2 = np.array(sel_feats_norm2)
        sel_feats_norm2 = sel_feats_norm2 ** 2
        Sorted_Feature_Indx = (-sel_feats_norm2).argsort()
        sel_feats_norm2_sorted = sel_feats_norm2[Sorted_Feature_Indx]
        cum_orthogonalization = np.cumsum(sel_feats_norm2_sorted) / sum(sel_feats_norm2_sorted)
        nonredundant_Orth = sum(cum_orthogonalization < options.selection_thresholds[depth])
        sel_feats_indices_local = Sorted_Feature_Indx[:nonredundant_Orth]

        nonredundant_Features.extend(features_cluster[sel_feats_indices_local])
    return nonredundant_Features


def UnsupervisedFourierFS(X, options):
    '''
    The main function for unsupervised Fourier feature selection algorithm (UFFS)
    Arguments:
        X: the input data with columns as features and rows correspond to data samples
        mean_emp: vector of empirical mean of each features
        std_emp: vector of empirical standard deviation of each features
        output_all_depth: if it's set to false, only output UFFS selected features for t=3 
                        Otherwise, output selected features for t=1, t=2, and t=3
    '''
    # mask = (np.std(X, ddof=1, axis=0) > 1e-5)
    # orig_features = np.arange(X.shape[1])
    # valid_features = orig_features[mask]
    # X = X[:, valid_features]

    mean_emp = np.mean(X, axis=0)
    std_emp = np.std(X, ddof=1, axis=0)

    X_nmlzd = (X - mean_emp) / std_emp
    n_X, d = X_nmlzd.shape

    input_features = np.arange(d)
    for depth in range(options.max_depth):
        print("depth: ", depth)
        nonredundant_features_indx = \
            UnsupervisedFourierFS_helper(X_nmlzd, depth, input_features, options)
        input_features = input_features[nonredundant_features_indx]
        print(f"No. of selected features in UnsupervisedFourierFS, depth {depth}: {len(input_features)}")
    nonredundant_features = input_features
    # return orig_features[mask][nonredundant_features]
    return nonredundant_features


'''
The main class for supervised Fourier feature selection algorithm (SFFS)
'''
class SupervisedFourierFS(BaseEstimator, ClassifierMixin):
    '''
    Arguments:
        k: no of features to be outputted by the feature selection algorithm
        mean_emp: vector of empirical mean of each features
        std_emp: vector of empirical standard deviation of each features
        approx: approximation technique in the implementation.
                It can be "none", depth-based or greedy-based.
        depth: t in Algorithm 2
    '''
    def __init__(self, k, mean_emp, std_emp, approx=None, depth= None):
        self.k = k
        self.mean_emp = mean_emp
        self.std_emp = std_emp
        self.approx = approx
        self.depth = depth
        self.f_bar_S_dict = {}

    '''
    Arguments:
        X_train: input data
        y_train: output data
    '''
    def fit(self, X_train, y_train):
        _, d = X_train.shape

        X_1 = (X_train - self.mean_emp) / self.std_emp

        # Exhaustive search
        if self.approx == "none":
            # build dictionary
            subsets_J = powerset(range(d), self.k)
            for S in subsets_J:
                _ = self.f_bar_est(S, X_1, y_train)

            optimal_norm = float('-inf')
            combinations_d_k = combinations(range(d), self.k)
            for J in combinations_d_k:
                norm2_J = self.f_bar_est_norm2(J, X_1, y_train)
                if optimal_norm < norm2_J:
                    self.J_bar = J
                    optimal_norm = norm2_J

        elif self.approx == "depth_based":
            self.f_bar_S_dict = compute_fourier_coeff_supervised.f_bar_est(X_1, y_train, d, self.depth)
            assert len(self.f_bar_S_dict) != 0, 'Building of dictionary f_bar_S_dict failed'
            combinations_d_k = powerset(range(d), self.depth)
            norm2_J_list = []
            for J in combinations_d_k:
                norm2_J = self.f_bar_est_norm2(J, X_train, y_train)
                norm2_J_list.append((norm2_J, J))
            norm2_J_list.sort(reverse=True)

            J_bar_temp = np.array(norm2_J_list[0][1], dtype=np.int)
            ii = 1
            while len(J_bar_temp) < self.k:
                J_bar_temp = np.hstack((J_bar_temp, norm2_J_list[ii][1]))
                _, idx = np.unique(J_bar_temp, return_index=True)
                J_bar_temp = J_bar_temp[np.sort(idx)]
                ii += 1
            if len(J_bar_temp) > self.k:
                J_bar_temp = J_bar_temp[:self.k]
            self.J_bar = J_bar_temp

        elif self.approx == "greedy":
            self.J_bar = []
            attr_list = list(range(d))
            J_bar_ind_temp = None
            for i in range(self.k):
                if J_bar_ind_temp is not None:
                    attr_list.remove(J_bar_ind_temp)
                optimal_norm = float('-inf')
                for J_ind in attr_list:
                    J = self.J_bar.copy()
                    J.append(J_ind)
                    norm2_J = self.f_bar_est_norm2(J, X_train, y_train)
                    if optimal_norm < norm2_J:
                        J_bar_ind_temp = J_ind
                        optimal_norm = norm2_J
                self.J_bar.append(J_bar_ind_temp)

        return self.J_bar

    def f_bar_est(self, S, X, y):
        if S in self.f_bar_S_dict:
            return self.f_bar_S_dict[S]
        else:
            s = 0
            X_len = len(X)
            X_1 = (X - self.mean_emp) / self.std_emp
            S_len = len(S)
            for i in range(X_len):
                p = 1
                for j in range(S_len):
                    p *= X_1[i, S[j]]
                s += y[i] * p
            s = s / X_len
            self.f_bar_S_dict[S] = s
            return s

    def f_bar_est_norm2(self, J, X, y):
        s = 0
        subsets_J = powerset(J, len(J))
        for S in subsets_J:
            s += (self.f_bar_est(S, X, y)) ** 2
        return s
