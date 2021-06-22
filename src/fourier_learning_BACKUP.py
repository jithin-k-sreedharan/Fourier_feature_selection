'''
This is the OLDER version, that's in use before NeurIPS rebuttal. The unusupervised algorithm
is changed in the later version to accommodate Mohsen's additions

Implementation of unuspervised and supervised Fourier feature selection algorithms
Algorithm 1 and Algorithm 2 in the paper
'''

from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np
from itertools import chain, combinations
import sys
import compute_fourier_coeff_supervised
import compute_norms_features_unsupervised


# Generates the set of all subsets with the size of each subset as maximum k
def powerset(iterable, k):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, k + 1))


'''
Helper class for unsupervised feature selection
An instance is created for each level of depth (t) in the UFFS algorithm
'''
class orth_class():
    def __init__(self, X_nmlzd, depth, select_features = []):
        self.depth = depth
        self.select_features = select_features
        self.X_nmlzd = X_nmlzd
        if len(select_features) > 0:
            self.X_nmlzd = self.X_nmlzd[:, select_features]
        self.n_X, self.d = self.X_nmlzd.shape

    '''
    Arguments:
        clusters: split the features into multiple non-overlapping clusters. 
                  Useful when we are not able to run UFFS on the entire dataset  
        epsilon: epsilon is used as mentioned in Algorithm 1 in the paper
    '''
    def fit(self, clusters=[], epsilon=0.001):
        n_redundant_threshold = 0.95

        if len(clusters) == 0:
            clusters = [0, self.d]
        all_feats_norm2 = []
        for i in range(1, len(clusters)):
            features_cluster = np.arange(clusters[i - 1], clusters[i])
            X_cluster = self.X_nmlzd[:, features_cluster]
            # all_feats_norm2.extend(compute_B_matrix.estimate_A(X_cluster, self.depth, epsilon))
            all_feats_norm2.extend(
                compute_norms_features_unsupervised.estimate_A(X_cluster, self.depth, epsilon))

        self.all_feats_norm2 = np.array(all_feats_norm2)
        self.all_feats_norm2 = self.all_feats_norm2 ** 2
        Sorted_Feature_Indx = (-self.all_feats_norm2).argsort()
        all_feats_norm2_sorted = self.all_feats_norm2[Sorted_Feature_Indx]
        cum_orthogonalization = np.cumsum(all_feats_norm2_sorted) / sum(all_feats_norm2_sorted)
        nonredundant_Orth = sum(cum_orthogonalization < n_redundant_threshold)
        self.nonredundant_Features = Sorted_Feature_Indx[:nonredundant_Orth]
        if len(self.select_features) > 0:
            self.nonredundant_Features = self.select_features[self.nonredundant_Features]


'''
The main function for unsupervised Fourier feature selection algorithm (UFFS)
Arguments:
    X: the input data with columns as features and rows correspond to data samples
    mean_emp: vector of empirical mean of each features
    std_emp: vector of empirical standard deviation of each features
    output_all_depth: if it's set to false, only output UFFS selected features for t=3 
                      Otherwise, output selected features for t=1, t=2, and t=3
Version before NeurIPS response
'''
def UnsupervisedFourierFS(X, mean_emp, std_emp, output_all_depth=False):
    X_nmlzd = (X - mean_emp) / std_emp
    n_X, d = X_nmlzd.shape

    max_depth = 3
    for depth in range(1, max_depth+1):
        print("depth: ", depth)
        if depth == 1:
            orth1 = orth_class(X_nmlzd, 1)
            orth1.fit()
        elif depth == 2:
            if d < 50:
                orth2 = orth_class(X_nmlzd, 2)
                orth2.fit()
            elif len(orth1.nonredundant_Features) < 50:
                input_features = orth1.nonredundant_Features
                orth2 = orth_class(X_nmlzd, 2, input_features)
                orth2.fit()
            else:
                n_clusters = len(orth1.nonredundant_Features) // 50
                clusters = np.linspace(0, len(orth1.nonredundant_Features), n_clusters + 1, dtype=np.int)
                input_features = orth1.nonredundant_Features
                orth2 = orth_class(X_nmlzd, 2, input_features)
                orth2.fit(clusters)
        elif depth == 3:
            if d < 31:
                orth3 = orth_class(X_nmlzd, 3)
                orth3.fit()
            elif len(orth2.nonredundant_Features) < 31:
                input_features = orth2.nonredundant_Features
                orth3 = orth_class(X_nmlzd, 3, input_features)
                orth3.fit()
            else:
                n_clusters = len(orth2.nonredundant_Features) // 31
                clusters = np.linspace(0, len(orth2.nonredundant_Features), n_clusters+1, dtype=np.int)
                input_features = orth2.nonredundant_Features
                orth3 = orth_class(X_nmlzd, 3, input_features)
                orth3.fit(clusters)
            print("len(orth3.nonredundant_Features)", len(orth3.nonredundant_Features))
        else:
            print("Too much depth")
            sys.exit(2)
    if output_all_depth:
        return (orth1.nonredundant_Features, orth2.nonredundant_Features, orth3.nonredundant_Features)
    else:
        return orth3.nonredundant_Features

class OptionsUnsupervisedFourierFS():
    def __init__(self, max_depth, cluster_sizes, selection_thresholds, norm_epsilon, shuffle, preranking):
        self.max_depth = max_depth
        self.cluster_sizes = cluster_sizes
        self.selection_thresholds = selection_thresholds
        self.norm_epsilon = norm_epsilon
        self.shuffle = shuffle
        self.preranking = preranking

'''
The main function for unsupervised Fourier feature selection algorithm (UFFS)
Arguments:
    X: the input data with columns as features and rows correspond to data samples
    mean_emp: vector of empirical mean of each features
    std_emp: vector of empirical standard deviation of each features
    output_all_depth: if it's set to false, only output UFFS selected features for t=3 
                      Otherwise, output selected features for t=1, t=2, and t=3
'''
def UnsupervisedFourierFS_Workspace(X, mean_emp, std_emp, options, output_all_depth=False):
    X_nmlzd = (X - mean_emp) / std_emp
    n_X, d = X_nmlzd.shape

    input_features = range(0, d)
    for depth in range(1, options.max_depth+1):
        print("depth: ", depth)

    #     n_cluster = len(input_features) // options.cluster_sizes[depth]
    #     orth_class(X_nmlzd, depth, input_features)
    #
    #     Nonredundant_Features = FUFS(depth, n_cluster, X_train(:, input_features), ...
    #     SelectionThresholds(depth), NormEpsilon(depth), Shuffle
    #     {depth}, Preranking
    #     {depth});
    #     input_features = input_features(Nonredundant_Features);
    # end
    # Nonredundant_Features = input_features;

    if depth == 1:
        orth1 = orth_class(X_nmlzd, 1)
        orth1.fit()
    elif depth == 2:
        if d < 50:
            orth2 = orth_class(X_nmlzd, 2)
            orth2.fit()
        elif len(orth1.nonredundant_Features) < 50:
            input_features = orth1.nonredundant_Features
            orth2 = orth_class(X_nmlzd, 2, input_features)
            orth2.fit()
        else:
            n_clusters = len(orth1.nonredundant_Features) // 50
            clusters = np.linspace(0, len(orth1.nonredundant_Features), n_clusters + 1, dtype=np.int)
            input_features = orth1.nonredundant_Features
            orth2 = orth_class(X_nmlzd, 2, input_features)
            orth2.fit(clusters)
    elif depth == 3:
        if d < 31:
            orth3 = orth_class(X_nmlzd, 3)
            orth3.fit()
        elif len(orth2.nonredundant_Features) < 31:
            input_features = orth2.nonredundant_Features
            orth3 = orth_class(X_nmlzd, 3, input_features)
            orth3.fit()
        else:
            n_clusters = len(orth2.nonredundant_Features) // 31
            clusters = np.linspace(0, len(orth2.nonredundant_Features), n_clusters+1, dtype=np.int)
            input_features = orth2.nonredundant_Features
            orth3 = orth_class(X_nmlzd, 3, input_features)
            orth3.fit(clusters)
        print("len(orth3.nonredundant_Features)", len(orth3.nonredundant_Features))
    else:
        print("Too much depth")
        sys.exit(2)
    if output_all_depth:
        return (orth1.nonredundant_Features, orth2.nonredundant_Features, orth3.nonredundant_Features)
    else:
        return orth3.nonredundant_Features


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
