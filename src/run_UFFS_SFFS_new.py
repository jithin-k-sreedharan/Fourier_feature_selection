"""
Syntax: python run_UFFS_SFFS --data <data_name>
<data_name> and the preprocessing on the data need to be added in the load_data function

Edit the load_data(data_name) function to load a dataset named data_name

Meaning of the constants, that's set after loading libraries
    NO_RUNS: no. of cross validation folds
    CLF_NAME: name of the classifier
    UFFS_FOLDS: Is unsupervised Fourier feature selection running for all the folds separately?
    UFFS_SINGLE: Is unsupervised Fourier feature selection running the entire dataset?
    If both UFFS_FOLDS and UFFS_SINGLE are False, then there is no UFFS, only SFFS
    PARALLELIZE_CV: Use parallelized cross validation, except for mRMR algorithm
"""

import sys
import numpy as np
import pandas as pd
import scipy
import scipy.io
from argparse import ArgumentParser

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif

import matplotlib.pyplot as plt
import matplotlib
import subprocess
from multiprocessing import Pool

import fourier_learning
import time
from datetime import datetime
import pickle

from skfeature.function.similarity_based import reliefF
from skfeature.function.sparse_learning_based import MCFS
from skfeature.utility import construct_W
from skfeature.utility.sparse_learning import feature_ranking
from skfeature.function.similarity_based import lap_score
from skfeature.function.sparse_learning_based import UDFS
from skfeature.function.sparse_learning_based import NDFS

from tqdm import tqdm

# Constants ===================================================================
NO_RUNS = 5
CLF_NAME = "kernel SVM with RBF"  # {"kernel SVM with RBF", "random forest"}

UNSUPERVISED_FEAT_SEL = False
SUPERVISED_FEAT_SEL = True

# Existing supervised algorithms
FEAT_SEL_ALGS_SUPERVISED = {
    'FL-Depth-2': False,
    'FL-Depth-1': False,
    'FL': False,  # Fourier Learning exhaustive search
    'SFFS': False,
    'SFFS (exhaustive)': False,  # Fourier Learning exhaustive search
    'UFFS + SFFS (t=2)': True,
    'UFFS + SFFS (t=1)': True,
    'UFFS + SFFS (exhaustive)': False,  # Fourier Learning exhaustive search
    'CCM': True,
    'ReliefF': True,
    'mRMR': True,
    'MI': True,
    'RFE': False,
    'F-Value': True
}

# Existing unsupervised algorithms
FEAT_SEL_ALGS_UNSUPERVISED = {
    'UFFS': False,
    'NO_FS': False,
    'NDFS': False,
    'UDFS': False,
    'MCFS': False,
    'LS': False
}

UFFS_CALCN_CV = True
PARALLELIZE_CV = True
UFFS_K = 200
UFFS_FOLDS = False
UFFS_SINGLE = True

FIGSIZE_SCALE_REQD = 0.5

# =============================================================================
if FEAT_SEL_ALGS_SUPERVISED['CCM']:
    sys.path.append('../lib/CCM/core')
    import ccm_v1 as ccm

# DON'T CHANGE IT
ORTHOGONALIZE = False
N_UNIQUE_CLASSES_Y = 20

FEAT_SEL_ALGS_FN_LABEL = {
    "FL": "fourier-learning",
    "FL-Depth-2": "fourier-learning_depth_2",
    "FL-Depth-1": "fourier-learning_depth_1",
    "UFFS + SFFS (t=2)": "UFFS + SFFS (t=2)",
    "UFFS + SFFS (t=1)": "UFFS + SFFS (t=1)",
    # Existing supervised algorithms
    "CCM": "ccm",
    "ReliefF": "relieff",
    "mRMR": "mrmr",
    "RFE": "rfe",
    "MI": "mi",
    "F-Value": "fval",
    # Existing unsupervised algorithms
    "NDFS": True,
    "UDFS": True,
    "MCFS": True,
    "LS": True
}

UNSUPERVISED_FEAT_SEL_ALGS_FN_LABEL = {
    'NDFS': True,
    'UDFS': True,
    'MCFS': True,
    "LAP_SCORE": "lap_score"
}

# Figure settings =============================================================

def figsize(scale):
    fig_width_pt = 503.295     # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27   # Convert pt to inch
    # Aesthetic ratio (you could change this)
    golden_mean = (np.sqrt(5.0)-1.0)/2.0
    fig_width = fig_width_pt*inches_per_pt*scale  # width in inches
    fig_height = fig_width*golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    # blank entries should cause plots to inherit fonts from the document
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 8,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
}
matplotlib.rcParams.update(pgf_with_latex)

cust_color = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
              "#ff7f00", "#ffff33", "#a65628", "#f781bf"]
matplotlib.rcParams['savefig.dpi'] = 125
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath,amssymb,amsfonts}"
# =============================================================================

# function to load data. It also does some initial processing:
# Whenever using a new dataset, add an entry here.

def load_data(data_name):
    if data_name == 'wine':
        X = np.loadtxt("../data/wine/data.txt", dtype=np.float)
        y = np.loadtxt("../data/wine/y.txt", dtype=np.float)
        type_y = 'binary'
    elif data_name == 'HAPT':
        temp_data = scipy.io.loadmat("../data/HAPT/HAPT_Data_Set.mat")
        X, y = temp_data['X'], temp_data['Y'].squeeze()
        type_y = 'categorical'
    elif data_name == 'Isolet':
        temp_data = scipy.io.loadmat(
            "../lib/scikit-feature/skfeature/data/Isolet.mat")
        X, y = temp_data['X'], temp_data['Y'].squeeze()
        type_y = 'categorical'
    elif data_name == 'USPS':
        temp_data = scipy.io.loadmat(
            "../lib/scikit-feature/skfeature/data/USPS.mat")
        X, y = temp_data['X'], temp_data['Y'].squeeze()
        type_y = 'categorical'
    elif data_name == 'vowel':
        X = np.loadtxt("../data/vowel/data.txt", dtype=np.float)
        y = np.loadtxt("../data/vowel/y.txt", dtype=np.float)
        type_y = 'categorical'
        encoded = np.arange(11)
        y = y.dot(encoded).astype(int)
        shuffle_index = np.random.permutation(X.shape[0])
        X, y = X[shuffle_index], y[shuffle_index]
    elif data_name == 'glass':
        column_names = ['ID', 'refractive_index', 'sodium', 'magnesium', 'aluminum',
                        'silicon', 'potassium', 'calcium', 'barium', 'iron', 'type_glass']
        df = pd.read_csv("../data/glass/glass.txt",
                         names=column_names, header=None)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        type_y = 'categorical'
    elif data_name == 'heart-disease-UCI':
        # X.shape = 303, 13
        data = np.loadtxt('../data/heart-disease-UCI.csv',
                          delimiter=',', skiprows=1, dtype=np.float)
        X = data[:, :-1]
        y = data[:, -1]
        y[y == 0] = -1
        type_y = 'binary'
    elif data_name == 'MADELON NIPS-2003':
        X = np.loadtxt("../data/madelon/madelon.csv", delimiter=",",
                       usecols=np.arange(500), skiprows=1, dtype=np.int)
        y = np.loadtxt("../data/madelon/madelon.csv",
                       delimiter=",", usecols=500, skiprows=1, dtype=np.int)
        y[y == 1] = -1
        y[y == 2] = 1
        type_y = 'binary'
    elif data_name == 'SYN1':
        temp_data = scipy.io.loadmat('../data/temp/SYN1.mat')
        X, y = temp_data['X'], np.squeeze(temp_data['Y'])
        type_y = 'binary'
    elif data_name == 'SYN2':
        temp_data = scipy.io.loadmat('../data/temp/SYN2.mat')
        X, y = temp_data['X'], np.squeeze(temp_data['Y'])
        type_y = 'binary'
    elif data_name == 'SYN3':
        temp_data = scipy.io.loadmat('../data/temp/SYN3.mat')
        X, y = temp_data['X'], np.squeeze(temp_data['Y'])
        type_y = 'binary'
    elif data_name.startswith('lesong-icml-data'):
        folder = "../data/Le_Song_icml_data/{0}/".format(
            data_name.split('-')[2])
        X = np.loadtxt(folder+"data.txt", dtype=np.float)
        y = np.loadtxt(folder+"y.txt", dtype=np.int)
        if len(y.shape) == 2:
            _, d_y = y.shape
            if d_y > 1:
                type_y = 'categorical'
                encoded = np.arange(d_y)
                y = y.dot(encoded).astype(int)
            else:
                type_y = 'binary'
        elif len(y.shape) == 1:
            type_y = 'binary'
    elif data_name.startswith('skfeature_'):
        file = "../lib/scikit-feature/skfeature/data/{0}.mat".format(
            data_name.split('_')[1])
        temp_data = scipy.io.loadmat(file)
        X, y = temp_data['X'], temp_data['Y'].squeeze()
        if len(np.unique(y)) > 2:
            type_y = 'categorical'
        else:
            type_y = 'binary'
    elif data_name == "Erlang-best":
        # Random boolean function with Fourier spectrum closee to an Erlang distribution
        # Given complete space
        X = np.loadtxt("../data/complete_f/X_d12_p03_4.txt", dtype=np.int)
        y = np.loadtxt("../data/complete_f/y_d12_p03_4.txt", dtype=np.int)
        type_y = 'binary'
    elif data_name in ["data6", "data13"]:
        file = "../data/data_Mohsen/{0}.mat".format(data_name)
        temp_data = scipy.io.loadmat(file)
        X, y = temp_data['X'], temp_data['Y'].squeeze()
        if len(np.unique(y)) > 2:
            type_y = 'categorical'
        else:
            type_y = 'binary'
    else:
        print("No valid data_name entered")
        sys.exit(2)
    if type_y == 'binary' or type_y == 'categorical':
        if y.dtype != np.int:
            y = y.astype(np.int, copy=False)

    return X, y, type_y


# Helper function to call supervised fourier selection
def fourier_feature_selection(X, y, k, approx="depth_based", depth=2):
    mean_emp = np.mean(X, axis=0)
    std_emp = np.std(X, ddof=1, axis=0)
    fourier_featsel = fourier_learning.SupervisedFourierFS(
        k, mean_emp, std_emp, approx, depth)
    feats_selected = fourier_featsel.fit(X, y)
    return feats_selected


# Helper function to call the classifier using only the selected features
def clf_score_with_feature_selection(X_train, y_train, X_test, y_test, clf, feats_selected):
    X_sel_train = X_train[:, feats_selected]
    X_sel_test = X_test[:, feats_selected]

    clf.fit(X_sel_train, y_train)
    y_sel_pred = clf.predict(X_sel_test)
    return accuracy_score(y_test, y_sel_pred)


# Condiational co-variance feature selection
def ccm_feature_selection(X, y, type_y):
    epsilon = 0.001
    _, d = X.shape
    if d <= 100:
        num_features = math.ceil(d / 5)
    else:
        num_features = 100
    rank_ccm = ccm.ccm(X, y, num_features, type_y, epsilon,
                       iterations=100, verbose=False)
    all_feats_selected = np.argsort(rank_ccm)
    return all_feats_selected


def accuracy_multiple_train_test(X, y, type_y, xaxis, feat_selection, clf, no_runs,
                                 sel_features_UFFS_X=[]):
    '''
    Feature selection with cross validation: NOT a parallel implementation

    Arguments:
        X: the entire input data with columns as features and rows as samples
        y: the entire output labels
        type_y: type of y - binary, categorical or real, for CCM feature selection
        feat_selection: feature selection algorithm name
        clf: classifier function
        no_runs: Number of folds of cross-validation
        sel_features_UFFS_X: set of all features selected by unsupervised Fourier feature selection 
                            if we use UFFS_CALCN_CV = False 
    '''
    score_temp = np.zeros(len(xaxis))
    kf = KFold(n_splits=no_runs, shuffle=True, random_state=42)

    for train_index, test_index in tqdm(kf.split(X)):
        X_train, X_test, y_train, y_test = X[train_index, :], X[test_index, :], y[train_index], \
            y[test_index]

        # Variance threshold --------------------------------------------------
        _, d = X_train.shape
        mask = (np.std(X_train, ddof=1, axis=0) > 1e-5)
        valid_features = np.arange(d)
        valid_features = valid_features[mask]

        if ORTHOGONALIZE:
            if UFFS_CALCN_CV:
                X_train = X_train[:, valid_features]
                X_test = X_test[:, valid_features]
                if feat_selection in ["fourier-learning_depth_1", "fourier-learning_depth_2"]:
                    mean_emp = np.mean(X_train, axis=0)
                    std_emp = np.std(X_train, ddof=1, axis=0)
                    sel_features_UFFS = fourier_learning.UnsupervisedFourierFS(X_train)
                    X_train = X_train[:, sel_features_UFFS]
                    X_test = X_test[:, sel_features_UFFS]
            else:
                if feat_selection in ["fourier-learning_depth_1", "fourier-learning_depth_2"]:
                    valid_features = list(
                        set(valid_features).intersection(sel_features_UFFS_X))
                X_train = X_train[:, valid_features]
                X_test = X_test[:, valid_features]
        else:
            X_train = X_train[:, valid_features]
            X_test = X_test[:, valid_features]
        # ---------------------------------------------------------------------
        _, d = X_train.shape

        if feat_selection == "ccm":
            all_feats_selected = ccm_feature_selection(
                X_train, y_train, type_y)
        elif feat_selection == "relieff":
            score_reliefF_temp = reliefF.reliefF(X_train, y_train)
            # rank features in descending order according to score
            all_feats_selected = reliefF.feature_ranking(score_reliefF_temp)[
                :d]
        elif feat_selection == "mrmr":
            # Prepare data
            y_train = y_train.reshape(-1, 1)
            a = ["class"] + [str(i) for i in range(0, d)]
            df_mRMR = pd.DataFrame(data=np.concatenate(
                (y_train, X_train), axis=1), columns=a)
            df_mRMR.to_csv('../data/temp_mRMR_data.csv', index=False)
            y_train = y_train.ravel()

            # Call the C++ executable
            command_mRMR = '../lib/mrmr_c_Peng/./mrmr -i ../data/temp_mRMR_data.csv -n {0}'.format(
                d)
            out_code = subprocess.call([command_mRMR], shell=True)
            if out_code != 0:
                print("something wrong")
                sys.exit(2)
            all_feats_selected = np.loadtxt('out_temp.txt', dtype=np.int)
            # ==================================================================
        elif feat_selection == "mi":
            feat_sel_alg = SelectKBest(score_func=mutual_info_classif, k=d)
            feat_sel_alg.fit(X_train, y_train)
            all_feats_selected = feat_sel_alg.get_support(indices=True)
        elif feat_selection == "fval":
            feat_sel_alg = SelectKBest(score_func=f_classif, k=d)
            feat_sel_alg.fit(X_train, y_train)
            all_feats_selected = feat_sel_alg.get_support(indices=True)
        elif feat_selection == "fourier-learning_depth_2":
            all_feats_selected = fourier_feature_selection(X_train, y_train, d,
                                                           approx="depth_based", depth=2)
        elif feat_selection == "fourier-learning_depth_1":
            all_feats_selected = fourier_feature_selection(X_train, y_train, d,
                                                           approx="depth_based", depth=1)
        elif feat_selection == "fourier-learning":
            all_feats_selected = fourier_feature_selection(
                X_train, y_train, d, approx="none")
        elif feat_selection == "rfe":
            clf_rfe = LinearSVC(loss="squared_hinge",
                                penalty="l1", dual=False, max_iter=2000)
            feat_sel_alg = RFE(clf_rfe, n_features_to_select=d, step=1)
            X_train_1 = StandardScaler().fit_transform(X_train)
            feat_sel_alg.fit(X_train_1, y_train)
            all_feats_selected = feat_sel_alg.get_support(indices=True)

        for i, k in enumerate(xaxis):
            if k > d:
                break
            feats_selected = all_feats_selected[:k]
            score_temp[i] += clf_score_with_feature_selection(
                X_train, y_train, X_test, y_test, clf, feats_selected)

    score_temp /= no_runs
    score_temp = score_temp[score_temp > 0]
    return score_temp


'''
Feature selection with single fold of cross validation: 
for parallel implementation of cross validation

Arguments:
    X: input data of one fold with columns as features and rows as samples
    y: output lables of one fold
    type_y: type of y - binary, categorical or real, for CCM feature selection
    feat_selection: feature selection algorithm name
    clf: classifier function
    partitions: the set of all cross-validation partitions
    k: partition index
    sel_features_UFFS_X: set of all features selected by unsupervised Fourier feature selection 
                         if we use UFFS_SINGLE = True 
'''


def cv_sample_single_run(X, y, type_y, xaxis, feat_selection, clf, partitions, k,
                         sel_features_UFFS_X=[]):
    train_indices, test_indices = partitions[k]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    # Variance threshold --------------------------------------------------
    _, d = X_train.shape
    mask = (np.std(X_train, ddof=1, axis=0) > 1e-5)
    valid_features = np.arange(d)
    valid_features = valid_features[mask]

    if UFFS_FOLDS:
        X_train = X_train[:, valid_features]
        X_test = X_test[:, valid_features]

        if feat_selection in ["fourier-learning_depth_1", "fourier-learning_depth_2", "UFFS + SFFS (t=2)", "UFFS + SFFS (t=1)"]:
            mean_emp = np.mean(X_train, axis=0)
            std_emp = np.std(X_train, ddof=1, axis=0)
            sel_features_UFFS = fourier_learning.UnsupervisedFourierFS(
                X_train, mean_emp, std_emp)
            X_train = X_train[:, sel_features_UFFS]
            X_test = X_test[:, sel_features_UFFS]
    elif UFFS_SINGLE:
        if feat_selection in ["fourier-learning_depth_1", "fourier-learning_depth_2", "UFFS + SFFS (t=2)", "UFFS + SFFS (t=1)"]:
            valid_features = list(
                set(valid_features).intersection(sel_features_UFFS_X))
        X_train = X_train[:, valid_features]
        X_test = X_test[:, valid_features]
    else:
        X_train = X_train[:, valid_features]
        X_test = X_test[:, valid_features]
    # ---------------------------------------------------------------------
    _, d = X_train.shape

    if feat_selection == "ccm":
        all_feats_selected = ccm_feature_selection(X_train, y_train, type_y)
    elif feat_selection == "relieff":
        score_reliefF_temp = reliefF.reliefF(X_train, y_train)
        # rank features in descending order according to score
        all_feats_selected = reliefF.feature_ranking(score_reliefF_temp)[:d]
    elif feat_selection == "mrmr":
        # Prepare data
        y_train = y_train.reshape(-1, 1)
        a = ["class"] + [str(i) for i in range(0, d)]
        df_mRMR = pd.DataFrame(data=np.concatenate(
            (y_train, X_train), axis=1), columns=a)
        df_mRMR.to_csv('../data/temp_mRMR_data.csv', index=False)
        y_train = y_train.ravel()

        # Call the C++ executable
        command_mRMR = '../lib/mrmr_c_Peng/./mrmr -i ../data/temp_mRMR_data.csv -n {0}'.format(
            d)
        out_code = subprocess.call([command_mRMR], shell=True)
        if out_code != 0:
            print("something wrong")
            sys.exit(2)
        all_feats_selected = np.loadtxt('out_temp.txt', dtype=np.int)
        # ==================================================================
    elif feat_selection == "mi":
        feat_sel_alg = SelectKBest(score_func=mutual_info_classif, k=d)
        feat_sel_alg.fit(X_train, y_train)
        all_feats_selected = feat_sel_alg.get_support(indices=True)
    elif feat_selection == "fval":
        feat_sel_alg = SelectKBest(score_func=f_classif, k=d)
        feat_sel_alg.fit(X_train, y_train)
        all_feats_selected = feat_sel_alg.get_support(indices=True)
    elif feat_selection == "UFFS + SFFS (t=2)":
        all_feats_selected = fourier_feature_selection(X_train, y_train, d,
                                                       approx="depth_based", depth=2)
    elif feat_selection == "UFFS + SFFS (t=1)":
        all_feats_selected = fourier_feature_selection(X_train, y_train, d,
                                                       approx="depth_based", depth=1)
    elif feat_selection == "fourier-learning":
        all_feats_selected = fourier_feature_selection(
            X_train, y_train, d, approx="none")
    elif feat_selection == "rfe":
        clf_rfe = LinearSVC(loss="squared_hinge",
                            penalty="l1", dual=False, max_iter=2000)
        feat_sel_alg = RFE(clf_rfe, n_features_to_select=d, step=1)
        X_train_1 = StandardScaler().fit_transform(X_train)
        feat_sel_alg.fit(X_train_1, y_train)
        all_feats_selected = feat_sel_alg.get_support(indices=True)
    else:
        print("none selected")

    score_temp = np.zeros(len(xaxis))
    for i, k in enumerate(xaxis):
        if k > d:
            break
        feats_selected = all_feats_selected[:k]
        score_temp[i] += clf_score_with_feature_selection(
            X_train, y_train, X_test, y_test, clf, feats_selected)

    return score_temp


def cv_sample_single_run_unsupervised(X, y, type_y, xaxis, feat_selection, clf, partitions, fold_i,
                                      sel_features_UFFS_X=[]):
    train_indices, test_indices = partitions[fold_i]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    # Variance threshold --------------------------------------------------
    _, d = X_train.shape
    mask = (np.std(X_train, ddof=1, axis=0) > 1e-5)
    valid_features = np.arange(d)
    valid_features = valid_features[mask]
    X_train = X_train[:, valid_features]
    X_test = X_test[:, valid_features]
    # ---------------------------------------------------------------------
    _, d = X_train.shape

    if feat_selection == "UFFS":
        mean_emp = np.mean(X_train, axis=0)
        std_emp = np.std(X_train, ddof=1, axis=0)
        all_feats_selected = fourier_learning.UnsupervisedFourierFS(
            X_train, mean_emp, std_emp)
        # UFFS_K.append(len(all_feats_selected))
        # print("UFFS_K: ", UFFS_K)
    elif feat_selection == "MCFS":
        # construct affinity matrix
        kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode":
                  "heatKernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X_train, **kwargs)
        # obtain the feature weight matrix
        Weight = MCFS.mcfs(X_train, n_selected_features=d, W=W,
                           n_clusters=N_UNIQUE_CLASSES_Y)
        # sort the feature scores in an ascending order according to the feature scores
        all_feats_selected = MCFS.feature_ranking(Weight)[:d]
    elif feat_selection == "NDFS":
        # construct affinity matrix
        kwargs = {"metric": "euclidean", "neighborMode": "knn",
                  "weightMode": "heatKernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X_train, **kwargs)
        # obtain the feature weight matrix
        Weight = NDFS.ndfs(X_train, W=W, n_clusters=N_UNIQUE_CLASSES_Y)
        all_feats_selected = feature_ranking(Weight)[:d]
    elif feat_selection == "UDFS":
        # obtain the feature weight matrix
        Weight = UDFS.udfs(X_train, gamma=0.1, n_clusters=N_UNIQUE_CLASSES_Y)
        all_feats_selected = feature_ranking(Weight)[:d]
    elif feat_selection == "LS":
        # construct affinity matrix
        kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn",
                    "weight_mode": "heat_kernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X_train, **kwargs_W)
        # obtain the scores of features
        score = lap_score.lap_score(X_train, W=W)
        # sort the feature scores in an ascending order according to the feature scores
        all_feats_selected = lap_score.feature_ranking(score)[:d]

    if feat_selection == "NO_FS":
        score_temp = clf_score_with_feature_selection(
            X_train, y_train, X_test, y_test, clf, list(range(d)))
    elif feat_selection != "UFFS":
        score_temp = clf_score_with_feature_selection(X_train, y_train, X_test, y_test,
                                                      clf, all_feats_selected[:UFFS_K])
    else:
        score_temp = clf_score_with_feature_selection(
            X_train, y_train, X_test, y_test, clf, all_feats_selected)
    return score_temp


def plot_ccm_comparison(xaxis, data_name, scores, file_name=None):
    '''
    Plot the results
    arguments:
        xaxis: xaxis vector
        data_name: name of the dataset
        scores: dictionary with key as the feature selection algorithm name and value as the vector 
                of cross-validation scores for different k's
        file_nmae: save to a specific file, otherwise create a time-stamped file
    '''

    fig, ax = plt.subplots(figsize=figsize(FIGSIZE_SCALE_REQD))
    # ax = plt.axes(frameon=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.grid(alpha=1, linestyle='dotted')
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=True)
    ax.tick_params(axis='y', which='minor', left=False)

    Fourier_FS_length = min(len(scores["UFFS + SFFS (t=2)"]), len(scores["UFFS + SFFS (t=1)"]))
    # Fourier_FS_length = len(scores["UFFS + SFFS (t=2)"])
    for k, v in scores.items():
        plt.plot(xaxis[:Fourier_FS_length],
                 v[:Fourier_FS_length]*100, label=k, alpha=0.7)
        # plt.legend(loc='best')
        plt.xlabel(r"$k$")
        plt.ylabel("Mean accuracy")
    title_str = r"Dataset: {0}".format(data_name)
    title_str = title_str.replace('_', '\_')
    plt.title(title_str)

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center',
                    bbox_to_anchor=(0.45, -0.2), ncol=3)

    if file_name == None:
        timestr = time.strftime("%Y%m%d_%H%M%S")
        file_name = "comparison_feat_sel_{0}".format(timestr)
    plt.savefig('../results/{0}.pdf'.format(file_name),
                bbox_extra_artists=(lgd,), bbox_inches='tight', pad_inches=0.02)
    return file_name

# Function to initially set x-axis of the final figure


def xaxis_fn(d):
    if d <= 50:
        xaxis = list(range(1, min(20, d)+1))
    elif 50 < d <= 100:
        xaxis = list(range(5, 51, 5))
    elif d > 100:
        xaxis = list(range(10, 101, 10))
    return xaxis


def main(DATA_NAME):
    if CLF_NAME == 'kernel SVM with RBF':
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", C=1, gamma='auto'))
        ])
    elif CLF_NAME == 'random forest':
        clf = RandomForestClassifier(n_estimators=100)

    # Load data
    X, y, type_y = load_data(DATA_NAME)
    no_samples, d = X.shape
    if len(np.unique(y)) < no_samples//2:
        global N_UNIQUE_CLASSES_Y
        N_UNIQUE_CLASSES_Y = len(np.unique(y))

    # Set an initial x-axis
    xaxis = xaxis_fn(d)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y, shuffle=True)

    X = X_train.copy()
    y = y_train.copy()

    FEAT_SEL_ALGS = {}
    global ORTHOGONALIZE
    if UNSUPERVISED_FEAT_SEL:
        FEAT_SEL_ALGS.update(FEAT_SEL_ALGS_UNSUPERVISED)
        ORTHOGONALIZE = True
    elif SUPERVISED_FEAT_SEL:
        FEAT_SEL_ALGS.update(FEAT_SEL_ALGS_SUPERVISED)
        if FEAT_SEL_ALGS_SUPERVISED['UFFS + SFFS (t=2)'] \
                or FEAT_SEL_ALGS_SUPERVISED['UFFS + SFFS (t=1)'] \
                or FEAT_SEL_ALGS_SUPERVISED['UFFS + SFFS (exhaustive)']:
            ORTHOGONALIZE = True

    feat_sel_algs_chosen = [k for k, v in FEAT_SEL_ALGS.items() if v == True]

    sel_features_UFFS_X = []
    # if not PARALLELIZE_CV and ORTHOGONALIZE:
    if ORTHOGONALIZE:
        print("entered")
        _, d = X.shape
        mask = (np.std(X, ddof=1, axis=0) > 1e-5)
        valid_features = np.arange(d)
        valid_features = valid_features[mask]
        X = X[:, valid_features]

        mean_emp_X = np.mean(X, axis=0)
        std_emp_X = np.std(X, ddof=1, axis=0)
        UFFS_options = \
            fourier_learning.OptionsUnsupervisedFourierFS(max_depth=3,
                                                        cluster_sizes=[d, 50, 25],
                                                        selection_thresholds=[0.95,0.95,0.95],
                                                        norm_epsilon=[0.001, 0.001, 0.001],
                                                        shuffle=False,
                                                        preranking="non"
                                                        )
        sel_features_UFFS_X = fourier_learning.UnsupervisedFourierFS(X,UFFS_options)
        # sel_features_UFFS_X = fourier_learning.UnsupervisedFourierFS(X)

    if UNSUPERVISED_FEAT_SEL and PARALLELIZE_CV:
        scores_unsupervised = {}
        time_taken_algs_unsupervised = {}
        for alg in feat_sel_algs_chosen:
            start_time = datetime.now()
            kf = KFold(n_splits=NO_RUNS, shuffle=True, random_state=42)
            partitions = list(kf.split(X))
            pool = Pool()
            args = [(X, y, type_y, xaxis, alg, clf, partitions, k,
                     sel_features_UFFS_X) for k in range(NO_RUNS)]
            Accuracies = np.array(pool.starmap(
                cv_sample_single_run_unsupervised, args))
            pool.close()
            scores_unsupervised[alg] = np.mean(Accuracies)

            end_time = datetime.now()
            time_taken_alg = end_time - start_time
            time_taken_algs_unsupervised[alg] = time_taken_alg

        print(scores_unsupervised)
        print('Execution time:')
        for k, v in time_taken_algs_unsupervised.items():
            print("{0}: {1}".format(k, v))

    if SUPERVISED_FEAT_SEL:
        scores_supervised = {}
        time_taken_algs_supervised = {}
        for alg in feat_sel_algs_chosen:
            if alg == 'FL' and d > 10:
                continue
            print("{0}: ".format(alg))

            start_time = datetime.now()
            if PARALLELIZE_CV and alg != 'mRMR':
                kf = KFold(n_splits=NO_RUNS, shuffle=True, random_state=42)
                partitions = list(kf.split(X))
                pool = Pool()
                args = [(X, y, type_y, xaxis, FEAT_SEL_ALGS_FN_LABEL[alg], clf, partitions, fold_i,
                         sel_features_UFFS_X) for fold_i in range(0, NO_RUNS)]
                # args = [(X, y, type_y, xaxis, alg, clf, partitions, fold_i,
                        #  sel_features_UFFS_X) for fold_i in range(0, NO_RUNS)]

                Accuracies = np.array(pool.starmap(cv_sample_single_run, args))
                pool.close()
                scores_alg_temp = np.mean(Accuracies, axis=0)
                scores_supervised[alg] = scores_alg_temp[scores_alg_temp > 0]
            else:
                scores_supervised[alg] = accuracy_multiple_train_test(
                    X, y, type_y, xaxis, feat_selection=FEAT_SEL_ALGS_FN_LABEL[alg],
                    clf=clf, no_runs=NO_RUNS, sel_features_UFFS_X=sel_features_UFFS_X)
            end_time = datetime.now()
            time_taken_alg = end_time - start_time
            time_taken_algs_supervised[alg] = time_taken_alg

        print(scores_supervised)
        print('Execution time:')
        for k, v in time_taken_algs_supervised.items():
            print("{0}: {1}".format(k, v))
        file_name = plot_ccm_comparison(xaxis, DATA_NAME, scores=scores_supervised)

        # Add more data:
        scores_supervised['xaxis'] = xaxis
        scores_supervised['data_name'] = DATA_NAME
        scores_supervised['exec_times'] = time_taken_algs_supervised
        scores_supervised['constants'] = {"NO_RUNS": NO_RUNS, "UFFS_SINGLE": UFFS_SINGLE, "UFFS_FOLDS": UFFS_FOLDS,
                               "PARALLELIZE_CV": PARALLELIZE_CV}
        # Store data (serialize)
        with open("../results/{0}.pkl".format(file_name), 'wb') as handle:
            pickle.dump(scores_supervised, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    arguments = ArgumentParser()
    arguments.add_argument('--data', type=str, help='Data name', default=None)
    args = arguments.parse_args()
    if args.data is not None:
        DATA_NAME = args.data
    else:
        print("Data name not specified")
        sys.exit(2)

    print("Data_name: ", DATA_NAME)

    main(DATA_NAME)
    print('=' * 75)
    print("Constants:")
    print("NO_RUNS = ", NO_RUNS)
    print("DATA_NAME = ", DATA_NAME)
    print("UFFS_SINGLE = ", UFFS_SINGLE)
    print("UFFS_FOLDS = ", UFFS_FOLDS)
    print("PARALLELIZE_CV= ", PARALLELIZE_CV)
