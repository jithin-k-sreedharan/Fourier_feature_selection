
import sys
import numpy as np
import pandas as pd
import scipy
import scipy.io
import os
import math
import pdb

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from skfeature.function.sparse_learning_based import RFS

import matplotlib.pyplot as plt
import matplotlib
import subprocess
# from multiprocessing import Pool

import fourier_learning
import time
from datetime import datetime
import pickle
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from collections import defaultdict
import argparse
import yaml
import re

from skfeature.function.similarity_based import reliefF
# from skfeature.function.sparse_learning_based import MCFS
# from skfeature.utility import construct_W
# from skfeature.utility.sparse_learning import feature_ranking
from skfeature.utility.sparse_learning import construct_label_matrix, feature_ranking
# from skfeature.function.similarity_based import lap_score
# from skfeature.function.sparse_learning_based import UDFS
# from skfeature.function.sparse_learning_based import NDFS
from tqdm import tqdm

sys.path.append('../lib/CCM/core')
import ccm_v1 as ccm

NO_RUNS = 5
CLF_NAME = "kernel SVM with RBF"  # {"kernel SVM with RBF", "random forest"}

FIGSIZE_SCALE_REQD = 0.5
# =============================================================================
# if FEAT_SEL_ALGS_SUPERVISED['CCM']:
#     sys.path.append('../lib/CCM/core')
#     import ccm_v1 as ccm

# DON'T CHANGE IT
N_UNIQUE_CLASSES_Y = 20
FEAT_SEL_ALGS_FN_LABEL = {
    "FL": "fourier-learning",
    "FL-Depth-2": "fourier-learning_depth_2",
    "FL-Depth-1": "fourier-learning_depth_1",
    "UFFS + SFFS (t=3)": "UFFS + SFFS (t=3)",
    "UFFS + SFFS (t=2)": "UFFS + SFFS (t=2)",
    "UFFS + SFFS (t=1)": "UFFS + SFFS (t=1)",
    # Existing supervised algorithms
    "CCM": "ccm",
    "ReliefF": "relieff",
    "mRMR": "mrmr",
    "RFE": "rfe",
    "RFS": "rfs",
    "MI": "mi",
    "F-Value": "fval",
    # Existing unsupervised algorithms
    "NDFS": True,
    "UDFS": True,
    "MCFS": True,
    "LS": True
}


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

def load_data(data_name):
    try:
        file = "../data/icml_paper/{0}.mat".format(data_name)
        temp_data = scipy.io.loadmat(file)
        X, y = temp_data['X'], temp_data['Y'].squeeze()
        if len(np.unique(y)) > 2:
            type_y = 'categorical'
        else:
            type_y = 'binary'
    except:
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


# def supervised_fs(X_train, y_train, type_y, feat_selection, sel_features_UFFS_X=[]):
def supervised_fs(X_train, y_train, type_y, feat_selection):
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

    # Variance threshold --------------------------------------------------
#     _, d = X_train.shape
#     mask = (np.std(X_train, ddof=1, axis=0) > 1e-5)
#     valid_features = np.arange(d)
#     valid_features = valid_features[mask]

#     if feat_selection in ["fourier-learning_depth_1", "fourier-learning_depth_2","UFFS + SFFS (t=2)", "UFFS + SFFS (t=1)"]:
#         X_train = X_train[:, sel_features_UFFS_X]
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
        feat_sel_alg.scores_
#         all_feats_selected = feat_sel_alg.get_support(indices=True)
        all_feats_selected = np.argsort(-feat_sel_alg.scores_)
    elif feat_selection == "fval":
        feat_sel_alg = SelectKBest(score_func=f_classif, k=d)
        feat_sel_alg.fit(X_train, y_train)
        all_feats_selected = np.argsort(-feat_sel_alg.scores_)
#         all_feats_selected = feat_sel_alg.get_support(indices=True)
    elif feat_selection == "UFFS + SFFS (t=3)":
        all_feats_selected = fourier_feature_selection(X_train, y_train, d,
                                                       approx="depth_based", depth=3)
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
    elif feat_selection == "rfs":
        Y_train = construct_label_matrix(y_train)
        Weight = RFS.rfs(X_train, Y_train, gamma=0.1)
        # sort the feature scores in an ascending order according to the feature scores
        all_feats_selected = feature_ranking(Weight)[:d]
        # obtain the dataset on the selected features
    else:
        print("none selected")
    return all_feats_selected


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
        plt.plot(xaxis[:Fourier_FS_length],np.array(v[:Fourier_FS_length])*100, label=k, alpha=0.7)
        # plt.legend(loc='best')
        plt.xlabel(r"$k$")
        plt.ylabel("Mean accuracy")
    title_str = r"Dataset: {0}".format(data_name)
    title_str = title_str.replace('_', '\_')
    plt.title(title_str)

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center',
                    bbox_to_anchor=(0.45, -0.2), ncol=3)
    
    plt.ylim(25,100)
    if file_name == None:
        timestr = time.strftime("%Y%m%d_%H%M%S")
        file_name = "../results/comparison_feat_sel_{0}".format(timestr)
    plt.savefig('{0}.pdf'.format(file_name),
                bbox_extra_artists=(lgd,), bbox_inches='tight', pad_inches=0.02)
    return file_name


def xaxis_fn(d):
    if d <= 50:
        # xaxis = list(range(1, min(20, d)+1))
        xaxis = list(range(1, d+1))
    elif 50 < d <= 100:
        xaxis = list(range(5, 51, 5))
    elif d > 100:
        xaxis = list(range(10, 101, 10))
    return xaxis


# X_train = X_train[0].copy()
# X_test = X_test[0].copy()
# y_train = y_train[0].copy()
# y_test = y_test[0].copy()
# orig_features = np.arange(X_train.shape[1])
# mask = (np.std(X_train, ddof=1, axis=0) > 1e-5)
# valid_features = orig_features[mask]

# valid_features = valid_features[mask]
# X_train_orig = X_train.copy()
# X_train = X_train[:, valid_features]

# mean_emp_X = np.mean(X_train, axis=0)
# std_emp_X = np.std(X_train, ddof=1, axis=0)

# if UFFS_SETTINGS["cluster_sizes"][0] == -1:
#     UFFS_SETTINGS["cluster_sizes"][0] = d
# UFFS_options = fourier_learning.OptionsUnsupervisedFourierFS(**UFFS_SETTINGS)
# sel_features_UFFS_X_train = fourier_learning.UnsupervisedFourierFS(X_train,UFFS_options)


# feat_sel_algs_chosen = [k for k, v in FEAT_SEL_ALGS_SUPERVISED.items() if v == True]


# seltd_features = {}
# time_taken_algs = {}
# for alg in feat_sel_algs_chosen:
#     print(alg)
#     start_time = datetime.now()
#     if alg in ["fourier-learning_depth_1", "fourier-learning_depth_2","UFFS + SFFS (t=2)","UFFS + SFFS (t=1)"]:
#         relevant_features = supervised_fs(X_train[:, sel_features_UFFS_X_train], y_train, type_y, FEAT_SEL_ALGS_FN_LABEL[alg])
#         seltd_features[alg] = orig_features[mask][sel_features_UFFS_X_train][relevant_features]
#     else:
#         relevant_features = supervised_fs(X_train, y_train, type_y, FEAT_SEL_ALGS_FN_LABEL[alg])
#         seltd_features[alg] = orig_features[mask][relevant_features]
#     end_time = datetime.now()
#     time_taken_algs[alg] = end_time - start_time

# # log-uniform: understand as search over p = exp(x) by varying x
# opt = BayesSearchCV(
#     SVC(),
#     {
#         'C': Real(1e-6, 1e+6, prior='log-uniform'),
#         'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
#         'degree': Integer(1,8),
#         'kernel': Categorical(['linear', 'poly', 'rbf']),
#     },
#     n_iter=32,
#     random_state=0,
#     n_jobs = 5
# )

# # executes bayesian optimization
# _ = opt.fit(X_train, y_train)

# print("Score on test set without feature selection:",opt.score(X_test, y_test))

# xaxis = xaxis_fn(len(sel_features_UFFS_X_train))
# print(xaxis)

# scores = defaultdict(list)
# params = {
#     'C': Real(1e-6, 100, prior='log-uniform'),
#     'gamma': Real(1e-6, 100, prior='log-uniform'),
#     'degree': Integer(1,5),
#     'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid'])
# }

# params_grid_search = { 
#     'C':[0.1,1,100,1000],
#     'kernel':['rbf'],
# #     'kernel':['rbf','poly','sigmoid','linear'],
#     'degree':[1,2,3,4,5,6],
#     'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
# }

# params_grid_search_1 = { 
#     'svm_clf__C':[0.1,1,100,1000],
#     'svm_clf__kernel':['rbf'],
# #     'kernel':['rbf','poly','sigmoid','linear'],
#     'svm_clf__degree':[1,2,3,4,5,6],
#     'svm_clf__gamma': [1, 0.1, 0.01, 0.001, 0.0001]
# }

# clf = Pipeline([
#     ("scaler", StandardScaler()),
#     ("svm_clf", SVC(kernel="rbf", C=1, gamma='auto'))
# ])

# for alg, features in seltd_features.items():
#     print(alg)
#     score_temp = np.zeros(len(xaxis))
#     for i, k in enumerate(xaxis):
#         if k > d:
#             break
#         feats_selected = features[:k]
#         X_train_orig_sel = X_train_orig[:, feats_selected]
#         X_test_sel = X_test[:, feats_selected]
#         # executes bayesian optimization
# #         opt = BayesSearchCV(estimator = SVC(), search_spaces=params, n_jobs=-1, random_state=20)
# #         opt = GridSearchCV(SVC(),params_grid_search, n_jobs = 6)
#         opt = GridSearchCV(clf,params_grid_search_1, n_jobs = 6)
#         opt.fit(X_train_orig_sel, y_train)
        
#         print(opt.score(X_test_sel, y_test))
#         print(opt.best_params_)
#         scores[alg].append(opt.score(X_test_sel, y_test))

# scores_no_opt = defaultdict(list)
# for alg, features in seltd_features.items():
#     print(alg)
#     score_temp = np.zeros(len(xaxis))
#     for i, k in enumerate(xaxis):
#         if k > d:
#             break
#         feats_selected = features[:k]
#         X_train_orig_sel = X_train_orig[:, feats_selected]
#         X_test_sel = X_test[:, feats_selected]
        
#         clf = Pipeline([
#             ("scaler", StandardScaler()),
#             ("svm_clf", SVC(kernel="rbf", C=1, gamma='auto'))
#         ])
#         clf.fit(X_train_orig_sel, y_train)
        
#         print(clf.score(X_test_sel, y_test))
#         scores_no_opt[alg].append(clf.score(X_test_sel, y_test))

# plot_ccm_comparison(xaxis, DATA_NAME, scores, file_name=None)

# # On the entire dataset
# params_grid_search = { 
#     'C':[0.1,1,100,1000],
#     'kernel':['rbf','poly','sigmoid','linear'],
#     'degree':[1,2,3,4,5,6],
#     'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
# }

# opt = GridSearchCV(SVC(),params_grid_search, n_jobs = 6)

# # executes bayesian optimization
# _ = opt.fit(X_train, y_train)

# print("Score on test set without feature selection:",opt.score(X_test, y_test))

# fpath = os.path.join("runs_ICML","{0}_seltd_features_train_seed-{1}.pickle".format(DATA_NAME, random_state))
# with open(fpath, 'wb') as handle:
#     pickle.dump(seltd_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

# scores["xaxis"] = xaxis
# fpath = os.path.join("runs_ICML","{0}_scores_test_seed-{1}.pickle".format(DATA_NAME, random_state))
# with open(fpath, 'wb') as handle:
#     pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_supervised_algs(index, X_train, X_test, y_train, y_test, type_y, config):
    X_train_orig = X_train.copy()
    d = X_train.shape[1]
    if not config["feature_selection_done"]:
        orig_features = np.arange(X_train.shape[1])
        mask = (np.std(X_train, ddof=1, axis=0) > 1e-5)
        valid_features = orig_features[mask]

        valid_features = valid_features[mask]
        X_train = X_train[:, valid_features]


        fourier_orth_settings = config["fourier_orth_settings"]
        # TODO-jithin: This needs to be corrected.
        fourier_orth_settings["cluster_sizes"] = [d if i==-1 else i for i in fourier_orth_settings["cluster_sizes"]]
        UFFS_options = fourier_learning.OptionsUnsupervisedFourierFS(**fourier_orth_settings)
        sel_features_UFFS_X_train = fourier_learning.UnsupervisedFourierFS(X_train,UFFS_options)
        
        feat_sel_algs_chosen = config["algorithms"]
        
        seltd_features = {}
        time_taken_algs = {}
        for alg in feat_sel_algs_chosen:
            print("supervised fs alg: ", alg)
            start_time = datetime.now()
            if alg in ["fourier-learning_depth_1", "fourier-learning_depth_2","UFFS + SFFS (t=2)","UFFS + SFFS (t=1)", "UFFS + SFFS (t=3)"]:
                relevant_features = supervised_fs(X_train[:, sel_features_UFFS_X_train], y_train, type_y, FEAT_SEL_ALGS_FN_LABEL[alg])
                seltd_features[alg] = orig_features[mask][sel_features_UFFS_X_train][relevant_features]
            else:
                relevant_features = supervised_fs(X_train, y_train, type_y, FEAT_SEL_ALGS_FN_LABEL[alg])
                seltd_features[alg] = orig_features[mask][relevant_features]
            end_time = datetime.now()
            time_taken_algs[alg] = end_time - start_time

    else:
        fpath = "../results/{0}/sel_features_train.pickle".format(config["data_name"])
        seltd_features = pickle.load(open(fpath, "rb"))
        seltd_features = {k: v[index] for k, v in seltd_features.items()}

    xaxis = xaxis_fn(len(seltd_features["UFFS + SFFS (t=1)"]))

    scores = defaultdict(list)
    params_grid_search_1 = { 
        'svm_clf__C':[0.1,1,100,1000],
        'svm_clf__kernel':['rbf'],
        'svm_clf__degree':[1,2,3,4,5,6],
        'svm_clf__gamma': [1, 0.1, 0.01, 0.001, 0.0001]
    }

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", C=1, gamma='auto'))
    ])

    for alg, features in seltd_features.items():
        print("classification on fs: ", alg)
        for i, k in enumerate(xaxis):
            if k > d:
                break
            feats_selected = features[:k]
            X_train_orig_sel = X_train_orig[:, feats_selected]
            X_test_sel = X_test[:, feats_selected]
#             opt = GridSearchCV(clf,params_grid_search_1, n_jobs = 6)
#             opt.fit(X_train_orig_sel, y_train)
#             print(opt.score(X_test_sel, y_test))
#             print(opt.best_params_)
#             scores[alg].append(opt.score(X_test_sel, y_test))
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("svm_clf", SVC(kernel="rbf", C=1, gamma='auto'))
            ])
            clf.fit(X_train_orig_sel, y_train)
            scores[alg].append(clf.score(X_test_sel, y_test))
    return scores, seltd_features

def read_config(config_file):
    """Read in data/model config file, parse yml and return dict

    Args: config_file (str): name of config file stored in yml format
    Returns: cfg (dict): dict created by reading in yml file from data or model
        config location
    """
    # Need to load 1e-6 and other such values as floats rather than strings.
    # Below is stack-overflow solution, can be changed if one has a better way
    # to do it.
    # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u"tag:yaml.org,2002:float",
        re.compile(
            u"""^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+) |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]* |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list(u"-+0123456789."),
    )
    config_file_path = os.path.join("../configs", config_file)
    with open(config_file_path, "r") as f:
        cfg = yaml.load(f, Loader=loader)
    return cfg

def fourier_orth(X_train, config):
    orig_features = np.arange(X_train.shape[1])
    mask = (np.std(X_train, ddof=1, axis=0) > 1e-5)
    valid_features = orig_features[mask]

    valid_features = valid_features[mask]
    X_train = X_train[:, valid_features]

    d = X_train.shape[1]

    fourier_orth_settings = config["fourier_orth_settings"]
    fourier_orth_settings["cluster_sizes"] = [d if i==-1 else i for i in fourier_orth_settings["cluster_sizes"]]
    UFFS_options = fourier_learning.OptionsUnsupervisedFourierFS(**fourier_orth_settings)
    sel_features_UFFS_X_train = fourier_learning.UnsupervisedFourierFS(X_train,UFFS_options)
    return orig_features[mask][sel_features_UFFS_X_train]

def main():
    # Parsing the inputs
    parser = argparse.ArgumentParser(description="Provide necessary arguments for running supervised feature selection")
    parser.add_argument("--config_file", type=str, required=True, help="YAML for data and model configuration")
    parser.add_argument("-f", action="store_true", help="feature selection already done")
    args = parser.parse_args()
    config = read_config(args.config_file)

    config["feature_selection_done"] = args.f

    X, y, type_y = load_data(config["data_name"])
    no_samples, d = X.shape
    random_state = config["random_states"]
    X_train, X_test, y_train, y_test = [], [], [], []
    for state in random_state:
        t = train_test_split(X, y, test_size=0.2, random_state=state, stratify=y, shuffle=True)
        for i, j in zip((X_train,X_test,y_train, y_test),t):
            i.append(j)

    scores_collection = defaultdict(list)
    sel_features = defaultdict(list)
    for i, (X_train_i, X_test_i, y_train_i, y_test_i) in enumerate(zip(X_train,X_test,y_train, y_test)):
        scores_temp, sel_features_temp =  run_supervised_algs(i, X_train_i, X_test_i, y_train_i, y_test_i, type_y, config)
        for k, v in scores_temp.items():
            scores_collection[k].append(v)
        for k, v in sel_features_temp.items():
            sel_features[k].append(v)

    scores_ensemble = {}
    for alg_i, scores_i in scores_collection.items():
        scores_i = np.array(scores_i)
    #     if alg_i in ['UFFS + SFFS (t=2)', 'UFFS + SFFS (t=1)']:
        min_len = min([len(score_ii) for score_ii in scores_i])
    #     else:
    #         min_len = None
        avg = np.array([0]*min_len, dtype=np.float64)
        for score_ii in scores_i:
            avg += np.array(score_ii[:min_len])
        avg /= len(scores_i)
        scores_ensemble[alg_i] = avg

    # Plotting    
    dir_name = "../results/{0}".format(config["data_name"])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    xaxis = xaxis_fn(len(scores_ensemble["UFFS + SFFS (t=1)"]))
    plot_ccm_comparison(xaxis, config["data_name"], scores_ensemble, file_name=dir_name+"/plot_comparison")

    # Saving results
    fpath = os.path.join(dir_name,"sel_features_train.pickle")
    with open(fpath, 'wb') as handle:
        pickle.dump(sel_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fpath = os.path.join(dir_name,"scores_collection_train.pickle")
    with open(fpath, 'wb') as handle:
        pickle.dump(scores_collection, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fpath = os.path.join(dir_name,"scores_ensemble_train.pickle")
    with open(fpath, 'wb') as handle:
        pickle.dump(scores_ensemble, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()

