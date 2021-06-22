# Find feature selection indices of NDFS algorithm for the Mohsen's datasets

import scipy.io
import os.path
import numpy as np
import pandas as pd
import sys
from argparse import ArgumentParser
from datetime import datetime
import math
import subprocess

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from skfeature.function.similarity_based import reliefF
from skfeature.function.sparse_learning_based import RFS
from skfeature.utility.sparse_learning import construct_label_matrix, feature_ranking

NO_FOLDS = 5
NO_FOLDS_USED = 5
FEAT_SELECTION = 'CCM'
SAVE_TO_FILE = True

if FEAT_SELECTION == 'CCM':
    sys.path.append('../lib/CCM/core')
    import ccm_v1 as ccm

def load_data(data_name):
    folder = "../data/data_Mohsen/"
    file = folder + "{0}.mat".format(data_name)
    temp_data = scipy.io.loadmat(file)
    X, y = temp_data['X'], temp_data['Y'].squeeze()
    n_data, d = X.shape
    if len(np.unique(y)) > 2:
        type_y = 'categorical'
    else:
        type_y = 'binary'

    if type_y == 'binary' or type_y == 'categorical':
        if y.dtype != np.int:
            y = y.astype(np.int, copy=False)

    segments_file = folder + "{0}_segments.mat".format(data_name)
    if os.path.exists(segments_file):
        print("segments file found")
        segments_data = scipy.io.loadmat(segments_file)
        SampleShuffle, segments = segments_data['SampleShuffle'].squeeze(), segments_data['segments'].squeeze()
        SampleShuffle = SampleShuffle - 1
    else:
        print("segments file not found")
        SampleShuffle = np.random.permutation(n_data)
        segments = np.round(np.linspace(0, n_data, NO_FOLDS+1)).astype(np.int)
    return X, y, type_y, segments, SampleShuffle


# Condiational co-variance feature selection
def ccm_feature_selection(X, y, type_y):
    epsilon = 0.001
    _, d = X.shape
    if d <= 100:
        num_features = math.ceil(d / 5)
    else:
        num_features = 100
    rank_ccm = ccm.ccm(X, y, num_features, type_y, epsilon, iterations=100, verbose=False)
    all_feats_selected = np.argsort(rank_ccm)
    return all_feats_selected


def classifier():


def main(DATA_NAME):
    # Load data
    X, y, type_y, segments, SampleShuffle = load_data(DATA_NAME)
    X = X[SampleShuffle, :]
    y = y[SampleShuffle]
    n_data, d = X.shape
    all_feats_selected = {}
    start_time = datetime.now()
    if FEAT_SELECTION == "RFS":
        Y = construct_label_matrix(y)
    for fold_idx in range(NO_FOLDS_USED):
        print(fold_idx)
        train_fold_1 = np.arange(segments[fold_idx])
        train_fold_2 = np.arange(segments[fold_idx + 1], n_data)
        train_fold = np.concatenate([train_fold_1, train_fold_2])

        X_train = X[train_fold, :]
        # Variance threshold --------------------------------------------------
        _, d = X_train.shape
        mask = (np.std(X_train, ddof=1, axis=0) > 1e-5)
        orig_features = np.arange(d)
        valid_features = orig_features[mask]
        X_train = X_train[:, valid_features]
        y_train = y[train_fold]
        _, d = X_train.shape

        if FEAT_SELECTION == "CCM":
            # Run CCM
            feats_seltd = ccm_feature_selection(X_train, y_train, type_y)
        elif FEAT_SELECTION == "reliefF":
            score_reliefF_temp = reliefF.reliefF(X_train, y_train)
            # rank features in descending order according to score
            feats_seltd = reliefF.feature_ranking(score_reliefF_temp)[:d]
        elif FEAT_SELECTION == "MI":
            feat_sel_alg = SelectKBest(score_func=mutual_info_classif, k=d)
            feat_sel_alg.fit(X_train, y_train)
            feats_seltd = feat_sel_alg.get_support(indices=True)
        elif FEAT_SELECTION == "mRMR":
            ## Prepare data
            y_train = y_train.reshape(-1, 1)
            a = ["class"] + [str(i) for i in range(0, d)]
            df_mRMR = pd.DataFrame(data=np.concatenate((y_train, X_train), axis=1), columns=a)
            df_mRMR.to_csv('../data/temp_mRMR_data.csv', index=False)
            y_train = y_train.ravel()

            ## Call the C++ executable
            command_mRMR = '../lib/mrmr_c_Peng/./mrmr -i ../data/temp_mRMR_data.csv -n {0}'.format(d)
            out_code = subprocess.call([command_mRMR], shell=True)
            if out_code != 0:
                print("something wrong")
                sys.exit(2)
            feats_seltd = np.loadtxt('out_temp.txt', dtype=np.int)
            #==================================================================
        elif FEAT_SELECTION == "RFS":
            # Run RFS
            Weight = RFS.rfs(X_train, Y[train_fold, :], gamma=0.1)
            feats_seltd = feature_ranking(Weight)

        all_feats_selected["fold_{0}".format(fold_idx)] = orig_features[mask][feats_seltd]

    all_feats_selected['SampleShuffle'] = SampleShuffle
    all_feats_selected['segments'] = segments

    if SAVE_TO_FILE:
        folder = "../results/data_Mohsen/"
        scipy.io.savemat(folder+DATA_NAME+"_{0}.mat".format(FEAT_SELECTION),\
                         all_feats_selected)
        print("Saved to file")

    end_time = datetime.now()
    time_taken_alg = end_time - start_time

    print("Data_name: ", DATA_NAME)
    print("Feature_selection: ",FEAT_SELECTION)
    print('Execution time: ', time_taken_alg)

if __name__ == "__main__":
    arguments = ArgumentParser()
    arguments.add_argument('--data', type=str, help='Data name', default= None)
    args = arguments.parse_args()
    if args.data is not None:
        DATA_NAME = args.data
    else:
        print("Data name not specified")
        sys.exit(2)
    main(DATA_NAME)