import sys
import numpy as np
from argparse import ArgumentParser
import scipy.io
from datetime import datetime
import math
from collections import Counter

import fourier_learning

from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, KFold

from multiprocessing import Pool

# Helper function to call the classifier using only the selected features
NO_RUNS = 5
CLF_NAME = "kernel SVM with RBF"  # {"kernel SVM with RBF", "random forest"}
UFFS_SINGLE = True
N_UNIQUE_CLASSES_Y = 20

def load_saved_UFFSfeatures_mohsen(data_name):
    file = "../data/data_Mohsen/selected_features/{0}.mat".format(data_name)
    temp_data = scipy.io.loadmat(file)
    selected_features = np.array(temp_data["SelectedFS"]).squeeze() - 1
    return list(selected_features)

def clf_score_with_feature_selection(X_train, y_train, X_test, y_test, clf, feats_selected):
    X_sel_train = X_train[:, feats_selected]
    X_sel_test = X_test[:, feats_selected]

    clf.fit(X_sel_train, y_train)
    y_sel_pred = clf.predict(X_sel_test)
    return accuracy_score(y_test, y_sel_pred)

def load_data(data_name):
    file = "../data/data_Mohsen/{0}.mat".format(data_name)
    temp_data = scipy.io.loadmat(file)
    X, y = temp_data['X'], temp_data['Y'].squeeze()
    if len(np.unique(y)) > 2:
        type_y = 'categorical'
    else:
        type_y = 'binary'

    if type_y == 'binary' or type_y == 'categorical':
        if y.dtype != np.int:
            y = y.astype(np.int, copy=False)
    return X, y, type_y

def cv_sample_single_run(X, y, type_y, xaxis, feat_selection, clf, partitions, k,
                         sel_features_UFFS_X = []):
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

        if feat_selection in ["fourier-learning_depth_1", "fourier-learning_depth_2"]:
            mean_emp = np.mean(X_train, axis=0)
            std_emp = np.std(X_train, ddof=1, axis=0)
            sel_features_UFFS = fourier_learning.UnsupervisedFourierFS(X_train, mean_emp, std_emp)
            X_train = X_train[:, sel_features_UFFS]
            X_test = X_test[:, sel_features_UFFS]
    elif UFFS_SINGLE:
        if feat_selection in ["fourier-learning_depth_1", "fourier-learning_depth_2"]:
            valid_features = list(set(valid_features).intersection(sel_features_UFFS_X))
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
        all_feats_selected = fourier_feature_selection(X_train, y_train, d, approx="none")
    elif feat_selection == "rfe":
        clf_rfe = LinearSVC(loss="squared_hinge", penalty="l1", dual=False, max_iter=2000)
        feat_sel_alg = RFE(clf_rfe, n_features_to_select=d, step=1)
        X_train_1 = StandardScaler().fit_transform(X_train)
        feat_sel_alg.fit(X_train_1, y_train)
        all_feats_selected = feat_sel_alg.get_support(indices=True)

    score_temp = np.zeros(len(xaxis))

    for i, k in enumerate(xaxis):
        if k > d:
            break
        feats_selected = all_feats_selected[:k]
        score_temp[i] += clf_score_with_feature_selection \
            (X_train, y_train, X_test, y_test, clf, feats_selected)

    return score_temp

def main(DATA_NAME, FEAT_SELECTION=None):
    np.random.seed(42)

    if CLF_NAME == 'kernel SVM with RBF':
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", C=1, gamma = 'auto'))
        ])
    elif CLF_NAME == 'random forest':
        clf = RandomForestClassifier(n_estimators=100)

    # Load data
    X, y, type_y = load_data(DATA_NAME)
    # TODO: change no_samples to n
    no_samples, d = X.shape
    if len(np.unique(y)) < no_samples//2:
        global N_UNIQUE_CLASSES_Y
        N_UNIQUE_CLASSES_Y = len(np.unique(y))

    # X = X[load_saved_UFFSfeatures_mohsen(DATA_NAME)]
    sel_features_UFFS_X = load_saved_UFFSfeatures_mohsen(DATA_NAME)

    kf = KFold(n_splits=NO_RUNS, shuffle=True, random_state=42)
    partitions = list(kf.split(X))
    pool = Pool()
    # args = [(X, y, type_y, xaxis, FEAT_SEL_ALGS_FN_LABEL[alg], clf, partitions, fold_i,
    #          sel_features_UFFS_X) for fold_i in range(0, NO_RUNS)]
    args = [(X, y, type_y, xaxis, alg, clf, partitions, fold_i,
                sel_features_UFFS_X) for fold_i in range(0, NO_RUNS)]

    Accuracies = np.array(pool.starmap(cv_sample_single_run, args))
    pool.close()
    scores_alg_temp = np.mean(Accuracies, axis=0)
    scores_supervised[alg] = scores_alg_temp[scores_alg_temp > 0]
    
    print(scores_supervised)

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