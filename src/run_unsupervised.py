"""
* Implementation of unsupervised feature selection algorithms
* Calculate / save classification accuracy
* Clustering NMI / accuracy
* No pipeline implemented

To find help with the command-line arguments, run: python run_unsupervised.py --help
"""


import sys
import numpy as np
from argparse import ArgumentParser
import scipy.io
from datetime import datetime
import math
from collections import Counter

from skfeature.utility import construct_W
from skfeature.utility.sparse_learning import feature_ranking
from skfeature.function.sparse_learning_based import NDFS
from skfeature.function.sparse_learning_based import MCFS
from skfeature.function.similarity_based import lap_score
from skfeature.function.sparse_learning_based import UDFS
from skfeature.utility import unsupervised_evaluation

import fourier_learning

TEST_SIZE = 0.25
N_UNIQUE_CLASSES_Y = 20
READ_UFFS_FEATUES_FRM_FILE = True
SAVE_TO_FILE = True
FIND_ACCURACY = False
REPORT_METRICS = True

NO_SEL_FEATURES_K = {
    "Isolet": 306, #254
    "data1": 34,
    "data6": 11,
    "data13": 35,
    "ALL_AML": 39
}
if FIND_ACCURACY:
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", C=1, gamma='auto'))
    ])


# Helper function to call the classifier using only the selected features
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

def load_saved_features_mohsen(data_name):
    file = "../data/data_Mohsen/selected_features/{0}.mat".format(data_name)
    temp_data = scipy.io.loadmat(file)
    selected_features = np.array(temp_data["SelectedFS"]).squeeze() - 1
    return list(selected_features)

def main(DATA_NAME, FEAT_SELECTION):
    np.random.seed(42)

    # Load data
    X, y, type_y = load_data(DATA_NAME)
    no_samples, d = X.shape
    if len(np.unique(y)) < no_samples//2:
        global N_UNIQUE_CLASSES_Y
        N_UNIQUE_CLASSES_Y = len(np.unique(y))

    # mask = (np.std(X_train, ddof=1, axis=0) > 1e-5)
    # orig_features = np.arange(d)
    # valid_features = orig_features[mask]
    # X_train = X_train[:, valid_features]
    # y_train = y[train_fold]
    # _, d = X_train.shape

    start_time = datetime.now()

    if FEAT_SELECTION == "UFFS":
        if READ_UFFS_FEATUES_FRM_FILE:
            all_feats_selected = load_saved_features_mohsen(DATA_NAME)
        else:
            UFFS_options = \
                fourier_learning.OptionsUnsupervisedFourierFS(max_depth=3,
                                                            cluster_sizes=[d, 50, 25],
                                                            selection_thresholds=[0.95,0.95,0.95],
                                                            norm_epsilon=[0.001, 0.001, 0.001],
                                                            shuffle=False,
                                                            preranking="non"
                                                            )
            all_feats_selected = fourier_learning.UnsupervisedFourierFS(X,UFFS_options)
        print("No of selected features by UFFS: ", len(all_feats_selected))
    elif FEAT_SELECTION == "MCFS":
        # construct affinity matrix
        kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": \
            "heatKernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X, **kwargs)
        # obtain the feature weight matrix
        Weight = MCFS.mcfs(X, n_selected_features=d, W=W,\
                           n_clusters=N_UNIQUE_CLASSES_Y)
        # sort the feature scores in an ascending order according to the feature scores
        all_feats_selected = MCFS.feature_ranking(Weight)[:d]

    elif FEAT_SELECTION == "NDFS":
        # construct affinity matrix
        kwargs = {"metric": "euclidean", "neighborMode": "knn",\
                  "weightMode": "heatKernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X, **kwargs)
        # obtain the feature weight matrix
        Weight = NDFS.ndfs(X, W=W, n_clusters=N_UNIQUE_CLASSES_Y)
        all_feats_selected = feature_ranking(Weight)[:d]
    elif FEAT_SELECTION == "UDFS":
        # obtain the feature weight matrix
        Weight = UDFS.udfs(X, gamma=0.1, n_clusters=N_UNIQUE_CLASSES_Y)
        all_feats_selected = feature_ranking(Weight)[:d]
    elif FEAT_SELECTION == "LS":
        # construct affinity matrix
        kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn",\
                    "weight_mode": "heat_kernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X, **kwargs_W)
        # obtain the scores of features
        score = lap_score.lap_score(X, W=W)
        # sort the feature scores in an ascending order according to the feature scores
        all_feats_selected = lap_score.feature_ranking(score)[:d]
    elif FEAT_SELECTION == "None":
        all_feats_selected = list(range(d))

    end_time = datetime.now()
    time_taken_alg = end_time - start_time

    if FIND_ACCURACY:
        acc = cross_val_score(clf, X[:,all_feats_selected], y, cv=5, scoring='accuracy', n_jobs=-1)
        print("Accuracy: {:0.2f} ".format(np.mean(acc)))

    if SAVE_TO_FILE and FEAT_SELECTION != "None":
        if FEAT_SELECTION == "UFFS" and (not READ_UFFS_FEATUES_FRM_FILE):
            np.savetxt("../results/data_Mohsen/"+DATA_NAME+"_{0}_indices.txt"\
                       .format(FEAT_SELECTION), all_feats_selected, fmt="%d")
            print("Saved to file")
        elif FEAT_SELECTION != "UFFS":
            np.savetxt("../results/data_Mohsen/"+DATA_NAME+"_{0}_UFFS-K-{1}_indices.txt"\
                       .format(FEAT_SELECTION, NO_SEL_FEATURES_K[DATA_NAME]), all_feats_selected, fmt="%d")
            print("Saved to file")

    if REPORT_METRICS:
        # perform kmeans clustering based on the selected features and repeats 20 times
        nmi_mean, nmi_std, acc_mean, acc_std = 0, 0, 0, 0
        if FEAT_SELECTION in ["None", "UFFS"]:
            k = len(all_feats_selected)
            if FEAT_SELECTION == "UFFS" and READ_UFFS_FEATUES_FRM_FILE:
                assert k == NO_SEL_FEATURES_K[DATA_NAME], "Mismatch in NO_SEL_FEATURES_K ({0}) " \
                                                          "and selected features {1} from the file given by Mohsen"\
                    .format(NO_SEL_FEATURES_K[DATA_NAME],k)
        else:
            k = NO_SEL_FEATURES_K[DATA_NAME]

        # for _ in range(0, 20):
        #     nmi, acc = unsupervised_evaluation.evaluation\
        #         (X_selected=X[:,all_feats_selected[:k]], n_clusters=N_UNIQUE_CLASSES_Y, y=y)
        #     nmi_mean += nmi
        #     acc_mean+= acc
        #     nmi_std += nmi**2
        #     acc_std += acc**2
        # nmi_mean /= 20
        # acc_mean /= 20
        # nmi_std, acc_std = 0, 0
        # # nmi_std = math.sqrt(nmi_std/20 - nmi_mean*nmi_mean)
        # # acc_std = math.sqrt(acc_std/20 - acc_mean*acc_mean)
        # print("NMI: {0:.3f} \u00B1 {1:0.3f}, ACC: {2:.3f} \u00B1 {3:.3f}".format(nmi_mean, nmi_std, acc_mean, acc_std))

        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import make_scorer
        acc_scorer = make_scorer(accuracy_score)

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", C=1, gamma='auto', random_state=42))
        ])
        # clf = SVC(kernel="rbf", C=1, gamma='auto', random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
        X_sel_train = X_train[:, all_feats_selected[:k]]
        X_sel_test = X_test[:, all_feats_selected[:k]]
        clf.fit(X_sel_train, y_train)
        y_sel_pred = clf.predict(X_sel_test)
        print("accuracy_score: {0:0.3f}".format(accuracy_score(y_test, y_sel_pred)))

        def svc_param_selection(X, y, nfolds=5):
            Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            gammas = [0.001, 0.01, 0.1, 1, 10, 100, 'auto']
            # Cs = [1]
            # gammas = ['auto']
            # param_grid = {'C': Cs, 'gamma': gammas, 'random_state': [42]}
            param_grid = {'svm_clf__C': Cs, 'svm_clf__gamma': gammas}
            pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", random_state=42))
            ])
            # grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, scoring = acc_scorer, cv=nfolds, n_jobs = -1)
            grid_search = GridSearchCV(pipe, param_grid, scoring = acc_scorer, cv=nfolds, n_jobs = -1)
            grid_search.fit(X, y)
            # print("cv_results_: ", grid_search.cv_results_)
            # print("best_score_: ", grid_search.best_score_)
            return grid_search.best_estimator_, grid_search.best_params_
            # return grid_search, grid_search.best_params_

        best_svc_clf, best_svc_clf_params  = svc_param_selection(X_sel_train, y_train)
        # best_svc_clf.fit(X_sel_train, y_train)
        # X_sel_test = StandardScaler().fit_transform(X_sel_test)
        y_sel_pred = best_svc_clf.predict(X_sel_test)
        print("best accuracy_score: {0:0.3f}".format(accuracy_score(y_test, y_sel_pred)))
        print("best parameters: ", best_svc_clf_params)

    print("Data_name: ", DATA_NAME)
    print("data prevalence: ", Counter(y))
    print("no_samples, no_features: ",X.shape)
    print("Feature_selection: ",FEAT_SELECTION)
    print('Execution time: ', time_taken_alg)

if __name__ == "__main__":
    arguments = ArgumentParser()
    arguments.add_argument('--data', type=str, help='Data name: {Isolet,USPS}', default= None)
    arguments.add_argument('--alg', type=str, help='Algorithm name: {MCFS,UDFS,NDFS,LS,UFFS,None', default="None")
    args = arguments.parse_args()
    if args.data is not None:
        DATA_NAME = args.data
    else:
        print("Data name not specified")
        sys.exit(2)
    FEAT_SELECTION = args.alg
    main(DATA_NAME, FEAT_SELECTION)