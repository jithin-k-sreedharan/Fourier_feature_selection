#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import fourier_learning

from sklearn.model_selection import train_test_split
import scipy.io
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from multiprocessing import Pool
from sklearn import model_selection


# In[10]:


def figsize(scale):
    fig_width_pt = 503.295     # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27   # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0    # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale # width in inches
    fig_height = fig_width*golden_mean  # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
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

cust_color=["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf"]
matplotlib.rcParams['savefig.dpi'] = 125
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath,amssymb,amsfonts}"]


# In[11]:


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


# In[12]:


def load_saved_UFFSfeatures_mohsen(data_name):
    file = "../data/data_Mohsen/selected_features/{0}.mat".format(data_name)
    temp_data = scipy.io.loadmat(file)
    selected_features = np.array(temp_data["SelectedFS"]).squeeze() - 1
    return list(selected_features)


# In[13]:


def clf_score_with_feature_selection(X_train, y_train, X_test, y_test, clf, feats_selected):
    X_sel_train = X_train[:, feats_selected]
    X_sel_test = X_test[:, feats_selected]

    clf.fit(X_sel_train, y_train)
    y_sel_pred = clf.predict(X_sel_test)
    return accuracy_score(y_test, y_sel_pred)


# In[14]:


# Helper function to call supervised fourier selection
def fourier_feature_selection(X, y, k, approx= "depth_based", depth=2):
    mean_emp = np.mean(X, axis=0)
    std_emp = np.std(X, ddof=1, axis=0)
    fourier_featsel = fourier_learning.SupervisedFourierFS(k, mean_emp, std_emp, approx, depth)
    feats_selected = fourier_featsel.fit(X, y)
    return feats_selected


# In[15]:


def remove_zero_variance_fets(X_train, X_test, sel_features_UFFS_X):
    _, d = X_train.shape
    mask = (np.std(X_train, ddof=1, axis=0) > 1e-5)
    valid_features = np.arange(d)
    valid_features = valid_features[mask]
    valid_features = list(set(valid_features).intersection(sel_features_UFFS_X))
    X_train = X_train[:, valid_features]
    X_test = X_test[:, valid_features]
    return X_train, X_test


# In[16]:


FIGSIZE_SCALE_REQD = 1
def plot_score_vs_t(t_vec, score_dict, score_original, title):
    fig, ax = plt.subplots(figsize = figsize(FIGSIZE_SCALE_REQD))
    ax = plt.axes(frameon=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.grid(alpha=1,linestyle='dotted')
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=True)
    ax.tick_params(axis='y', which='minor', left=False)
    for k, scores in score_dict.items():
        plt.plot(t_vec, scores, label = "$k=${0}".format(k), alpha = 0.7)
    plt.axhline(y = score_original*100, color ="black", linestyle ="--", alpha=0.7, label = "no feature selection") 
    plt.xlabel(r"$t$")
    plt.ylabel("accuracy")
    plt.ylim((50,100))
    plt.legend(loc='best')
    plt.title(title)
    plt.savefig("temp.pdf", bbox_inches='tight', pad_inches = 0.02)
    plt.show()


# In[17]:


clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", C=1, gamma = 'auto'))
])


# In[18]:


DATA_NAME = "ALL_AML"


# In[19]:


X, y, type_y = load_data(DATA_NAME)
print("X.shape: ", X.shape)
sel_features_UFFS_X = load_saved_UFFSfeatures_mohsen(DATA_NAME)
print("len(sel_features_UFFS_X): ",len(sel_features_UFFS_X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y, shuffle=True)
score_original = clf_score_with_feature_selection(X_train, y_train, X_test, y_test, clf, np.arange(X.shape[1]))
print("score_original: ", score_original)

# option 1
X_train, X_test = X_train[:,sel_features_UFFS_X], X_test[:,sel_features_UFFS_X]
#---------
# option 2
# X_train, X_test = remove_zero_variance_fets(X_train, X_test, sel_features_UFFS_X)
# score_original = clf_score_with_feature_selection(X_train, y_train, X_test, y_test, clf, np.arange(X_train.shape[1]))
# print("score_original: ", score_original)
# ---------


# In[12]:


# out_file = open('./logs.txt', 'a')
# score = []
# max_t = 7
# k = 20
# for t in range(1,max_t):
#     feats_selected = fourier_feature_selection(X_train, y_train, k, approx="depth_based", depth=t)
#     score_temp = clf_score_with_feature_selection \
#             (X_train, y_train, X_test, y_test, clf, feats_selected)
#     score.append(score_temp)
#     print(t,"{0:0.3f}".format(score_temp))

# print("DATA_NAME: ", DATA_NAME, file=out_file)
# print("max_t: {0}, k: {1}".format(max_t, k), file=out_file)
# print("score_original: {0:3f}".format(score_original*100), file=out_file)
# print("score:{0}\n\n".format(np.array(score)*100), sep=", ", file=out_file)
# out_file.close()


# Covertype result, k = 10, t = 8
# [0.717948717948718,
#  0.7521367521367521,
#  0.7350427350427351,
#  0.7692307692307693,
#  0.7350427350427351,
#  0.7521367521367521,
#  0.7008547008547008,
#  0.6324786324786325]

# In[29]:


score_dict = {
    5: [62.64662541, 71.94665488, 72.80872384, 72.80872384, 72.80872384, 72.97671677], 
    10: [69.18508694, 75.72944297, 72.45652815, 70.39345712, 70.3919835,  69.87474212],
    20: [74.69643383, 75.21072797, 75.55408193, 75.55408193, 75.7250221,  75.55113469]
}


# In[30]:


t_vec = range(1,7)
score_original = 0.693619216033009
title = "Covertype"
plot_score_vs_t(t_vec, score_dict, score_original, "Covertype")


# In[13]:


def cv_sample_single_run(X, y, k, t, clf, partitions, m, sel_features_UFFS_X = []):
    train_indices, test_indices = partitions[m]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    # Variance threshold --------------------------------------------------
    _, d = X_train.shape
    mask = (np.std(X_train, ddof=1, axis=0) > 1e-5)
    valid_features = np.arange(d)
    valid_features = valid_features[mask]

    valid_features = list(set(valid_features).intersection(sel_features_UFFS_X))
    X_train = X_train[:, valid_features]
    X_test = X_test[:, valid_features]
    # ---------------------------------------------------------------------
#     X_train, X_test = X_train[:,sel_features_UFFS_X], X_test[:,sel_features_UFFS_X]
    # ---------------------------------------------------------------------
    feats_selected = fourier_feature_selection(X_train, y_train, k,
                                                   approx="depth_based", depth=t)
    score_temp = clf_score_with_feature_selection(X_train, y_train, X_test, y_test, clf, feats_selected)
    return score_temp    


# In[20]:


np.random.seed(42)
out_file = open('./logs.txt', 'a')
NO_RUNS = 8
k = 10
max_t = 7
scores_mean = []
scores_std = []
for t in range(1, max_t):
    kf = model_selection.StratifiedKFold(n_splits=NO_RUNS, shuffle=True, random_state=42)
    partitions = list(kf.split(X,y))
#     pool = Pool()
#     args = [(X, y, k, t, clf, partitions, fold_i,
#              sel_features_UFFS_X) for fold_i in range(0, NO_RUNS)]
#     Accuracies = np.array(pool.starmap(cv_sample_single_run, args))
#     pool.close()
    Accuracies = []
    for fold_i in range(NO_RUNS):
        Accuracies.append(cv_sample_single_run(X, y, k, t, clf, partitions, fold_i, sel_features_UFFS_X))
        
    scores_mean_temp = np.mean(Accuracies, axis=0)
    scores_std_temp = np.std(Accuracies, axis=0)
    scores_mean.append(scores_mean_temp)
    scores_std.append(scores_std_temp)
    print(scores_mean)
print("With cross-validation...", file=out_file)
print("DATA_NAME: ", DATA_NAME, file=out_file)
print("max_t: {0}, k: {1}, NO_RUNS: {2}".format(max_t, k, NO_RUNS), file=out_file)
print("score_original: {0:3f}".format(score_original*100), file=out_file)
print("score_mean:{0}".format(np.array(scores_mean)*100), sep=", ", file=out_file)
print("score_std:{0}\n\n".format(np.array(scores_std)*100), sep=", ", file=out_file)
out_file.close()


# In[25]:


from sklearn.model_selection import cross_val_score
skf = model_selection.StratifiedKFold(n_splits=5)
Accuracies = cross_val_score(clf, X, y, cv=skf)
scores_mean_orig = np.mean(Accuracies, axis=0)
scores_std_orig = np.std(Accuracies, axis=0)


# In[26]:


scores_mean_orig


# In[27]:


scores_std_orig


# In[ ]:




