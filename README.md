# Unsupervised and Supervised Fourier Feature Selection (UFFS and SFFS)

This repository provides implementation of the Fourier-based feature selection algorithms — Fourier-Orth or UFFS and SFFS proposed in the following papers:
* [Finding Relevant Information via a Discrete Fourier Expansion](https://jithin-k-sreedharan.github.io/files/publications/mohsen-jithin_icml2021.pdf)\
Mohsen Heidari, Jithin K. Sreedharan, Gil Shamir, and Wojciech Szpankowski\
_International Conference on Machine Learning (ICML), 2021_

* [Information Sufficiency via Fourier Expansion](https://jithin-k-sreedharan.github.io/files/publications/Jithin_ISIT2021.pdf)\
Mohsen Heidari, Jithin K. Sreedharan, Gil Shamir, and Wojciech Szpankowski\
_IEEE International Symposium on Information Theory (ISIT), 2021_


<!-- Computation complexity: The complexity of our Fourer-based feature selection algorithms is $O(d^t n),$ where $t$ is the depth of the algorithm (usually $t$ is 1 or 2). This complexity is independent of $k$. -->
## Prerequisites

* Python 3.5+ with NumPy, SciPy, scikit-learn, Cython support.
* GCC/Clang

## Installation

Compile the Cython files associated to Fourier-Orth and SFFS algorithms from the `src` folder

```bash
python setup.py build_ext --inplace
```
<!-- * Compile C++ implementation of `mRMR` from `lib\mrmr_c_Peng` folder with `bash compile.sh`
* Compile other feature selection implementations using `skfeature` library from `lib\scikit-feature`:

    ```bash
    python setup.py install
    ``` -->

## Using the Fourier-based feature selection algorithms

The following function shows how to use the UFFS (Fourier-Orth) and SFFS in a sequence.

```python
import fourier_learning
import numpy as np

def SFFS(X, y, t, k, fourier_orth_params):
    """Perform feature selection based on SFFS algorithm (Algorithm 1 in the paper)

    Args:
        X: Input data; NumPy 2-dimensional array of size (no. of data samples, no. of features)
        y: Output data; Numpy 1-dimensional array of size no. of data samples
        t: Depth parameter of SFFS; int
        k: No. of desired features after feature selection; int
        fourier_orth_params: Parameters of Fourier-Orth (Procedure 1) algorithm; dictionary

    Returns:
        features_selected: Selected feature indices; list of integers.
    """
    # Perform Fourier-Orth orthogonalization
    # This step is not necessary and can be avoided. 
    # We can run SFFS directly without Fourier-Orth step
    fourier_orth_options = fourier_learning.OptionsUnsupervisedFourierFS(**fourier_orth_params)
    sel_features_fourier_orth = fourier_learning.UnsupervisedFourierFS(X, fourier_orth_options)

    # Fourier-Orth can be considered as an unsupervised feature selection algorithm
    # It act as a preprocessing step before applying SFFS
    X = X[:, sel_features_fourier_orth]
    mean_emp = np.mean(X, axis=0)
    std_emp = np.std(X, ddof=1, axis=0)

    # Perform SFFS
    fourier_featsel = fourier_learning.SupervisedFourierFS(
        k, mean_emp, std_emp, approx="depth_based", depth=DEPTH_T_SFFS)
    feats_selected = fourier_featsel.fit(X, y)

    return features_selected[:k]
```

Parameters of the Fourier-Orth procedure are:
* `max_depth` (int): Depth parameter of Fourier-Orth procedure. We perform Fourier-Orth in a sequential way —- the procedure is first run with depth 1, then the training set is filtered with the selected features from depth 1 and is passed to depth 2 of the procedure. This process is repeated until the `max_depth` is reached.
* `cluster_sizes` (list of ints): At each depth, to reduce the demand for high computational resources, we partition the features into batches, run Fouier-Orth on each batch and finally combine the results. This parameter specifies the sizes of clusters at each depth
* `norm_epsilon` (list of floats): Threshold `epsilon` at step 9 in the procedure. The parameter lists `epsilon` at different depths.
* `selection_thresholds` (list of floats): Equivalent to threshold `epsilon` at step 15, for each depth. We either use `selection_thresholds` as it is when employing a technique fof explained variance or  1-`selection_thresholds` if we use exact step 15.

An example is given below:
```python
fourier_orth_params = {
    "max_depth": 3,
    "cluster_sizes": [-1, 50, 31],
    "selection_thresholds": [0.95, 0.995, 0.999],
    "norm_epsilon": [0.001, 0.001, 0.0001]
}
```
<!-- ## Notes

* The main function that implements UFFS and SFFS are available in the helper program `fourier_learning.py`.
* The code is parallelized except for mRMR implementation
* The code snippets that require heavy computations (B and A matrix computation in Algorithm 1 and Fourier coefficient calculation in Algorithm 2) are converted to C++ using Cython
* The other arguments and instructions that are specific to functions and classes are provided as comments in the code. -->
