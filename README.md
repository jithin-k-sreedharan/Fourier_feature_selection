# Feature Selection via a Fourier Framework

Computation complexity: The complexity of our Fourer-based feature selection algorithms is $O(d^t n),$ where $t$ is the depth of the algorithm (usually $t$ is 1 or 2). This complexity is independent of $k$.

## Prerequisites

* Python 3.5+ with Anaconda Distribution (Numpy, Scipy, scikit-learn, Cython support)
* gcc/clang with OpenMP support

## Installation

* Compile the Cython files associated to UFFS and SFFS algorithms from the `src` folder

    ```bash
    python setup.py build_ext --inplace
    ```
* Compile C++ implementation of `mRMR` from `lib\mrmr_c_Peng` folder with `bash compile.sh`
* Compile other feature selection implementations using `skfeature` library from `lib\scikit-feature`:

    ```bash
    python setup.py install
    ```

## Running the algorithms

The main Python program for running the algorithms is `src\run_UFFS_SFFS.py`. Use the following command for executing the program:

```bash
python run_UFFS_SFFS.py --data <data_name>
```
where `data_name` is the name of the dataset. The data corresponds to the `data_name` should be loaded appropriately inside the `load_data` function. After execution, output plot and the `Pickle` file associated to it will be then generated inside the `results` folder.


Due to size limitation, we don't provide datasets with this repository. Please download the data files to run the code.

## Notes

* The main function that implements UFFS and SFFS are available in the helper program `fourier_learning.py`.
* The code is parallelized except for mRMR implementation
* The code snippets that require heavy computations (B and A matrix computation in Algorithm 1 and Fourier coefficient calculation in Algorithm 2) are converted to C++ using Cython
* The other arguments and instructions that are specific to functions and classes are provided as comments in the code.
