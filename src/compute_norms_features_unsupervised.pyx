import numpy as np
cimport numpy as np
import math

cdef all_subsets(s, int depth):
    subsets = [[]]
    cdef int x = len(s)
    cdef list loc_set_size_1 = []
    cdef i
    for i in range(x):
        loc_set_size_1.append(len(subsets))
        temp = []
        for set_l in subsets:
            if len(set_l) < depth:
                temp.append(set_l + [s[i]])
        subsets.extend(temp)
    return (subsets, loc_set_size_1)

# def estimate_A(double[:, :] X, int depth, float epsilon):
def estimate_A(np.ndarray[np.float64_t, ndim=2] X, int depth, float epsilon):
    cdef Py_ssize_t n_X = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]

    cdef list loc_set_size_1
    subsets_d, loc_set_size_1 = all_subsets(range(d), depth)
    cdef int len_subsets_d = len(subsets_d)

    cdef np.ndarray[np.float64_t, ndim=2] B = np.zeros(shape=(len_subsets_d, len_subsets_d))
    cdef np.float64_t[:,:] A_temp
    A_temp = np.zeros(shape=(len_subsets_d, len_subsets_d))
    A = np.asarray(A_temp, dtype=np.float64)

    cdef np.ndarray[np.float64_t, ndim=2] X_parity = np.zeros(shape=(n_X, len_subsets_d))
    cdef int i = 0
    cdef list S
    for S in subsets_d:
        X_parity[:,i] = np.prod(X[:,S], axis=1)
        i += 1
    B = (X_parity.T @ X_parity)/n_X

    A[:] = B
    cdef np.ndarray[np.float64_t, ndim=1] norm2_S = np.zeros(len_subsets_d)
    cdef int l

    # Equivalent implementation in vector format
    for i in range(len_subsets_d):
        A[i, :] -= np.sum(A[:i, i].reshape(-1,1) * A[:i, :], axis = 0)
        norm2_S[i] = math.sqrt(max(0, B[i, i] - sum(A[:i, i]**2) ))
        if norm2_S[i] <= epsilon:
            A[i, :] = 0
        else:
            A[i, :] /= norm2_S[i]

    return norm2_S[loc_set_size_1]
