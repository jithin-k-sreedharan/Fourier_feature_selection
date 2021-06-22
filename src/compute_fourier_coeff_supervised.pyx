from itertools import chain, combinations
cimport cython

def powerset(iterable, k):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    return chain.from_iterable(combinations(iterable, r) for r in range(1, k + 1))

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def f_bar_est(double[:,:] X_1, long[:] y, int d, int depth):
    cdef Py_ssize_t X_len = X_1.shape[0]
    subsets_J = powerset(range(d), depth)
    cdef dict f_bar_S_dict = {}
    cdef double s, p
    cdef int i, j
    cdef tuple S
    for S in subsets_J:
        s = 0
        S_len = len(S)
        for i in range(X_len):
            p = 1
            for j in range(S_len):
                p *= X_1[i, S[j]]
            s += y[i] * p
        s /= X_len
        f_bar_S_dict[S] = s
    return f_bar_S_dict