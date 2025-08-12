# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
import numpy as np
cimport numpy as np

cdef extern from "<algorithm>" namespace "std":
    void nth_element(double* first, double* nth, double* last)

def l0prune(np.ndarray[np.double_t, ndim=1] v not None, int k):
    cdef int n = v.shape[0]
    cdef double *v_ptr = <double*>v.data
    # absolute values buffer
    cdef np.ndarray[np.double_t, ndim=1] abs_v = np.empty(n, dtype=np.double)
    cdef double *abs_ptr = <double*>abs_v.data
    cdef int i
    for i in range(n):
        abs_ptr[i] = abs(v_ptr[i])
    cdef int pos = n - k
    nth_element(abs_ptr, abs_ptr + pos, abs_ptr + n)
    cdef double thr = abs_ptr[pos]
    # output array
    cdef np.ndarray[np.double_t, ndim=1] sol = np.zeros(n, dtype=np.double)
    cdef double *sol_ptr = <double*>sol.data
    cdef int cnt = 0
    # assign values > thr
    for i in range(n):
        if abs_ptr[i] > thr:
            sol_ptr[i] = v_ptr[i]
    # assign ties (== thr) from end for stability
    for i in range(n-1, -1, -1):
        if abs_ptr[i] == thr and cnt < k:
            sol_ptr[i] = v_ptr[i]
            cnt += 1
    return sol