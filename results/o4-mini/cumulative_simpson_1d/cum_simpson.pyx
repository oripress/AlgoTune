#cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython cimport Py_ssize_t

def cum_simpson(np.ndarray[np.double_t, ndim=1] y, double dx):
    cdef Py_ssize_t N = y.shape[0]
    if N < 2:
        return np.empty(0, dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] res = np.empty(N-1, dtype=np.double)
    cdef double y0 = y[0]
    res[0] = 0.5 * dx * (y0 + y[1])
    cdef double W = 4.0 * y[1]
    cdef Py_ssize_t i
    for i in range(2, N):
        cdef double yi = y[i]
        if (i & 1) == 0:
            res[i-1] = dx / 3.0 * (y0 + yi + W)
        else:
            cdef double yim1 = y[i-1]
            res[i-1] = dx/3.0 * (y0 + yim1 + (W - 2.0*yim1)) + 0.5*dx*(yim1 + yi)
        if (i & 1) == 1:
            W += 4.0 * yi
        else:
            W += 2.0 * yi
    return res