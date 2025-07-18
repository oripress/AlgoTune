#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3
import numpy as np
cimport numpy as cnp

cdef inline double BSPLINE3(double x):
    cdef double ax = x if x >= 0 else -x
    if ax < 1.0:
        return (4.0 - 6.0*ax*ax + 3.0*ax*ax*ax)/6.0
    elif ax < 2.0:
        cdef double tmp = 2.0 - ax
        return tmp*tmp*tmp/6.0
    else:
        return 0.0

cpdef cnp.ndarray[cnp.double_t, ndim=2] shift2d(
    cnp.ndarray[cnp.double_t, ndim=2] img not None,
    double shift_row, double shift_col
):
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]
    cdef cnp.ndarray[cnp.double_t, ndim=2] out = np.zeros((h, w), dtype=np.float64)
    cdef cnp.ndarray[cnp.intc_t, ndim=2] indexR = np.empty((h, 4), dtype=np.intc)
    cdef cnp.ndarray[cnp.double_t, ndim=2] weightR = np.empty((h, 4), dtype=np.float64)
    cdef cnp.ndarray[cnp.intc_t, ndim=2] indexC = np.empty((w, 4), dtype=np.intc)
    cdef cnp.ndarray[cnp.double_t, ndim=2] weightC = np.empty((w, 4), dtype=np.float64)

    cdef double[:, :] img_mv = img
    cdef double[:, :] out_mv = out
    cdef int[:, :] indexR_mv = indexR
    cdef double[:, :] weightR_mv = weightR
    cdef int[:, :] indexC_mv = indexC
    cdef double[:, :] weightC_mv = weightC

    cdef int i, j, m, n
    cdef double di, dj, df, dfj, frac, val, w_i, w_j

    # Precompute row weights & indices
    for i in range(h):
        di = i - shift_row
        df = np.floor(di)
        frac = di - df
        for m in range(4):
            cdef int idx = <int>df + m - 1
            if idx < 0 or idx >= h:
                weightR_mv[i, m] = 0.0
                indexR_mv[i, m] = 0
            else:
                weightR_mv[i, m] = BSPLINE3((m - 1) - frac)
                indexR_mv[i, m] = idx

    # Precompute col weights & indices
    for j in range(w):
        dj = j - shift_col
        dfj = np.floor(dj)
        frac = dj - dfj
        for n in range(4):
            cdef int idx2 = <int>dfj + n - 1
            if idx2 < 0 or idx2 >= w:
                weightC_mv[j, n] = 0.0
                indexC_mv[j, n] = 0
            else:
                weightC_mv[j, n] = BSPLINE3((n - 1) - frac)
                indexC_mv[j, n] = idx2

    # Compute output via separable convolution
    for i in range(h):
        for j in range(w):
            val = 0.0
            for m in range(4):
                w_i = weightR_mv[i, m]
                if w_i != 0.0:
                    for n in range(4):
                        w_j = weightC_mv[j, n]
                        if w_j != 0.0:
                            val += img_mv[indexR_mv[i, m], indexC_mv[j, n]] * w_i * w_j
            out_mv[i, j] = val

    return out