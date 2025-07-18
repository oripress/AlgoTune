# Cython extension implementing efficient polyphase upfirdn
# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np

def upfirdn_ext(double[::1] h not None, double[::1] x not None, int up, int down):
    """
    Efficient polyphase upfirdn: y = downsample(convolve(upsampled x, h), down)
    Uses direct loops in C for performance.
    """
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t Lh = h.shape[0]
    # total length of upsampled-convolved signal
    cdef Py_ssize_t Nfull = N * up + Lh - 1
    # length after downsampling
    cdef Py_ssize_t M_out = (Nfull - 1) // down + 1
    # allocate output
    cdef np.ndarray[np.float64_t, ndim=1] y = np.zeros(M_out, dtype=np.float64)
    cdef double[::1] h_view = h
    cdef double[::1] x_view = x
    cdef double[::1] y_view = y
    cdef Py_ssize_t m, n, n_min, n_max, idx, k0, tmp
    cdef double acc
    for m in range(M_out):
        acc = 0.0
        k0 = m * down
        tmp = k0 - (Lh - 1)
        if tmp > 0:
            n_min = (tmp + up - 1) // up
        else:
            n_min = 0
        n_max = k0 // up
        if n_max >= N:
            n_max = N - 1
        for n in range(n_min, n_max + 1):
            idx = k0 - n * up
            acc += x_view[n] * h_view[idx]
        y_view[m] = acc
    return y