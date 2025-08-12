# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def wasserstein(double[::1] u not None, double[::1] v not None):
    """
    Fast C-level computation of 1D Wasserstein distance for arrays u and v.
    Expects contiguous double arrays of the same length. Normalizes each
    input by its sum (if > 0) and computes sum |cumsum(u') - cumsum(v')|.
    """
    cdef Py_ssize_t n = u.shape[0]
    if n == 0:
        return 0.0

    cdef double su = 0.0
    cdef double sv = 0.0
    cdef Py_ssize_t i
    for i in range(n):
        su += u[i]
        sv += v[i]

    cdef double inu = 0.0
    cdef double inv = 0.0
    if su > 0.0:
        inu = 1.0 / su
    if sv > 0.0:
        inv = 1.0 / sv

    cdef double c = 0.0
    cdef double s = 0.0
    for i in range(n):
        c += u[i] * inu - v[i] * inv
        if c >= 0.0:
            s += c
        else:
            s -= c

    return s