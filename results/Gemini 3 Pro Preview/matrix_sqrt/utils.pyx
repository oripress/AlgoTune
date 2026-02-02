import numpy as np
cimport numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.infer_types(True)
def is_hermitian(double complex[:, :] A, double tol=1e-8):
    cdef int n = A.shape[0]
    cdef int i, j
    cdef double complex v1, v2
    cdef double diff_real, diff_imag
    cdef double tol_sq = tol * tol
    
    for i in range(n):
        if A[i, i].imag > tol or A[i, i].imag < -tol:
            return False

    for i in range(n):
        for j in range(i + 1, n):
            v1 = A[i, j]
            v2 = A[j, i]
            diff_real = v1.real - v2.real
            diff_imag = v1.imag + v2.imag
            if (diff_real*diff_real + diff_imag*diff_imag) > tol_sq:
                return False
    return True

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.infer_types(True)
def sqrtm_2x2(double complex[:, :] A):
    cdef double complex a = A[0, 0]
    cdef double complex b = A[0, 1]
    cdef double complex c = A[1, 0]
    cdef double complex d = A[1, 1]
    
    cdef double complex detA = a*d - b*c
    cdef double complex s = detA ** 0.5
    cdef double complex trA = a + d
    cdef double complex t = (trA + 2*s) ** 0.5
    
    cdef double complex inv_t
    cdef np.ndarray[np.complex128_t, ndim=2] X
    
    if t.real*t.real + t.imag*t.imag > 1e-18:
        inv_t = 1.0 / t
        X = np.empty((2, 2), dtype=np.complex128)
        X[0, 0] = (a + s) * inv_t
        X[0, 1] = b * inv_t
        X[1, 0] = c * inv_t
        X[1, 1] = (d + s) * inv_t
        return X
    return None