import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def householder_qr_cython(double[:, :] A):
    cdef int m = A.shape[0]
    cdef int n = A.shape[1]
    cdef double[:, :] R = np.copy(A)
    cdef double[:, :] Q = np.eye(m, dtype=np.float64)
    
    cdef int i, j, k
    cdef double norm_x, sign, v_dot, tmp
    cdef double[::1] v = np.empty(m, dtype=np.float64)
    
    for j in range(min(m, n)):
        # Compute norm of the j-th column starting from j
        norm_x = 0.0
        for i in range(j, m):
            norm_x += R[i, j] * R[i, j]
        norm_x = sqrt(norm_x)
        
        sign = 1.0 if R[j, j] >= 0 else -1.0
        v[j] = R[j, j] + sign * norm_x
        
        for i in range(j+1, m):
            v[i] = R[i, j]
        
        # Normalize v
        v_dot = 0.0
        for i in range(j, m):
            v_dot += v[i] * v[i]
        v_dot = sqrt(v_dot)
        
        for i in range(j, m):
            v[i] /= v_dot
        
        # Apply Householder reflection to R
        for k in range(j, n):
            tmp = 0.0
            for i in range(j, m):
                tmp += v[i] * R[i, k]
            for i in range(j, m):
                R[i, k] -= 2 * v[i] * tmp
        
        # Apply Householder reflection to Q
        for k in range(m):
            tmp = 0.0
            for i in range(j, m):
                tmp += v[i] * Q[i, k]
            for i in range(j, m):
                Q[i, k] -= 2 * v[i] * tmp
    
    return np.asarray(Q), np.asarray(R)