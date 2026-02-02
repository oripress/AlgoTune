import numpy as np
cimport numpy as np
from scipy.linalg.cython_blas cimport dgemm
from libc.math cimport fabs

cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def solve_cython(double[:, ::1] A):
    cdef int n = A.shape[0]
    cdef int i, j, iter_idx
    cdef double norm_A, row_sum
    
    # Allocate memory
    cdef double[:, ::1] P = np.eye(n, dtype=np.float64)
    cdef double[:, ::1] curr_A = A.copy()
    cdef double[:, ::1] M = np.zeros((n, n), dtype=np.float64)
    cdef double[:, ::1] temp_A = np.zeros((n, n), dtype=np.float64)
    cdef double[:, ::1] term = np.zeros((n, n), dtype=np.float64)
    
    cdef double one = 1.0
    cdef double zero = 0.0
    cdef char *trans_n = 'N'
    cdef char *trans_t = 'T'
    
    # Loop for doubling algorithm
    for iter_idx in range(20):
        # Check convergence using infinity norm of curr_A
        norm_A = 0.0
        for i in range(n):
            row_sum = 0.0
            for j in range(n):
                row_sum += fabs(curr_A[i, j])
            if row_sum > norm_A:
                norm_A = row_sum
        
        if norm_A < 1e-10:
            # Converged
            # Symmetrize P
            for i in range(n):
                for j in range(i + 1, n):
                    P[i, j] = (P[i, j] + P[j, i]) * 0.5
                    P[j, i] = P[i, j]
            return {"is_stable": True, "P": np.asarray(P)}
            
        if norm_A > 1e10:
            return {"is_stable": False, "P": None}
            
        # M = P @ curr_A
        dgemm(trans_n, trans_n, &n, &n, &n, &one, &curr_A[0, 0], &n, &P[0, 0], &n, &zero, &M[0, 0], &n)
        
        # term = curr_A.T @ M
        dgemm(trans_n, trans_t, &n, &n, &n, &one, &M[0, 0], &n, &curr_A[0, 0], &n, &zero, &term[0, 0], &n)
        
        # P += term
        for i in range(n):
            for j in range(n):
                P[i, j] += term[i, j]
                
        # A_new = curr_A @ curr_A
        dgemm(trans_n, trans_n, &n, &n, &n, &one, &curr_A[0, 0], &n, &curr_A[0, 0], &n, &zero, &temp_A[0, 0], &n)
        
        # Copy temp_A to curr_A
        for i in range(n):
            for j in range(n):
                curr_A[i, j] = temp_A[i, j]
                
    return {"is_stable": False, "P": None}