# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from scipy.linalg.cython_lapack cimport dgesv

cdef class Solver:
    def solve(self, dict problem):
        cdef:
            list A_list = problem["A"]
            list b_list = problem["b"]
            int n = len(b_list)
            int info = 0
            int[:] ipiv = np.empty(n, dtype=np.int32)
            double[:, ::1] A
            double[::1] b
            int i, j
            
        # Allocate contiguous arrays
        A_array = np.empty((n, n), dtype=np.float64, order='F')
        b_array = np.empty(n, dtype=np.float64)
        
        # Fast copy from lists to arrays
        for i in range(n):
            row = A_list[i]
            for j in range(n):
                A_array[i, j] = row[j]
            b_array[i] = b_list[i]
            
        A = A_array
        b = b_array
        
        # Call LAPACK directly
        dgesv(&n, &n, &A[0,0], &n, &ipiv[0], &b[0], &n, &info)
        
        # Convert back to list
        return [b[i] for i in range(n)]