# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
import scipy.linalg as la

np.import_array()

cdef class Solver:
    def solve(self, problem):
        """Solve the generalized eigenvalue problem for the given matrices A and B."""
        A, B = problem
        
        # Use scipy's generalized eigenvalue solver directly - it's already highly optimized
        eigenvalues = la.eigvals(A, B)
        
        # Convert to structured array for efficient sorting
        # This avoids creating multiple temporary arrays
        n = len(eigenvalues)
        dtype = [('real', 'f8'), ('imag', 'f8'), ('idx', 'i4')]
        arr = np.empty(n, dtype=dtype)
        
        cdef int i
        for i in range(n):
            arr[i] = (-eigenvalues[i].real, -eigenvalues[i].imag, i)
        
        # Sort by real part (descending), then imaginary part (descending)
        arr.sort(order=['real', 'imag'])
        
        # Extract sorted eigenvalues
        result = []
        for i in range(n):
            idx = arr[i]['idx']
            result.append(eigenvalues[idx])
        
        return result