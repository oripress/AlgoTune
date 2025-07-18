# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.math cimport sqrt

# Initialize NumPy C-API
cnp.import_array()

# This is a C-level helper function, not visible to Python.
# It is declared 'nogil' so it can be called from a parallel, no-GIL context.
cdef void _process_row(cnp.float64_t* a_row, cnp.float64_t* A_ub_row, int n) nogil:
    """
    Processes a single row: copies data and calculates the L2 norm.
    's' is a C stack variable, making it perfectly local to each function call.
    """
    cdef double s = 0.0
    cdef double val
    cdef int j
    
    # Loop to copy data and calculate the sum of squares simultaneously.
    for j in range(n):
        val = a_row[j]
        A_ub_row[j] = val
        s += val * val
    
    # Calculate the norm and place it in the last column of the output row.
    A_ub_row[n] = sqrt(s)

# This is the main function that will be called from our Python solver.
cpdef cnp.ndarray[cnp.float64_t, ndim=2] prepare_A_ub_cython(cnp.ndarray[cnp.float64_t, ndim=2] a):
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]
    
    cdef cnp.ndarray[cnp.float64_t, ndim=2] A_ub = np.empty((m, n + 1), dtype=np.float64)
    
    # Create typed memoryviews to get C pointers to the data rows.
    cdef cnp.float64_t[:, :] a_view = a
    cdef cnp.float64_t[:, :] A_ub_view = A_ub
    
    cdef int i
    
    # The prange loop now simply dispatches work to the nogil helper function.
    # The GIL is released for the entire parallel section.
    for i in prange(m, nogil=True):
        # Pass a pointer to the start of the i-th row for both arrays.
        _process_row(&a_view[i, 0], &A_ub_view[i, 0], n)
        
    return A_ub