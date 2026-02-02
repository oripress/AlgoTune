# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from scipy.linalg.cython_lapack cimport dgges
from libc.stdlib cimport malloc, free

ctypedef int (*select_func)(double*, double*, double*)

def solve_qz(list A_list, list B_list):
    cdef int n = len(A_list)
    
    # Allocate memory
    cdef double* a_ptr = <double*> malloc(n * n * sizeof(double))
    cdef double* b_ptr = <double*> malloc(n * n * sizeof(double))
    cdef double* q_ptr = <double*> malloc(n * n * sizeof(double))
    cdef double* z_ptr = <double*> malloc(n * n * sizeof(double))
    
    cdef double* alphar_ptr = <double*> malloc(n * sizeof(double))
    cdef double* alphai_ptr = <double*> malloc(n * sizeof(double))
    cdef double* beta_ptr = <double*> malloc(n * sizeof(double))
    
    cdef int* bwork_ptr = <int*> malloc(n * sizeof(int))
    cdef double* work_ptr = NULL
    
    if not a_ptr or not b_ptr or not q_ptr or not z_ptr or not alphar_ptr or not alphai_ptr or not beta_ptr or not bwork_ptr:
        if a_ptr: free(a_ptr)
        if b_ptr: free(b_ptr)
        if q_ptr: free(q_ptr)
        if z_ptr: free(z_ptr)
        if alphar_ptr: free(alphar_ptr)
        if alphai_ptr: free(alphai_ptr)
        if beta_ptr: free(beta_ptr)
        if bwork_ptr: free(bwork_ptr)
        raise MemoryError()

    cdef int i, j
    cdef list row
    cdef double val
    cdef int idx
    
    try:
        # Fill A and B (Fortran order)
        for i in range(n):
            row = A_list[i]
            for j in range(n):
                val = row[j]
                # A[i, j] -> a_ptr[i + j*n]
                a_ptr[i + j*n] = val
                
        for i in range(n):
            row = B_list[i]
            for j in range(n):
                val = row[j]
                b_ptr[i + j*n] = val
                
        # Prepare dgges
        cdef char jobvsl = 86 # 'V'
        cdef char jobvsr = 86 # 'V'
        cdef char sort = 78   # 'N'
        cdef int sdim = 0
        cdef int info = 0
        cdef double wkopt = 0
        cdef int lwork = -1
        cdef select_func sel = NULL
        
        # Workspace query
        dgges(&jobvsl, &jobvsr, &sort, sel, &n, a_ptr, &n, b_ptr, &n, &sdim,
              alphar_ptr, alphai_ptr, beta_ptr, q_ptr, &n, z_ptr, &n,
              &wkopt, &lwork, bwork_ptr, &info)
              
        if info != 0:
            raise RuntimeError("dgges workspace query failed")
            
        lwork = <int>wkopt
        work_ptr = <double*> malloc(lwork * sizeof(double))
        if not work_ptr:
            raise MemoryError()
            
        # Compute
        dgges(&jobvsl, &jobvsr, &sort, sel, &n, a_ptr, &n, b_ptr, &n, &sdim,
              alphar_ptr, alphai_ptr, beta_ptr, q_ptr, &n, z_ptr, &n,
              work_ptr, &lwork, bwork_ptr, &info)
              
        if info != 0:
            raise RuntimeError("dgges failed")
            
        # Convert to lists
        cdef list AA_list = [None] * n
        cdef list BB_list = [None] * n
        cdef list Q_list = [None] * n
        cdef list Z_list = [None] * n
        
        cdef list row_aa, row_bb, row_q, row_z
        
        for i in range(n):
            row_aa = [None] * n
            row_bb = [None] * n
            row_q = [None] * n
            row_z = [None] * n
            for j in range(n):
                idx = i + j*n
                row_aa[j] = a_ptr[idx]
                row_bb[j] = b_ptr[idx]
                row_q[j] = q_ptr[idx]
                row_z[j] = z_ptr[idx]
            AA_list[i] = row_aa
            BB_list[i] = row_bb
            Q_list[i] = row_q
            Z_list[i] = row_z
            
        return AA_list, BB_list, Q_list, Z_list
        
    finally:
        free(a_ptr)
        free(b_ptr)
        free(q_ptr)
        free(z_ptr)
        free(alphar_ptr)
        free(alphai_ptr)
        free(beta_ptr)
        free(bwork_ptr)
        if work_ptr:
            free(work_ptr)