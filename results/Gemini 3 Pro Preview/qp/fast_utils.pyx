# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
import scipy.sparse as spa
from scipy.sparse import csc_matrix

def prepare_data(problem):
    cdef object P_in = problem.get("P", problem.get("Q"))
    cdef object q_in = problem["q"]
    cdef object G_in = problem["G"]
    cdef object h_in = problem["h"]
    cdef object A_in = problem["A"]
    cdef object b_in = problem["b"]

    # Convert vectors
    cdef np.ndarray[np.float64_t, ndim=1] q = np.array(q_in, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] h = np.array(h_in, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] b = np.array(b_in, dtype=np.float64)
    
    cdef int n = q.shape[0]
    cdef int m = h.shape[0]
    cdef int p = b.shape[0]

    # Convert matrices to numpy arrays for fast access
    # We assume inputs are lists of lists or arrays.
    # Convert matrices to numpy arrays for fast access
    # P_arr: C-contiguous for fast P[j, i] access
    cdef np.ndarray[np.float64_t, ndim=2] P_arr = np.array(P_in, dtype=np.float64, order='C')
    # P_T_arr: C-contiguous copy of P.T for fast P[i, j] access (accessed as P_T[j, i])
    cdef np.ndarray[np.float64_t, ndim=2] P_T_arr = np.ascontiguousarray(P_arr.T)
    
    cdef np.ndarray[np.float64_t, ndim=2] G_arr
    cdef np.ndarray[np.float64_t, ndim=2] A_arr
    
    if m > 0:
        G_arr = np.array(G_in, dtype=np.float64, order='F')
        if G_arr.ndim == 1: G_arr = G_arr.reshape(1, -1)
    else:
        G_arr = np.empty((0, n), dtype=np.float64, order='F')
        
    if p > 0:
        A_arr = np.array(A_in, dtype=np.float64, order='F')
        if A_arr.ndim == 1: A_arr = A_arr.reshape(1, -1)
    else:
        A_arr = np.empty((0, n), dtype=np.float64, order='F')

    # Construct P sparse (CSC, upper triangular of (P+P.T)/2)
    cdef int max_nnz_P = n * (n + 1) // 2
    cdef np.ndarray[np.float64_t, ndim=1] P_data = np.empty(max_nnz_P, dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] P_indices = np.empty(max_nnz_P, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] P_indptr = np.zeros(n + 1, dtype=np.int32)
    
    cdef int idx_P = 0
    cdef int i, j
    cdef double val
    
    for j in range(n):
        for i in range(j + 1):
            # P[i, j] comes from P_T_arr[j, i]
            # P[j, i] comes from P_arr[j, i]
            val = 0.5 * (P_T_arr[j, i] + P_arr[j, i])
            if val != 0:
                P_data[idx_P] = val
                P_indices[idx_P] = i
                idx_P += 1
        P_indptr[j+1] = idx_P
        
    P_sparse = csc_matrix((P_data[:idx_P], P_indices[:idx_P], P_indptr), shape=(n, n))
    P_sparse = csc_matrix((P_data[:idx_P], P_indices[:idx_P], P_indptr), shape=(n, n))

    # Construct A sparse (CSC, [G; A])
    cdef int max_nnz_A = (m + p) * n
    cdef np.ndarray[np.float64_t, ndim=1] A_data = np.empty(max_nnz_A, dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] A_indices = np.empty(max_nnz_A, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] A_indptr = np.zeros(n + 1, dtype=np.int32)
    
    cdef int idx_A = 0
    
    for j in range(n):
        # G part
        for i in range(m):
            val = G_arr[i, j]
            if val != 0:
                A_data[idx_A] = val
                A_indices[idx_A] = i
                idx_A += 1
        # A part
        for i in range(p):
            val = A_arr[i, j]
            if val != 0:
                A_data[idx_A] = val
                A_indices[idx_A] = m + i
                idx_A += 1
        A_indptr[j+1] = idx_A
        
    A_sparse = csc_matrix((A_data[:idx_A], A_indices[:idx_A], A_indptr), shape=(m + p, n))
    
    # l and u
    cdef np.ndarray[np.float64_t, ndim=1] l = np.empty(m + p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] u = np.empty(m + p, dtype=np.float64)
    
    if m > 0:
        l[:m] = -np.inf
        u[:m] = h
    if p > 0:
        l[m:] = b
        u[m:] = b
        
    return P_sparse, q, A_sparse, l, u