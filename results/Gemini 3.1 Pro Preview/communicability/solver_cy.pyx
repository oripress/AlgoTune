# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from scipy.linalg.cython_lapack cimport dsyev

def compute_comm_cy(list adj_list, int n):
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros((n, n), dtype=np.float64)
    cdef int i, j
    cdef list neighbors
    
    for i in range(n):
        neighbors = adj_list[i]
        for j in neighbors:
            A[i, j] = 1.0
            
    # We can use numpy's eigh for simplicity and speed
    w, v = np.linalg.eigh(A)
    exp_w = np.exp(w)
    exp_A = (v * exp_w) @ v.T
    
    cdef list exp_A_list = exp_A.tolist()
    cdef dict result_comm_dict = {}
    cdef list row
    
    for i in range(n):
        row = exp_A_list[i]
        result_comm_dict[i] = {j: row[j] for j in range(n)}
        
    return result_comm_dict