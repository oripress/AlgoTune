# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.math cimport fabs

def compute_pagerank(list adj_list, int n, double alpha, int max_iter, double tol):
    cdef int i, j, d_i, num_dangling = 0
    cdef double dangling_sum, base_score, val, err, diff
    
    cdef double* r = <double*>malloc(n * sizeof(double))
    cdef double* r_next = <double*>malloc(n * sizeof(double))
    cdef double* inv_degree = <double*>malloc(n * sizeof(double))
    cdef int* dangling_nodes = <int*>malloc(n * sizeof(int))
    
    if not r or not r_next or not inv_degree or not dangling_nodes:
        if r: free(r)
        if r_next: free(r_next)
        if inv_degree: free(inv_degree)
        if dangling_nodes: free(dangling_nodes)
        raise MemoryError()
        
    cdef double initial_val = 1.0 / n
    for i in range(n):
        r[i] = initial_val
        
    cdef list neighbors
    for i in range(n):
        neighbors = <list>adj_list[i]
        d_i = len(neighbors)
        if d_i > 0:
            inv_degree[i] = alpha / d_i
        else:
            inv_degree[i] = 0.0
            dangling_nodes[num_dangling] = i
            num_dangling += 1
            
    cdef int iter_idx
    for iter_idx in range(max_iter):
        dangling_sum = 0.0
        for i in range(num_dangling):
            dangling_sum += r[dangling_nodes[i]]
            
        base_score = (1.0 - alpha) / n + alpha * dangling_sum / n
        for i in range(n):
            r_next[i] = base_score
            
        for i in range(n):
            neighbors = <list>adj_list[i]
            d_i = len(neighbors)
            if d_i > 0:
                val = r[i] * inv_degree[i]
                for j in range(d_i):
                    r_next[<int>neighbors[j]] += val
                    
        err = 0.0
        for i in range(n):
            diff = r_next[i] - r[i]
            err += diff if diff > 0 else -diff
            r[i] = r_next[i]
            
        if err < n * tol:
            break
            
    cdef list result = [0.0] * n
    if err < n * tol:
        for i in range(n):
            result[i] = r[i]
            
    free(r)
    free(r_next)
    free(inv_degree)
    free(dangling_nodes)
    
    return result