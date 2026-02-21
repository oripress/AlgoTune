# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
from libc.stdlib cimport malloc, free
from libc.math cimport fabs

def solve_cython(list G, list sigma, list P_min, list P_max, double S_min):
    cdef int n = len(G)
    cdef int i, j, iter_count
    cdef double max_diff, s, p_new, diff, obj, interf, sinr
    cdef bint success = True
    
    # Single allocation for all arrays
    cdef double* mem = <double*>malloc((5 * n + 2 * n * n) * sizeof(double))
    if not mem:
        raise MemoryError()
        
    cdef double* P = mem
    cdef double* c = P + n
    cdef double* p_min_arr = c + n
    cdef double* p_max_arr = p_min_arr + n
    cdef double* sigma_arr = p_max_arr + n
    cdef double* M = sigma_arr + n
    cdef double* G_arr = M + n * n
    
    # Initialize arrays
    cdef list row
    for i in range(n):
        p_min_arr[i] = float(P_min[i])
        p_max_arr[i] = float(P_max[i])
        sigma_arr[i] = float(sigma[i])
        P[i] = p_min_arr[i]
        
        row = G[i]
        for j in range(n):
            G_arr[i * n + j] = float(row[j])
            
    for i in range(n):
        c[i] = S_min * sigma_arr[i] / G_arr[i * n + i]
        for j in range(n):
            if i != j:
                M[i * n + j] = S_min * G_arr[i * n + j] / G_arr[i * n + i]
            else:
                M[i * n + j] = 0.0
                
    for iter_count in range(2000):
        max_diff = 0.0
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += M[i * n + j] * P[j]
                
            p_new = s + c[i]
            if p_new < p_min_arr[i]:
                p_new = p_min_arr[i]
                
            diff = fabs(p_new - P[i])
            if diff > max_diff:
                max_diff = diff
            P[i] = p_new
            
        if max_diff < 1e-9:
            break
            
    obj = 0.0
    for i in range(n):
        obj += P[i]
        if P[i] > p_max_arr[i] + 1e-6:
            success = False
            break
            
        interf = sigma_arr[i]
        for j in range(n):
            if i != j:
                interf += G_arr[i * n + j] * P[j]
        sinr = G_arr[i * n + i] * P[i] / interf
        if sinr < S_min - 1e-6:
            success = False
            break
            
    cdef list P_out = [0.0] * n
    for i in range(n):
        P_out[i] = P[i]
        
    free(mem)
    
    return P_out, success, obj