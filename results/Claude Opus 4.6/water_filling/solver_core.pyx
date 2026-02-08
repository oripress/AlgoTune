# cython: boundscheck=False, wraparound=False, cdivision=True
from libc.math cimport log
from libc.stdlib cimport malloc, free, qsort

cdef int compare_doubles(const void* a, const void* b) noexcept nogil:
    cdef double da = (<double*>a)[0]
    cdef double db = (<double*>b)[0]
    if da < db:
        return -1
    elif da > db:
        return 1
    return 0

def solve_water_filling(list alpha_list, double P_total):
    cdef int n = len(alpha_list)
    cdef int i, k, k_idx
    cdef double prefix_sum, w, w_candidate, s, scale, cap, val
    
    if n == 0 or P_total <= 0:
        return None, None
    
    cdef double* alpha = <double*>malloc(n * sizeof(double))
    cdef double* sorted_alpha = <double*>malloc(n * sizeof(double))
    cdef double* x_opt = <double*>malloc(n * sizeof(double))
    
    for i in range(n):
        alpha[i] = <double>alpha_list[i]
        sorted_alpha[i] = alpha[i]
        if alpha[i] <= 0:
            free(alpha)
            free(sorted_alpha)
            free(x_opt)
            return None, None
    
    qsort(sorted_alpha, n, sizeof(double), compare_doubles)
    
    prefix_sum = 0.0
    w = 0.0
    for k_idx in range(n):
        prefix_sum += sorted_alpha[k_idx]
        k = k_idx + 1
        w_candidate = (P_total + prefix_sum) / k
        if k == n or w_candidate <= sorted_alpha[k]:
            w = w_candidate
            break
    
    s = 0.0
    for i in range(n):
        val = w - alpha[i]
        if val > 0.0:
            x_opt[i] = val
            s += val
        else:
            x_opt[i] = 0.0
    
    cap = 0.0
    if s > 0.0:
        scale = P_total / s
        for i in range(n):
            x_opt[i] *= scale
            cap += log(alpha[i] + x_opt[i])
    else:
        for i in range(n):
            cap += log(alpha[i])
    
    result_x = [x_opt[i] for i in range(n)]
    result_cap = cap
    
    free(alpha)
    free(sorted_alpha)
    free(x_opt)
    
    return result_x, result_cap