# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True  
from libc.stdlib cimport malloc, free  

cdef inline int popcount64(unsigned long long x) nogil:  
    cdef int cnt = 0  
    while x:  
        x &= x - 1  
        cnt += 1  
    return cnt  

cdef inline int ctz64(unsigned long long x) nogil:  
    cdef int i = 0  
    while (x & 1) == 0:  
        x >>= 1  
        i += 1  
    return i  

cdef unsigned long long best_set_c  
cdef int best_clique_c  

cdef void bk_c(unsigned long long R, unsigned long long P, unsigned long long X,  
               unsigned long long* neighbors) nogil:  
    cdef int size_r, deg, u, idx  
    cdef unsigned long long union_px, tmp, v_bit, candidates, Nv  
    if P == 0 and X == 0:  
        size_r = popcount64(R)  
        if size_r > best_clique_c:  
            best_clique_c = size_r  
            best_set_c = R  
        return  
    if popcount64(R) + popcount64(P) <= best_clique_c:  
        return  
    union_px = P | X  
    tmp = union_px  
    u = 0  
    cdef int max_deg = -1  
    while tmp:  
        v_bit = tmp & -tmp  
        tmp ^= v_bit  
        idx = ctz64(v_bit)  
        deg = popcount64(P & neighbors[idx])  
        if deg > max_deg:  
            max_deg = deg  
            u = idx  
    candidates = P & ~neighbors[u]  
    while candidates:  
        v_bit = candidates & -candidates  
        candidates ^= v_bit  
        idx = ctz64(v_bit)  
        Nv = neighbors[idx]  
        bk_c(R | v_bit, P & Nv, X & Nv, neighbors)  
        P ^= v_bit  
        X |= v_bit  

def max_clique_ext(neighbors_py):  
    """  
    neighbors_py: Python list of int bit masks  
    returns: Python list of int vertices  
    """  
    cdef int n = len(neighbors_py)  
    cdef unsigned long long *arr  
    cdef int i  
    cdef unsigned long long full, mask, v_bit  
    cdef list result  
    arr = <unsigned long long*> malloc(n * sizeof(unsigned long long))  
    if not arr:  
        raise MemoryError()  
    for i in range(n):  
        arr[i] = <unsigned long long>neighbors_py[i]  
    best_clique_c = 0  
    best_set_c = 0  
    if n >= 64:  
        full = <unsigned long long>-1  
    else:  
        full = (<unsigned long long>1 << n) - 1  
    with nogil:  
        bk_c(0, full, 0, arr)  
    result = []  
    mask = best_set_c  
    while mask:  
        v_bit = mask & -mask  
        mask ^= v_bit  
        result.append(ctz64(v_bit))  
    free(arr)  
    result.sort()  
    return result