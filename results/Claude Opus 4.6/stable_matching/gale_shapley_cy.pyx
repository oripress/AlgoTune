# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False
from libc.stdlib cimport malloc, free
from libc.string cimport memset
import numpy as np
cimport numpy as np

np.import_array()

def gale_shapley(proposer_prefs, receiver_prefs, int n):
    cdef int* recv_rank_flat = <int*>malloc(n * n * sizeof(int))
    cdef int* next_prop = <int*>malloc(n * sizeof(int))
    cdef int* recv_match = <int*>malloc(n * sizeof(int))
    cdef int* prop_prefs_flat = <int*>malloc(n * n * sizeof(int))
    cdef int* stack = <int*>malloc(n * sizeof(int))
    
    cdef int i, j, rank, p, r, cur, stack_top
    cdef list prefs
    
    # Fill proposer prefs flat
    for i in range(n):
        prefs = proposer_prefs[i]
        for j in range(n):
            prop_prefs_flat[i * n + j] = <int>prefs[j]
    
    # Build receiver rank tables
    for r in range(n):
        prefs = receiver_prefs[r]
        for rank in range(n):
            recv_rank_flat[r * n + <int>prefs[rank]] = rank
    
    # Initialize
    memset(next_prop, 0, n * sizeof(int))
    for i in range(n):
        recv_match[i] = -1
        stack[i] = i
    
    stack_top = n
    
    while stack_top > 0:
        stack_top -= 1
        p = stack[stack_top]
        r = prop_prefs_flat[p * n + next_prop[p]]
        next_prop[p] += 1
        
        cur = recv_match[r]
        if cur == -1:
            recv_match[r] = p
        else:
            if recv_rank_flat[r * n + p] < recv_rank_flat[r * n + cur]:
                recv_match[r] = p
                stack[stack_top] = cur
                stack_top += 1
            else:
                stack[stack_top] = p
                stack_top += 1
    
    # Build matching
    result = [0] * n
    for r in range(n):
        result[recv_match[r]] = r
    
    free(recv_rank_flat)
    free(next_prop)
    free(recv_match)
    free(prop_prefs_flat)
    free(stack)
    
    return result

def gale_shapley_np(np.ndarray[int, ndim=2, mode="c"] prop_prefs,
                    np.ndarray[int, ndim=2, mode="c"] recv_prefs, int n):
    cdef int* pp = <int*>prop_prefs.data
    cdef int* rp = <int*>recv_prefs.data
    cdef int* recv_rank_flat = <int*>malloc(n * n * sizeof(int))
    cdef int* next_prop = <int*>malloc(n * sizeof(int))
    cdef int* recv_match = <int*>malloc(n * sizeof(int))
    cdef int* stack = <int*>malloc(n * sizeof(int))
    
    cdef int i, rank, p, r, cur, stack_top
    
    # Build receiver rank tables
    for r in range(n):
        for rank in range(n):
            recv_rank_flat[r * n + rp[r * n + rank]] = rank
    
    # Initialize
    memset(next_prop, 0, n * sizeof(int))
    for i in range(n):
        recv_match[i] = -1
        stack[i] = i
    
    stack_top = n
    
    while stack_top > 0:
        stack_top -= 1
        p = stack[stack_top]
        r = pp[p * n + next_prop[p]]
        next_prop[p] += 1
        
        cur = recv_match[r]
        if cur == -1:
            recv_match[r] = p
        else:
            if recv_rank_flat[r * n + p] < recv_rank_flat[r * n + cur]:
                recv_match[r] = p
                stack[stack_top] = cur
                stack_top += 1
            else:
                stack[stack_top] = p
                stack_top += 1
    
    # Build matching
    result = [0] * n
    for r in range(n):
        result[recv_match[r]] = r
    
    free(recv_rank_flat)
    free(next_prop)
    free(recv_match)
    free(stack)
    
    return result