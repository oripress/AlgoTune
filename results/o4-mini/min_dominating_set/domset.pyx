# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import cython
from libc.stdint cimport uint64_t

cdef extern from *:
    int __builtin_popcountll(uint64_t)
    int __builtin_ctzll(uint64_t)

cdef int n_nodes
cdef int max_cov
cdef uint64_t all_mask
cdef uint64_t neighbor_masks_arr[64]
cdef int best_depth
cdef int best_sol[64]
cdef int sol_stack[64]

@cython.inline
cdef int popcount(uint64_t x):
    return __builtin_popcountll(x)

@cython.inline
cdef int ctz(uint64_t x):
    return __builtin_ctzll(x)

cdef void dfs(uint64_t covered_mask, int depth):
    cdef uint64_t rem
    cdef int rem_bits, lower, u, i
    if covered_mask == all_mask:
        if depth < best_depth:
            best_depth = depth
            for i in range(depth):
                best_sol[i] = sol_stack[i]
        return
    if depth >= best_depth:
        return
    rem = all_mask & (~covered_mask)
    rem_bits = popcount(rem)
    lower = (rem_bits + max_cov - 1) // max_cov
    if depth + lower >= best_depth:
        return
    u = ctz(rem)
    for i in range(n_nodes):
        if neighbor_masks_arr[i] & (1 << u):
            sol_stack[depth] = i
            dfs(covered_mask | neighbor_masks_arr[i], depth+1)

def solve_c(list masks, int nnodes, int max_coverage):
    cdef int i
    global n_nodes, max_cov, best_depth, all_mask
    n_nodes = nnodes
    max_cov = max_coverage
    for i in range(nnodes):
        neighbor_masks_arr[i] = <uint64_t> masks[i]
    all_mask = (1 << nnodes) - 1
    best_depth = nnodes + 1
    dfs(<uint64_t>0, 0)
    res = []
    for i in range(best_depth):
        res.append(best_sol[i])
    return res