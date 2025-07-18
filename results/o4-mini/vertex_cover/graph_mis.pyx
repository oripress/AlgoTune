# distutils: language = c
cimport cython
from libc.stdint cimport uint64_t
from libc.stdlib cimport malloc, free

cdef extern from *:
    int __builtin_popcountll(unsigned long long x)
    int __builtin_ctzll(unsigned long long x)

cdef uint64_t *CAF
cdef int best_size
cdef uint64_t best_mask

cdef void bk(uint64_t R, uint64_t P, uint64_t X):
    cdef int R_size = __builtin_popcountll(R)
    cdef int P_size = __builtin_popcountll(P)
    if R_size + P_size <= best_size:
        return
    if P == 0 and X == 0:
        best_size = R_size
        best_mask = R
        return
    cdef uint64_t U = P | X
    cdef int idx = __builtin_ctzll(U)
    cdef uint64_t mu = CAF[idx]
    cdef uint64_t candidates = P & ~mu
    while candidates:
        cdef uint64_t v_bit = candidates & -candidates
        candidates -= v_bit
        cdef int vidx = __builtin_ctzll(v_bit)
        bk(R | v_bit, P & CAF[vidx], X & CAF[vidx])
        P -= v_bit
        X |= v_bit
        if __builtin_popcountll(R) + __builtin_popcountll(P) <= best_size:
            return

@cython.boundscheck(False)
@cython.wraparound(False)
def max_clique_mask(comp_adj_py):
    """
    comp_adj_py: Python list of int masks (uint64)
    returns best_mask as Python int of max clique in the complement graph.
    """
    cdef int n = len(comp_adj_py)
    if n == 0:
        return 0
    global CAF, best_size, best_mask
    best_size = 0
    best_mask = 0
    CAF = <uint64_t *>malloc(n * sizeof(uint64_t))
    cdef int i
    for i in range(n):
        CAF[i] = <uint64_t>comp_adj_py[i]
    cdef uint64_t full = (<uint64_t>1 << n) - 1
    bk(<uint64_t>0, full, <uint64_t>0)
    free(CAF)
    return best_mask