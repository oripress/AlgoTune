# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
ctypedef unsigned long long ull
ctypedef unsigned int idx_t

cdef extern from *:
    unsigned int __builtin_popcountll(unsigned long long) nogil
    unsigned int __builtin_ctzll(unsigned long long) nogil

cdef inline int bc(ull x) nogil:
    return <int>__builtin_popcountll(x)

cdef inline idx_t ctz(ull x) nogil:
    return <idx_t>__builtin_ctzll(x)
cdef inline int color_bound(ull P, ull* neighbors) nogil:
    cdef ull uncol = P, cur, v
    cdef int color = 0
    cdef idx_t i
    while uncol:
        color += 1
        cur = uncol
        while cur:
            v = cur & -cur
            cur -= v
            uncol &= ~v
            i = ctz(v)
            cur &= ~neighbors[i]
    return color
cdef void bronk(ull R, ull P, ull X, int *best_size_p, ull *bestR_p, ull* neighbors, idx_t n) nogil:
    cdef int sizeR = bc(R)
    if P == 0 and X == 0:
        if sizeR > best_size_p[0]:
            best_size_p[0] = sizeR
            bestR_p[0] = R
        return
    if sizeR + bc(P) <= best_size_p[0]:
        return
    cdef int bound = color_bound(P, neighbors)
    if sizeR + bound <= best_size_p[0]:
        return
    cdef ull PX = P | X
    cdef int maxd = -1
    cdef idx_t u = 0
    cdef ull tmp = PX, v
    cdef idx_t i
    cdef int d
    while tmp:
        v = tmp & -tmp
        tmp -= v
        i = ctz(v)
        d = bc(P & neighbors[i])
        if d > maxd:
            maxd = d
            u = i
    cdef ull ext = P & ~neighbors[u]
    tmp = ext
    while tmp:
        v = tmp & -tmp
        tmp -= v
        i = ctz(v)
        bronk(R | v, P & neighbors[i], X & neighbors[i], best_size_p, bestR_p, neighbors, n)
        P &= ~v
        X |= v

def solve_c(object problem):
    cdef Py_ssize_t n = len(problem)
    cdef ull* neighbors = <ull*> PyMem_Malloc(n * sizeof(ull))
    cdef Py_ssize_t i, j
    cdef ull mask, full
    cdef int best_size
    cdef ull bestR
    cdef int* best_size_p = &best_size
    cdef ull* bestR_p = &bestR

    full = (<ull>1 << n) - 1
    for i in range(n):
        mask = full & ~(<ull>1 << i)
        for j in range(n):
            if (<int>problem[i][j]):
                mask &= ~(<ull>1 << j)
        neighbors[i] = mask

    # greedy lower bound
    cdef list order = list(range(n))
    order.sort(key=lambda x: (neighbors[x]).bit_count())
    cdef ull R0 = 0
    for i in order:
        if (R0 & neighbors[i]) == R0:
            R0 |= (<ull>1 << i)

    best_size = bc(R0)
    bestR = R0

    bronk(<ull>0, full, <ull>0, best_size_p, bestR_p, neighbors, <idx_t>n)

    # extract result
    cdef list res = []
    for i in range(n):
        if (bestR >> i) & 1:
            res.append(i)

    PyMem_Free(neighbors)
    return res