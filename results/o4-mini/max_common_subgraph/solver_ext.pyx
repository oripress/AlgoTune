# cython: language_level=3
cimport cython
from libc.stdint cimport uint64_t

cdef inline int popcount64(uint64_t x):
    return __builtin_popcountll(x)

cdef inline int bit_index(uint64_t x):
    return __builtin_ctzll(x)

cdef class BKSolver64:
    cdef uint64_t *neis
    cdef int N
    cdef int max_size
    cdef uint64_t max_mask

    def __cinit__(self, uint64_t[:] neighbors_arg, int N_arg, int init_size, uint64_t init_mask):
        self.N = N_arg
        self.max_size = init_size
        self.max_mask = init_mask
        self.neis = &neighbors_arg[0]

    cdef void expand(self, uint64_t P, int r_count, uint64_t R_mask):
        cdef uint64_t Ptemp, lsb, cand
        cdef int u, max_u, cnt, max_cnt, v
        if P == 0:
            if r_count > self.max_size:
                self.max_size = r_count
                self.max_mask = R_mask
            return
        if r_count + popcount64(P) <= self.max_size:
            return
        Ptemp = P
        max_u = -1
        max_cnt = -1
        while Ptemp:
            lsb = Ptemp & -Ptemp
            u = bit_index(lsb)
            Ptemp -= lsb
            cnt = popcount64(P & self.neis[u])
            if cnt > max_cnt:
                max_cnt = cnt
                max_u = u
        cand = P & ~self.neis[max_u]
        while cand:
            lsb = cand & -cand
            v = bit_index(lsb)
            cand -= lsb
            self.expand(P & self.neis[v], r_count + 1, R_mask | lsb)
            P -= lsb

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple solve_graph(uint64_t[:] neighbors, int N, uint64_t P0, int init_size, uint64_t init_mask):
    cdef BKSolver64 solver = BKSolver64(neighbors, N, init_size, init_mask)
    solver.expand(P0, 0, <uint64_t>0)
    return solver.max_size, solver.max_mask