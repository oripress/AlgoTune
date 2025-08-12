# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free

ctypedef np.int32_t I32
ctypedef np.int16_t I16

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef object gs_solve(np.ndarray[I32, ndim=2] prop_prefs,
                      np.ndarray[I32, ndim=2] recv_prefs):
    cdef int n = prop_prefs.shape[0]
    cdef int i, p, r, rk, cur, p_next, base, r_base, p_base
    cdef I32* prop_ptr = <I32*> prop_prefs.data
    cdef I32* recv_ptr = <I32*> recv_prefs.data

    # State arrays
    cdef I32* next_ptr = <I32*> malloc(n * sizeof(I32))
    cdef I32* recv_match_ptr = <I32*> malloc(n * sizeof(I32))

    # Rank arrays (one of them will be used)
    cdef I16* rank16 = NULL
    cdef I32* rank32 = NULL

    cdef bint use16 = n <= 32767

    if next_ptr is NULL or recv_match_ptr is NULL:
        if next_ptr is not NULL: free(next_ptr)
        if recv_match_ptr is not NULL: free(recv_match_ptr)
        raise MemoryError()

    if use16:
        rank16 = <I16*> malloc(n * n * sizeof(I16))
        if rank16 is NULL:
            free(next_ptr)
            free(recv_match_ptr)
            raise MemoryError()
    else:
        rank32 = <I32*> malloc(n * n * sizeof(I32))
        if rank32 is NULL:
            free(next_ptr)
            free(recv_match_ptr)
            raise MemoryError()

    with nogil:
        # Build receiver ranking matrix
        if use16:
            for r in range(n):
                base = r * n
                for rk in range(n):
                    p = recv_ptr[base + rk]
                    rank16[base + p] = <I16> rk
        else:
            for r in range(n):
                base = r * n
                for rk in range(n):
                    p = recv_ptr[base + rk]
                    rank32[base + p] = rk

        # Initialize state arrays
        for i in range(n):
            next_ptr[i] = 0
            recv_match_ptr[i] = -1

        # Gale-Shapley algorithm with chained proposals
        if use16:
            for i in range(n):
                p = i
                while True:
                    p_next = next_ptr[p]
                    p_base = p * n
                    r = prop_ptr[p_base + p_next]
                    next_ptr[p] = p_next + 1

                    cur = recv_match_ptr[r]
                    if cur == -1:
                        recv_match_ptr[r] = p
                        break

                    r_base = r * n
                    if rank16[r_base + p] < rank16[r_base + cur]:
                        recv_match_ptr[r] = p
                        p = cur  # displaced proposer continues proposing
                    else:
                        # rejected; proposer p tries next choice (loop continues with same p)
                        pass
        else:
            for i in range(n):
                p = i
                while True:
                    p_next = next_ptr[p]
                    p_base = p * n
                    r = prop_ptr[p_base + p_next]
                    next_ptr[p] = p_next + 1

                    cur = recv_match_ptr[r]
                    if cur == -1:
                        recv_match_ptr[r] = p
                        break

                    r_base = r * n
                    if rank32[r_base + p] < rank32[r_base + cur]:
                        recv_match_ptr[r] = p
                        p = cur  # displaced proposer continues proposing
                    else:
                        # rejected; proposer p tries next choice (loop continues with same p)
                        pass

    # Build proposer->receiver mapping as Python list
    cdef list matching = [0] * n
    for r in range(n):
        p = recv_match_ptr[r]
        matching[p] = r

    # Free allocated C arrays
    if rank16 is not NULL:
        free(rank16)
    if rank32 is not NULL:
        free(rank32)
    free(next_ptr)
    free(recv_match_ptr)

    return matching