# cy_gs.pyx
# Cython implementation of Gale-Shapley (proposer-optimal)
# Compiles to C for much faster inner loops.

from libc.stdlib cimport malloc, free
from libc.string cimport memset
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef match(object proposer_prefs, object receiver_prefs):
    """
    proposer_prefs and receiver_prefs: list-of-lists or dict-of-lists
    Returns a Python list 'matching' where matching[p] = r.
    """
    cdef Py_ssize_t n
    cdef object proposers, receivers, prefs

    # Normalize proposer prefs to list-like sequence without unnecessary copies
    if isinstance(proposer_prefs, dict):
        n = len(proposer_prefs)
        proposers = [proposer_prefs[i] for i in range(n)]
    else:
        if isinstance(proposer_prefs, list):
            proposers = proposer_prefs
        else:
            proposers = list(proposer_prefs)
        n = len(proposers)

    # Normalize receiver prefs to list-like sequence
    if isinstance(receiver_prefs, dict):
        receivers = [receiver_prefs[i] for i in range(n)]
    else:
        if isinstance(receiver_prefs, list):
            receivers = receiver_prefs
        else:
            receivers = list(receiver_prefs)

    if n == 0:
        return []

    # allocate flattened arrays: pf (proposer prefs) and rr (receiver rank)
    cdef int *pf = <int*> malloc(n * n * sizeof(int))
    if pf == NULL:
        raise MemoryError()
    cdef int *rr = <int*> malloc(n * n * sizeof(int))
    if rr == NULL:
        free(pf)
        raise MemoryError()

    # helper arrays
    cdef int *nxt = <int*> malloc(n * sizeof(int))
    if nxt == NULL:
        free(pf); free(rr); raise MemoryError()
    # initialize nxt to zeros
    memset(nxt, 0, n * sizeof(int))

    cdef int *rmatch = <int*> malloc(n * sizeof(int))
    if rmatch == NULL:
        free(pf); free(rr); free(nxt); raise MemoryError()
    # set to -1 (all bytes 0xFF yields -1 for two's-complement ints)
    memset(rmatch, 0xFF, n * sizeof(int))

    # stack for free proposers (LIFO)
    cdef int *stack = <int*> malloc(n * sizeof(int))
    if stack == NULL:
        free(pf); free(rr); free(nxt); free(rmatch); raise MemoryError()
    cdef int i
    for i in range(n):
        stack[i] = i
    cdef int top = n  # stack size

    # Use int copy of n for fastest indexing in tight loop
    cdef int nn = <int> n

    # fill pf: pf[p*n + i] = receiver index (cache row pointer)
    cdef int p, rank, r, cur
    cdef int *pf_row = NULL
    cdef int *rr_row = NULL
    for p in range(n):
        prefs = proposers[p]
        pf_row = pf + p * nn
        for i in range(n):
            pf_row[i] = <int> prefs[i]

    # fill rr: rr[r*n + p] = rank (lower is better) (cache row pointer)
    for r in range(n):
        prefs = receivers[r]
        rr_row = rr + r * nn
        for rank in range(n):
            p = <int> prefs[rank]
            rr_row[p] = <int> rank

    # main GS loop -- run without GIL for speed (pure C operations)
    cdef int cur_p, propose_idx, rec
    cdef int *pf_row_local
    cdef int *rr_row_local

    with nogil:
        while top > 0:
            top -= 1
            cur_p = stack[top]
            propose_idx = nxt[cur_p]
            pf_row_local = pf + cur_p * nn
            rec = pf_row_local[propose_idx]
            nxt[cur_p] = propose_idx + 1

            cur = rmatch[rec]
            if cur == -1:
                rmatch[rec] = cur_p
            else:
                rr_row_local = rr + rec * nn
                # receiver prefers lower rank value
                if rr_row_local[cur_p] < rr_row_local[cur]:
                    rmatch[rec] = cur_p
                    # push the displaced proposer
                    stack[top] = cur
                    top += 1
                else:
                    # proposer remains free, push back
                    stack[top] = cur_p
                    top += 1

    # build proposer->receiver matching (back in GIL)
    matching = [0] * n
    for r in range(n):
        p = rmatch[r]
        matching[p] = r

    # free memory
    free(pf)
    free(rr)
    free(nxt)
    free(rmatch)
    free(stack)

    return matching