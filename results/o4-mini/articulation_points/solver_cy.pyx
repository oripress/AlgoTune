# cython: language_level=3
# solver_cy.pyx
# Cython implementation of articulation point detection (Tarjan algorithm)
# distutils: extra_compile_args = -Ofast -march=native -funroll-loops -flto
# distutils: extra_link_args = -flto

cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cpython.list cimport PyList_New, PyList_SET_ITEM
from cpython.long cimport PyLong_FromLong

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void dfs_c(int* head, int* to, int* nxt,
                       int* disc, int* low, int* parent,
                       bint* ap, int* time_ptr, int u) nogil:
    cdef int v, i, child_count
    disc[u] = time_ptr[0]
    low[u] = time_ptr[0]
    time_ptr[0] += 1
    child_count = 0
    i = head[u]
    while i != -1:
        v = to[i]
        if disc[v] == -1:
            parent[v] = u
            child_count += 1
            dfs_c(head, to, nxt, disc, low, parent, ap, time_ptr, v)
            if low[v] < low[u]:
                low[u] = low[v]
            if parent[u] == -1:
                if child_count > 1:
                    ap[u] = True
            else:
                if low[v] >= disc[u]:
                    ap[u] = True
        elif v != parent[u]:
            if disc[v] < low[u]:
                low[u] = disc[v]
        i = nxt[i]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list find_aps(int n, int[:, :] edges):
    """
    edges: memoryview of shape (m,2) with C-contiguous int data (e.g., numpy intc).
    """
    cdef int m = edges.shape[0]
    cdef int M = m * 2
    # allocate C arrays
    cdef int* head = <int*>malloc(n * sizeof(int))
    cdef int* to   = <int*>malloc(M * sizeof(int))
    cdef int* nxt  = <int*>malloc(M * sizeof(int))
    cdef int* disc = <int*>malloc(n * sizeof(int))
    cdef int* low  = <int*>malloc(n * sizeof(int))
    cdef int* parent = <int*>malloc(n * sizeof(int))
    cdef bint* ap    = <bint*>malloc(n * sizeof(bint))
    if head == NULL or to == NULL or nxt == NULL or disc == NULL \
       or low == NULL or parent == NULL or ap == NULL:
        # allocation failure
        if head    != NULL: free(head)
        if to      != NULL: free(to)
        if nxt     != NULL: free(nxt)
        if disc    != NULL: free(disc)
        if low     != NULL: free(low)
        if parent  != NULL: free(parent)
        if ap      != NULL: free(ap)
        return []
    # initialize arrays using memset
    memset(head,   0xFF, n * sizeof(int))   # -1
    memset(disc,   0xFF, n * sizeof(int))   # -1
    memset(parent, 0xFF, n * sizeof(int))   # -1
    memset(low,     0,   n * sizeof(int))   # 0
    memset(ap,      0,   n * sizeof(bint))  # False
    # build adjacency list
    cdef int i, idx = 0, u, v
    for i in range(m):
        u = edges[i, 0]
        v = edges[i, 1]
        to[idx] = v
        nxt[idx] = head[u]
        head[u] = idx
        idx += 1
        to[idx] = u
        nxt[idx] = head[v]
        head[v] = idx
        idx += 1
    cdef int time = 0
    # DFS without GIL
    with nogil:
        for i in range(n):
            if disc[i] == -1:
                parent[i] = -1
                dfs_c(head, to, nxt, disc, low, parent, ap, &time, i)
    # collect results under GIL
    cdef int count = 0
    for i in range(n):
        if ap[i]:
            count += 1
    cdef object result = PyList_New(count)
    cdef int pos = 0
    for i in range(n):
        if ap[i]:
            PyList_SET_ITEM(result, pos, PyLong_FromLong(i))
            pos += 1
    # free memory
    free(head); free(to); free(nxt)
    free(disc); free(low); free(parent); free(ap)
    return <list>result