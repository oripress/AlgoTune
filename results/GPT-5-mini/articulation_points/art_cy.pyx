# cython: boundscheck=False
# cython: wraparound=False

from libc.stdlib cimport malloc, free
cimport cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def find_aps_csr(int n, row_ptr_obj, col_obj):
    """
    Find articulation points given CSR arrays:
      row_ptr_obj: array-like of length n+1 (int)
      col_obj: array-like of neighbor indices (int)
    Returns a Python list of articulation point node indices.
    """
    cdef int[::1] r = row_ptr_obj
    cdef int[::1] c = col_obj

    cdef int i
    cdef int *disc = <int*> malloc(n * sizeof(int))
    cdef int *low = <int*> malloc(n * sizeof(int))
    cdef int *parent = <int*> malloc(n * sizeof(int))
    cdef unsigned char *ap = <unsigned char*> malloc(n * sizeof(unsigned char))
    if not disc or not low or not parent or not ap:
        if disc: free(disc)
        if low: free(low)
        if parent: free(parent)
        if ap: free(ap)
        raise MemoryError()

    for i in range(n):
        disc[i] = -1
        low[i] = 0
        parent[i] = -1
        ap[i] = 0

    cdef int *node_stack = <int*> malloc(n * sizeof(int))
    cdef int *parent_stack = <int*> malloc(n * sizeof(int))
    cdef int *idx_stack = <int*> malloc(n * sizeof(int))
    cdef int *child_stack = <int*> malloc(n * sizeof(int))
    if not node_stack or not parent_stack or not idx_stack or not child_stack:
        if node_stack: free(node_stack)
        if parent_stack: free(parent_stack)
        if idx_stack: free(idx_stack)
        if child_stack: free(child_stack)
        free(disc); free(low); free(parent); free(ap)
        raise MemoryError()

    cdef int top = -1
    cdef int start, u, v, p, idx, children, s, e
    cdef int time = 0

    for start in range(n):
        if disc[start] != -1:
            continue

        disc[start] = time
        low[start] = time
        time += 1
        parent[start] = -1

        # push start
        top += 1
        node_stack[top] = start
        parent_stack[top] = -1
        idx_stack[top] = 0
        child_stack[top] = 0

        while top >= 0:
            u = node_stack[top]
            p = parent_stack[top]
            idx = idx_stack[top]
            children = child_stack[top]
            s = r[u]
            e = r[u + 1]
            if idx < e - s:
                v = c[s + idx]
                idx_stack[top] = idx + 1
                if disc[v] == -1:
                    parent[v] = u
                    child_stack[top] = children + 1
                    disc[v] = time
                    low[v] = time
                    time += 1
                    # push v
                    top += 1
                    node_stack[top] = v
                    parent_stack[top] = u
                    idx_stack[top] = 0
                    child_stack[top] = 0
                elif v != p:
                    if low[u] > disc[v]:
                        low[u] = disc[v]
            else:
                # finished u
                top -= 1
                if p != -1:
                    if low[p] > low[u]:
                        low[p] = low[u]
                    if parent[p] != -1 and low[u] >= disc[p]:
                        ap[p] = 1
                else:
                    if children > 1:
                        ap[u] = 1

    # build python list of articulation points
    res = []
    for i in range(n):
        if ap[i]:
            res.append(i)

    free(disc)
    free(low)
    free(parent)
    free(ap)
    free(node_stack)
    free(parent_stack)
    free(idx_stack)
    free(child_stack)

    return res