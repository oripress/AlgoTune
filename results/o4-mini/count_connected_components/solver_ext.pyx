# distutils: language = c
cimport cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef long count_cc(long n, np.ndarray[np.int64_t, ndim=2] ed):
    cdef long[::1] parent = np.arange(n, dtype=np.int64)
    cdef long[::1] rank = np.zeros(n, dtype=np.int64)
    cdef Py_ssize_t m = ed.shape[0]
    cdef Py_ssize_t i
    cdef long u, v, x, ru, rv, comp
    comp = n
    for i in range(m):
        u = ed[i, 0]; v = ed[i, 1]
        x = u
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        ru = x
        x = v
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        rv = x
        if ru != rv:
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            else:
                parent[rv] = ru
                if rank[ru] == rank[rv]:
                    rank[ru] += 1
            comp -= 1
            if comp == 1:
                return comp
    return comp