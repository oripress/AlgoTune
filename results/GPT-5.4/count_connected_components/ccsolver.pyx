# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False, cdivision=True

from cpython.list cimport PyList_GET_ITEM, PyList_GET_SIZE
from libc.stdlib cimport free, malloc
from libc.string cimport memset

cdef inline int _find_root(int x, int* parent) noexcept:
    while parent[x] >= 0:
        x = parent[x]
    return x

cpdef int count_components(int n, list edges):
    cdef int* parent
    cdef int components, u, v, x, y
    cdef Py_ssize_t i, m
    cdef tuple e

    if n <= 1:
        return n

    parent = <int*>malloc(n * sizeof(int))
    if parent == NULL:
        raise MemoryError()

    try:
        memset(parent, 255, n * sizeof(int))
        components = n
        m = PyList_GET_SIZE(edges)

        for i in range(m):
            e = <tuple>PyList_GET_ITEM(edges, i)
            u = e[0]
            v = e[1]

            x = _find_root(u, parent)
            y = _find_root(v, parent)

            if x != y:
                if parent[x] > parent[y]:
                    x, y = y, x
                parent[x] += parent[y]
                parent[y] = x
                components -= 1
                if components == 1:
                    break

        return components
    finally:
        free(parent)