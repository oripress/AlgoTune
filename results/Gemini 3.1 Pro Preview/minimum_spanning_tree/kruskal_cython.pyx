# cython: boundscheck=False, wraparound=False, cdivision=True
from libc.stdlib cimport malloc, free

cdef struct Edge:
    int u
    int v
    double w
    int original_idx

cdef int compare_edges(const void *a, const void *b) noexcept nogil:
    cdef Edge *ea = <Edge *>a
    cdef Edge *eb = <Edge *>b
    if ea.w < eb.w:
        return -1
    elif ea.w > eb.w:
        return 1
    else:
        if ea.original_idx < eb.original_idx:
            return -1
        elif ea.original_idx > eb.original_idx:
            return 1
        else:
            return 0

cdef extern from "stdlib.h":
    void qsort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *) noexcept nogil) nogil

def solve_kruskal(int num_nodes, object edges):
    cdef int num_edges = len(edges)
    if num_edges == 0 or num_nodes <= 1:
        return []

    cdef Edge *c_edges = <Edge *>malloc(num_edges * sizeof(Edge))
    cdef int i = 0
    for edge in edges:
        c_edges[i].u = int(edge[0])
        c_edges[i].v = int(edge[1])
        c_edges[i].w = float(edge[2])
        c_edges[i].original_idx = i
        i += 1
    qsort(c_edges, num_edges, sizeof(Edge), compare_edges)

    cdef int *parent = <int *>malloc(num_nodes * sizeof(int))
    for i in range(num_nodes):
        parent[i] = i

    cdef list mst_edges = []
    cdef int count = 0
    cdef int u, v, root_u, root_v
    cdef double w

    for i in range(num_edges):
        u = c_edges[i].u
        v = c_edges[i].v
        w = c_edges[i].w

        root_u = u
        while parent[root_u] != root_u:
            parent[root_u] = parent[parent[root_u]]
            root_u = parent[root_u]

        root_v = v
        while parent[root_v] != root_v:
            parent[root_v] = parent[parent[root_v]]
            root_v = parent[root_v]

        if root_u != root_v:
            parent[root_u] = root_v
            if u > v:
                mst_edges.append([v, u, w])
            else:
                mst_edges.append([u, v, w])
            count += 1
            if count == num_nodes - 1:
                break

    free(c_edges)
    free(parent)

    return mst_edges