# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as cnp

cpdef cnp.ndarray[cnp.float32_t, ndim=2] apsp_floyd_warshall_undirected(
    cnp.ndarray[cnp.float32_t, ndim=1] data,
    cnp.ndarray[cnp.int32_t, ndim=1] indices,
    cnp.ndarray[cnp.int32_t, ndim=1] indptr,
    int n,
):
    cdef:
        cnp.ndarray[cnp.float32_t, ndim=2] dist = np.empty((n, n), dtype=np.float32)
        float inf = np.float32(np.inf)
        int i, j, k, p, start, end
        float w, dik, alt
        float[:, :] dist_view = dist
        float[:] row
        float[:] dk
        float[:] ri

    for i in range(n):
        row = dist_view[i]
        for j in range(n):
            row[j] = inf
        row[i] = 0.0

    for i in range(n):
        row = dist_view[i]
        start = indptr[i]
        end = indptr[i + 1]
        for p in range(start, end):
            j = indices[p]
            w = data[p]
            if w < row[j]:
                row[j] = w
            if w < dist_view[j, i]:
                dist_view[j, i] = w

    for k in range(n):
        dk = dist_view[k]
        for i in range(n):
            dik = dist_view[i, k]
            if dik == inf:
                continue
            ri = dist_view[i]
            for j in range(i + 1, n):
                alt = dik + dk[j]
                if alt < ri[j]:
                    ri[j] = alt
                    dist_view[j, i] = alt

    return dist
cpdef cnp.ndarray[cnp.float32_t, ndim=2] apsp_dijkstra_undirected(
    cnp.ndarray[cnp.float32_t, ndim=1] data,
    cnp.ndarray[cnp.int32_t, ndim=1] indices,
    cnp.ndarray[cnp.int32_t, ndim=1] indptr,
    int n,
):
    cdef:
        int m = data.shape[0]
        cnp.ndarray[cnp.float32_t, ndim=2] dist = np.empty((n, n), dtype=np.float32)
        cnp.ndarray[cnp.float32_t, ndim=1] heap_d_arr = np.empty(m + 1, dtype=np.float32)
        cnp.ndarray[cnp.int32_t, ndim=1] heap_v_arr = np.empty(m + 1, dtype=np.int32)
        cnp.ndarray[cnp.uint8_t, ndim=1] seen_arr = np.empty(n, dtype=np.uint8)
        float inf = np.float32(np.inf)
        int s, i, u, v, p, start, end, heap_size, idx, parent, child, right, best
        float du, nd, last_d
        int last_v
        float[:, :] dist_view = dist
        float[:] row
        float[:] heap_d = heap_d_arr
        int[:] heap_v = heap_v_arr
        unsigned char[:] seen = seen_arr

    for s in range(n):
        row = dist_view[s]
        for i in range(n):
            row[i] = inf
            seen[i] = 0
        row[s] = 0.0

        heap_size = 1
        heap_d[0] = 0.0
        heap_v[0] = s

        while heap_size > 0:
            du = heap_d[0]
            u = heap_v[0]
            heap_size -= 1

            if heap_size > 0:
                last_d = heap_d[heap_size]
                last_v = heap_v[heap_size]
                idx = 0
                while True:
                    child = (idx << 1) + 1
                    if child >= heap_size:
                        break
                    right = child + 1
                    best = child
                    if right < heap_size and heap_d[right] < heap_d[child]:
                        best = right
                    if heap_d[best] >= last_d:
                        break
                    heap_d[idx] = heap_d[best]
                    heap_v[idx] = heap_v[best]
                    idx = best
                heap_d[idx] = last_d
                heap_v[idx] = last_v

            if seen[u]:
                continue
            if du != row[u]:
                continue
            seen[u] = 1

            start = indptr[u]
            end = indptr[u + 1]
            for p in range(start, end):
                v = indices[p]
                if seen[v]:
                    continue
                nd = du + data[p]
                if nd < row[v]:
                    row[v] = nd
                    idx = heap_size
                    heap_size += 1
                    while idx > 0:
                        parent = (idx - 1) >> 1
                        if heap_d[parent] <= nd:
                            break
                        heap_d[idx] = heap_d[parent]
                        heap_v[idx] = heap_v[parent]
                        idx = parent
                    heap_d[idx] = nd
                    heap_v[idx] = v

    return dist