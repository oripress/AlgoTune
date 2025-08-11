# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False
import numpy as np
cimport numpy as cnp

ctypedef cnp.int32_t i32
ctypedef cnp.uint8_t u8

cpdef cnp.ndarray[i32, ndim=1] articulation_points_cython(int n, cnp.ndarray[i32, ndim=2] edges):
    """
    Cython-optimized iterative Tarjan algorithm using CSR adjacency and GIL-released loops.
    edges: (m,2) int32 array with undirected edges (u,v), 0 <= u < v < n
    Returns: int32 array of articulation points in ascending order.
    """
    cdef int m = edges.shape[0]

    # Views
    cdef i32[:, ::1] E = edges

    # Degree and CSR
    cdef cnp.ndarray[i32, ndim=1] deg_arr = np.empty(n, dtype=np.int32)
    cdef cnp.ndarray[i32, ndim=1] row_ptr = np.empty(n + 1, dtype=np.int32)
    cdef cnp.ndarray[i32, ndim=1] col_idx
    cdef cnp.ndarray[i32, ndim=1] nxt

    cdef i32[::1] deg = deg_arr
    cdef i32[::1] row = row_ptr
    cdef i32[::1] col
    cdef i32[::1] nextp

    cdef int i
    cdef i32 u, v, pos
    cdef int total

    # Compute degrees and prefix sums
    with nogil:
        row[0] = 0
        for i in range(n):
            deg[i] = 0
        for i in range(m):
            u = E[i, 0]
            v = E[i, 1]
            deg[u] += 1
            deg[v] += 1
        for i in range(n):
            row[i + 1] = row[i] + deg[i]

    total = row[n]
    col_idx = np.empty(total, dtype=np.int32)
    nxt = row_ptr.copy()
    col = col_idx
    nextp = nxt

    # Fill CSR neighbors
    with nogil:
        for i in range(m):
            u = E[i, 0]
            v = E[i, 1]
            pos = nextp[u]
            col[pos] = v
            nextp[u] = pos + 1
            pos = nextp[v]
            col[pos] = u
            nextp[v] = pos + 1

    # Tarjan iterative DFS arrays
    cdef cnp.ndarray[i32, ndim=1] disc_arr = np.zeros(n, dtype=np.int32)
    cdef cnp.ndarray[i32, ndim=1] low_arr = np.zeros(n, dtype=np.int32)
    cdef cnp.ndarray[i32, ndim=1] parent_arr = np.empty(n, dtype=np.int32)  # will set per-visit
    cdef cnp.ndarray[u8, ndim=1] is_art_arr = np.zeros(n, dtype=np.uint8)
    cdef cnp.ndarray[i32, ndim=1] idx_arr = np.empty(n, dtype=np.int32)
    cdef cnp.ndarray[i32, ndim=1] stack_arr = np.empty(n, dtype=np.int32)

    cdef i32[::1] disc = disc_arr
    cdef i32[::1] low = low_arr
    cdef i32[::1] parent = parent_arr
    cdef u8[::1] is_art = is_art_arr
    cdef i32[::1] idx = idx_arr
    cdef i32[::1] stack = stack_arr

    cdef int sp = 0
    cdef i32 time = 1
    cdef int start, end
    cdef i32 pu, du, dv, lu, lp
    cdef int s
    cdef i32 root_children

    # DFS over all components
    with nogil:
        for s in range(n):
            if disc[s] != 0:
                continue

            # fast path: isolated vertex
            if row[s] == row[s + 1]:
                disc[s] = time
                low[s] = time
                time += 1
                continue

            parent[s] = -1
            disc[s] = time
            low[s] = time
            time += 1
            idx[s] = row[s]
            stack[sp] = s
            sp += 1
            root_children = 0

            while sp > 0:
                u = stack[sp - 1]
                start = idx[u]
                end = row[u + 1]
                pu = parent[u]
                du = disc[u]

                if start < end:
                    v = col[start]
                    idx[u] = start + 1

                    if v == pu:
                        continue

                    dv = disc[v]
                    if dv == 0:
                        parent[v] = u
                        if u == s:
                            root_children += 1
                        disc[v] = time
                        low[v] = time
                        time += 1
                        idx[v] = row[v]
                        stack[sp] = v
                        sp += 1
                    else:
                        if dv < du and dv < low[u]:
                            low[u] = dv
                else:
                    sp -= 1
                    if pu != -1:
                        lu = low[u]
                        lp = low[pu]
                        if lu < lp:
                            low[pu] = lu
                        # Non-root articulation condition
                        if pu != s and lu >= disc[pu]:
                            is_art[pu] = 1
                    else:
                        # Root articulation condition
                        if root_children >= 2:
                            is_art[u] = 1

    # Pack results in ascending order
    cdef int cnt = 0
    for i in range(n):
        if is_art[i] != 0:
            cnt += 1

    cdef cnp.ndarray[i32, ndim=1] res = np.empty(cnt, dtype=np.int32)
    cdef i32[::1] resv = res
    cdef int j = 0
    for i in range(n):
        if is_art[i] != 0:
            resv[j] = i
            j += 1
    return res