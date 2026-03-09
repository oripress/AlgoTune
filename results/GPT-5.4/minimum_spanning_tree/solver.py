from typing import Any

import numpy as np
from numba import njit

_SMALL_M = 128

def _solve_small(n, edges):
    edge_info = {}
    next_order = [0] * n
    get_info = edge_info.get

    for e in edges:
        u = int(e[0])
        v = int(e[1])
        w = e[2]
        if u > v:
            u, v = v, u
        key = u * n + v
        info = get_info(key)
        if info is None:
            edge_info[key] = [next_order[u], w]
            next_order[u] += 1
        else:
            info[1] = w

    weighted_edges = []
    append_edge = weighted_edges.append
    for key, info in edge_info.items():
        u = key // n
        v = key - u * n
        if u != v:
            append_edge((info[1], u, info[0], v))

    weighted_edges.sort()

    parent = [-1] * n
    mst_edges = []
    append_mst = mst_edges.append
    remaining = n - 1

    for w, u, _order, v in weighted_edges:
        x = u
        while parent[x] >= 0:
            px = parent[x]
            if parent[px] >= 0:
                parent[x] = parent[px]
            x = px

        y = v
        while parent[y] >= 0:
            py = parent[y]
            if parent[py] >= 0:
                parent[y] = parent[py]
            y = py

        if x != y:
            if parent[x] > parent[y]:
                x, y = y, x
            parent[x] += parent[y]
            parent[y] = x
            append_mst([u, v, w])
            remaining -= 1
            if remaining == 0:
                break

    mst_edges.sort()
    return {"mst_edges": mst_edges}

@njit
def _dedup_edge_order(n, u_arr, v_arr, w_arr):
    m = len(w_arr)

    size = 1
    target = m * 4
    while size < target:
        size <<= 1
    mask = size - 1

    keys = np.full(size, -1, dtype=np.int64)
    vals = np.empty(size, dtype=np.int64)
    vals.fill(-1)

    head = np.full(n, -1, dtype=np.int64)
    tail = np.full(n, -1, dtype=np.int64)
    nxt = np.full(m, -1, dtype=np.int64)

    tmp_u = np.empty(m, dtype=np.int64)
    tmp_v = np.empty(m, dtype=np.int64)
    tmp_w = np.empty(m, dtype=np.float64)

    k = 0
    for i in range(m):
        u = int(u_arr[i])
        v = int(v_arr[i])
        if u > v:
            u, v = v, u

        key = u * n + v
        j = (key * 1315423911) & mask

        while True:
            kk = keys[j]
            if kk == -1:
                keys[j] = key
                if u != v:
                    vals[j] = k
                    tmp_u[k] = u
                    tmp_v[k] = v
                    tmp_w[k] = w_arr[i]

                    t = tail[u]
                    if t >= 0:
                        nxt[t] = k
                    else:
                        head[u] = k
                    tail[u] = k
                    k += 1
                else:
                    vals[j] = -2
                break

            if kk == key:
                p = vals[j]
                if p >= 0:
                    tmp_w[p] = w_arr[i]
                break

            j = (j + 1) & mask

    out_u = np.empty(k, dtype=np.int64)
    out_v = np.empty(k, dtype=np.int64)
    out_w = np.empty(k, dtype=np.float64)

    p = 0
    for u in range(n):
        idx = head[u]
        while idx >= 0:
            out_u[p] = tmp_u[idx]
            out_v[p] = tmp_v[idx]
            out_w[p] = tmp_w[idx]
            p += 1
            idx = nxt[idx]

    return out_u, out_v, out_w

@njit
def _kruskal_choose_idx(n, u_arr, v_arr, idx_arr):
    m = len(idx_arr)
    parent = np.full(n, -1, dtype=np.int64)
    chosen = np.empty(m, dtype=np.int64)

    c = 0
    remaining = n - 1

    for t in range(m):
        i = idx_arr[t]

        x = u_arr[i]
        while parent[x] >= 0:
            px = parent[x]
            if parent[px] >= 0:
                parent[x] = parent[px]
            x = px

        y = v_arr[i]
        while parent[y] >= 0:
            py = parent[y]
            if parent[py] >= 0:
                parent[y] = parent[py]
            y = py

        if x != y:
            if parent[x] > parent[y]:
                x, y = y, x
            parent[x] += parent[y]
            parent[y] = x
            chosen[c] = i
            c += 1
            remaining -= 1
            if remaining == 0:
                break

    return chosen[:c]

class Solver:
    def __init__(self):
        u = np.array([0], dtype=np.int64)
        v = np.array([0], dtype=np.int64)
        w = np.array([0.0], dtype=np.float64)
        idx = np.array([0], dtype=np.int64)
        _dedup_edge_order(1, u, v, w)
        _kruskal_choose_idx(1, u, v, idx)

    def solve(self, problem, **kwargs) -> Any:
        n = int(problem["num_nodes"])
        edges = problem["edges"]
        m = len(edges)

        if n <= 1 or m == 0:
            return {"mst_edges": []}

        if m <= _SMALL_M:
            return _solve_small(n, edges)

        arr = np.asarray(edges)
        if arr.ndim != 2 or arr.shape[1] != 3:
            arr = np.array(list(edges), dtype=np.float64)

        u_arr = arr[:, 0].astype(np.int64, copy=False)
        v_arr = arr[:, 1].astype(np.int64, copy=False)
        w_arr = arr[:, 2].astype(np.float64, copy=False)

        u, v, w = _dedup_edge_order(n, u_arr, v_arr, w_arr)
        if len(w) == 0:
            return {"mst_edges": []}

        sort_idx = np.argsort(w, kind="mergesort")
        chosen = _kruskal_choose_idx(n, u, v, sort_idx)

        key_idx = np.argsort(u[chosen] * n + v[chosen])
        mst_edges = [
            [int(u[chosen[i]]), int(v[chosen[i]]), float(w[chosen[i]])]
            for i in key_idx
        ]
        return {"mst_edges": mst_edges}