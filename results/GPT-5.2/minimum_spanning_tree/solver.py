from __future__ import annotations

# pylint: disable=not-callable

from operator import itemgetter
from typing import Any

import numpy as np

class Solver:
    _kruskal_numba = None
    _numba_inited = False

    def __init__(self) -> None:
        # Init/compile time isn't counted.
        if Solver._numba_inited:
            return
        Solver._numba_inited = True

        try:
            from numba import njit  # type: ignore
        except Exception:
            Solver._kruskal_numba = None
            return

        @njit(cache=True)  # type: ignore[misc]
        def _kruskal(n: int, u: np.ndarray, v: np.ndarray, idx: np.ndarray) -> np.ndarray:
            parent = np.empty(n, dtype=np.int64)
            size = np.ones(n, dtype=np.int64)
            for i in range(n):
                parent[i] = i

            out = np.empty(n - 1 if n > 0 else 0, dtype=np.int64)
            cnt = 0

            for t in range(idx.shape[0]):
                ei = idx[t]
                a = u[ei]
                b = v[ei]

                ra = a
                while True:
                    pa = parent[ra]
                    if pa == ra:
                        break
                    ga = parent[pa]
                    parent[ra] = ga
                    ra = ga

                rb = b
                while True:
                    pb = parent[rb]
                    if pb == rb:
                        break
                    gb = parent[pb]
                    parent[rb] = gb
                    rb = gb

                if ra == rb:
                    continue

                if size[ra] < size[rb]:
                    tmp = ra
                    ra = rb
                    rb = tmp

                parent[rb] = ra
                size[ra] += size[rb]
                out[cnt] = ei
                cnt += 1
                if cnt == n - 1:
                    break

            return out[:cnt]

        # Warm-up compile
        _kruskal(1, np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64))
        Solver._kruskal_numba = _kruskal

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list[list[Any]]]:
        """
        Fast MST via Kruskal, matching the reference (NetworkX) output exactly.

        Replicates:
        - nx.Graph duplicate collapse: last weight wins, first insertion fixes edge order.
        - Edge iteration order: u ascending, neighbors insertion order, emit only u < v.
        - Kruskal: stable sort by weight (ties broken by edge iteration order).
        - Output: [u, v, weight] with u < v, sorted by (u, v).

        Note: weights are standardized to Python floats (problem spec says float).
        """
        n = int(problem["num_nodes"])
        edges_in = problem["edges"]
        if n <= 1 or len(edges_in) == 0:
            return {"mst_edges": []}

        nn = n
        fwd_v: list[list[int]] = [[] for _ in range(n)]
        fwd_w: list[list[float]] = [[] for _ in range(n)]
        idx_map: dict[int, int] = {}
        get = idx_map.get
        mask = 0xFFFFFFFF

        m = 0
        for e in edges_in:
            u = int(e[0])
            v = int(e[1])
            if u == v:
                continue
            if u > v:
                u, v = v, u
            key = u * nn + v
            packed = get(key)
            w = float(e[2])
            if packed is None:
                vs = fwd_v[u]
                ws = fwd_w[u]
                pos = len(vs)
                vs.append(v)
                ws.append(w)
                idx_map[key] = (u << 32) | pos
                m += 1
            else:
                uu = packed >> 32
                pos = packed & mask
                fwd_w[uu][pos] = w

        if m == 0:
            return {"mst_edges": []}

        kr = Solver._kruskal_numba

        # Python path for small instances (avoids numpy/numba overhead).
        if kr is None or m <= 768:
            items: list[tuple[float, int, int]] = []
            items_append = items.append
            for u in range(n):
                vs = fwd_v[u]
                if vs:
                    ws = fwd_w[u]
                    for j in range(len(vs)):
                        items_append((ws[j], u, vs[j]))
            items.sort(key=itemgetter(0))  # stable

            parent = list(range(n))
            size = [1] * n
            parent_l = parent
            size_l = size

            def find(x: int) -> int:
                while True:
                    px = parent_l[x]
                    if px == x:
                        return x
                    gx = parent_l[px]
                    parent_l[x] = gx
                    x = gx

            mst: list[list[Any]] = []
            mst_append = mst.append
            target = n - 1
            for w, u, v in items:
                ru = find(u)
                rv = find(v)
                if ru == rv:
                    continue
                if size_l[ru] < size_l[rv]:
                    ru, rv = rv, ru
                parent_l[rv] = ru
                size_l[ru] += size_l[rv]
                mst_append([u, v, w])
                if len(mst) == target:
                    break

            mst.sort(key=itemgetter(0, 1))
            return {"mst_edges": mst}

        # NumPy/Numba path
        u_arr = np.empty(m, dtype=np.int64)
        v_arr = np.empty(m, dtype=np.int64)
        w_val = np.empty(m, dtype=np.float64)

        k = 0
        for u in range(n):
            vs = fwd_v[u]
            if vs:
                ws = fwd_w[u]
                for j in range(len(vs)):
                    u_arr[k] = u
                    v_arr[k] = vs[j]
                    w_val[k] = ws[j]
                    k += 1

        idx = np.argsort(w_val, kind="mergesort")  # stable
        chosen = kr(n, u_arr, v_arr, idx)

        mst = [[int(u_arr[i]), int(v_arr[i]), float(w_val[i])] for i in chosen]
        mst.sort(key=itemgetter(0, 1))
        return {"mst_edges": mst}