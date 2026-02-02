from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numba import njit

    _NUMBA_OK = True
except Exception:  # pragma: no cover
    njit = None  # type: ignore[assignment]
    _NUMBA_OK = False

if _NUMBA_OK:

    @njit(cache=True)
    def _articulation_points_forward_star(n: int, e: np.ndarray) -> np.ndarray:
        """
        Build forward-star adjacency (head/to/nxt) and compute articulation points
        via iterative Tarjan DFS. Returns sorted vertex ids.
        """
        m = e.shape[0]

        head = np.empty(n, np.int32)
        disc = np.empty(n, np.int32)
        low = np.empty(n, np.int32)
        parent = np.empty(n, np.int32)
        child = np.zeros(n, np.int32)
        ap = np.zeros(n, np.uint8)
        for i in range(n):
            head[i] = -1
            disc[i] = -1
            parent[i] = -1

        to = np.empty(m * 2, np.int32)
        nxt = np.empty(m * 2, np.int32)
        ei = 0
        for i in range(m):
            a = e[i, 0]
            b = e[i, 1]

            to[ei] = b
            nxt[ei] = head[a]
            head[a] = ei
            ei += 1

            to[ei] = a
            nxt[ei] = head[b]
            head[b] = ei
            ei += 1

        stack_node = np.empty(n, np.int32)
        stack_edge = np.empty(n, np.int32)  # current edge iterator for each stack frame
        top = -1
        t = 0

        for start in range(n):
            if disc[start] != -1:
                continue

            hs = head[start]
            if hs == -1:
                disc[start] = t
                low[start] = t
                t += 1
                continue

            disc[start] = t
            low[start] = t
            t += 1

            top += 1
            stack_node[top] = start
            stack_edge[top] = hs

            while top >= 0:
                u0 = stack_node[top]
                eidx = stack_edge[top]

                if eidx != -1:
                    v0 = to[eidx]
                    stack_edge[top] = nxt[eidx]  # advance iterator

                    if disc[v0] == -1:
                        parent[v0] = u0
                        child[u0] += 1

                        disc[v0] = t
                        low[v0] = t
                        t += 1

                        top += 1
                        stack_node[top] = v0
                        stack_edge[top] = head[v0]
                    elif v0 != parent[u0]:
                        dv = disc[v0]
                        if dv < low[u0]:
                            low[u0] = dv
                else:
                    # finish u0
                    top -= 1
                    p = parent[u0]
                    if p == -1:
                        if child[u0] > 1:
                            ap[u0] = 1
                    else:
                        lu = low[u0]
                        if lu < low[p]:
                            low[p] = lu
                        if parent[p] != -1 and lu >= disc[p]:
                            ap[p] = 1

        cnt = 0
        for i in range(n):
            if ap[i] != 0:
                cnt += 1
        out = np.empty(cnt, np.int32)
        k = 0
        for i in range(n):
            if ap[i] != 0:
                out[k] = i
                k += 1
        return out

    # Force compilation at import time (outside solve timing).
    _dummy_e = np.array([[0, 1]], dtype=np.int32)
    _ = _articulation_points_forward_star(2, _dummy_e)

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, list[int]]:
        n = int(problem["num_nodes"])
        edges = problem["edges"]
        m = len(edges)

        if n <= 1 or m == 0:
            return {"articulation_points": []}

        # Prefer Numba for all but very tiny graphs.
        if (not _NUMBA_OK) or m <= 64:
            adj: list[list[int]] = [[] for _ in range(n)]
            to_int = int
            for a0, b0 in edges:
                a = to_int(a0)
                b = to_int(b0)
                adj[a].append(b)
                adj[b].append(a)

            disc = [-1] * n
            low = [0] * n
            parent = [-1] * n
            child = [0] * n
            ap = [False] * n
            time = 0

            stack: list[int] = []
            it_idx: list[int] = []

            for start in range(n):
                if disc[start] != -1:
                    continue
                if not adj[start]:
                    disc[start] = time
                    low[start] = time
                    time += 1
                    continue

                disc[start] = time
                low[start] = time
                time += 1
                stack.append(start)
                it_idx.append(0)
                parent[start] = -1

                while stack:
                    u = stack[-1]
                    i = it_idx[-1]
                    au = adj[u]

                    if i < len(au):
                        v = au[i]
                        it_idx[-1] = i + 1

                        if disc[v] == -1:
                            parent[v] = u
                            child[u] += 1
                            disc[v] = time
                            low[v] = time
                            time += 1
                            stack.append(v)
                            it_idx.append(0)
                        elif v != parent[u]:
                            dv = disc[v]
                            lu = low[u]
                            if dv < lu:
                                low[u] = dv
                    else:
                        stack.pop()
                        it_idx.pop()

                        p = parent[u]
                        if p == -1:
                            if child[u] > 1:
                                ap[u] = True
                        else:
                            lu = low[u]
                            lp = low[p]
                            if lu < lp:
                                low[p] = lu
                            if parent[p] != -1 and lu >= disc[p]:
                                ap[p] = True

            return {"articulation_points": [i for i, flag in enumerate(ap) if flag]}

        # Numba path: avoid copies when possible.
        if isinstance(edges, np.ndarray):
            e = edges
            if e.dtype != np.int32:
                e = e.astype(np.int32, copy=False)
            if e.ndim != 2:
                e = e.reshape((-1, 2))
        else:
            e = np.asarray(edges, dtype=np.int32)
            if e.ndim != 2:
                e = e.reshape((-1, 2))

        out = _articulation_points_forward_star(n, e)
        return {"articulation_points": out.tolist()}