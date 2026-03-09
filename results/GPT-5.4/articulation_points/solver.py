from numbers import Integral
from typing import Any

def _solve_python(n: int, edges: list[list[int]]) -> dict[str, list[int]]:
    m = len(edges)
    if n <= 2 or m == 0:
        return {"articulation_points": []}

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    disc = [-1] * n
    low = [0] * n
    parent = [-1] * n
    is_ap = [0] * n

    time = 0
    stack_nodes: list[int] = []
    stack_idx: list[int] = []

    for root in range(n):
        if disc[root] != -1:
            continue

        disc[root] = time
        low[root] = time
        time += 1
        root_children = 0

        stack_nodes.append(root)
        stack_idx.append(0)

        while stack_nodes:
            u = stack_nodes[-1]
            idx = stack_idx[-1]
            nbrs = adj[u]

            if idx < len(nbrs):
                v = nbrs[idx]
                stack_idx[-1] = idx + 1

                if disc[v] == -1:
                    parent[v] = u
                    if u == root:
                        root_children += 1
                    disc[v] = time
                    low[v] = time
                    time += 1
                    stack_nodes.append(v)
                    stack_idx.append(0)
                elif v != parent[u]:
                    dv = disc[v]
                    if dv < low[u]:
                        low[u] = dv
            else:
                stack_nodes.pop()
                stack_idx.pop()
                p = parent[u]
                if p != -1:
                    lu = low[u]
                    if lu < low[p]:
                        low[p] = lu
                    if lu >= disc[p]:
                        is_ap[p] = 1

        is_ap[root] = 1 if root_children > 1 else 0

    return {"articulation_points": [i for i, flag in enumerate(is_ap) if flag]}

class Solver:
    def __init__(self) -> None:
        try:
            from articulation_cy import APSolver

            self._cython_articulation = APSolver().articulation_points
        except Exception:
            self._cython_articulation = None

    def solve(self, problem, **kwargs) -> Any:
        n = int(problem["num_nodes"])
        edges = problem["edges"]
        m = len(edges)

        if m:
            u0 = edges[0][0]
            if not isinstance(u0, Integral):
                edges = [[int(u), int(v)] for u, v in edges]

        cy = self._cython_articulation
        if cy is not None:
            return {"articulation_points": cy(n, edges)}

        return _solve_python(n, edges)