from typing import Any
import numpy as np

# Try to load the Cython implementation if available
try:
    from articulation import articulation_points_cython as _ap_cython
except Exception:  # pragma: no cover - fallback to numba path
    _ap_cython = None

# Numba fallback
try:
    from numba import njit
except Exception:  # In case numba is unavailable (shouldn't happen), set njit to identity
    def njit(*args, **kwargs):  # type: ignore
        def wrap(f):
            return f
        return wrap

@njit(cache=True)
def _articulation_points_numba(n: int, edges: np.ndarray) -> np.ndarray:
    """
    Numba-compiled iterative Tarjan algorithm using CSR adjacency.
    edges: (m,2) int32 array with undirected edges (u,v)
    Returns: int32 array of articulation points in ascending order.
    """
    m = edges.shape[0]
    # Degree computation
    deg = np.zeros(n, dtype=np.int32)
    for i in range(m):
        u = edges[i, 0]
        v = edges[i, 1]
        deg[u] += 1
        deg[v] += 1

    # Build CSR
    row_ptr = np.empty(n + 1, dtype=np.int32)
    row_ptr[0] = 0
    for i in range(n):
        row_ptr[i + 1] = row_ptr[i] + deg[i]
    total = row_ptr[n]
    col_idx = np.empty(total, dtype=np.int32)
    nxt = row_ptr.copy()
    for i in range(m):
        u = edges[i, 0]
        v = edges[i, 1]
        pos = nxt[u]
        col_idx[pos] = v
        nxt[u] = pos + 1
        pos = nxt[v]
        col_idx[pos] = u
        nxt[v] = pos + 1

    # Tarjan iterative DFS
    disc = np.zeros(n, dtype=np.int32)
    low = np.zeros(n, dtype=np.int32)
    parent = np.full(n, -1, dtype=np.int32)
    child = np.zeros(n, dtype=np.int32)
    is_art = np.zeros(n, dtype=np.uint8)
    idx = np.zeros(n, dtype=np.int32)

    stack = np.empty(n, dtype=np.int32)
    sp = 0
    time = 1

    for s in range(n):
        if disc[s] != 0:
            continue

        # Fast path: isolated vertex
        if row_ptr[s] == row_ptr[s + 1]:
            disc[s] = time
            low[s] = time
            time += 1
            continue

        parent[s] = -1
        disc[s] = time
        low[s] = time
        time += 1
        idx[s] = row_ptr[s]
        stack[sp] = s
        sp += 1

        while sp > 0:
            u = stack[sp - 1]
            start = idx[u]
            end = row_ptr[u + 1]
            pu = parent[u]
            du = disc[u]

            if start < end:
                v = col_idx[start]
                idx[u] = start + 1

                if v == pu:
                    continue

                dv = disc[v]
                if dv == 0:
                    parent[v] = u
                    child[u] += 1
                    disc[v] = time
                    low[v] = time
                    time += 1
                    idx[v] = row_ptr[v]
                    stack[sp] = v
                    sp += 1
                else:
                    # Only consider back-edges to ancestors
                    if dv < du:
                        lu = low[u]
                        if dv < lu:
                            low[u] = dv
            else:
                # backtrack
                sp -= 1
                if pu != -1:
                    lu = low[u]
                    lp = low[pu]
                    if lu < lp:
                        low[pu] = lu
                    # Non-root articulation condition
                    if parent[pu] != -1 and lu >= disc[pu]:
                        is_art[pu] = 1
                else:
                    # Root articulation condition
                    if child[u] >= 2:
                        is_art[u] = 1

    # Pack result in ascending order
    cnt = 0
    for i in range(n):
        if is_art[i] != 0:
            cnt += 1
    res = np.empty(cnt, dtype=np.int32)
    j = 0
    for i in range(n):
        if is_art[i] != 0:
            res[j] = i
            j += 1
    return res

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Find articulation points in an undirected graph using a highly optimized
        iterative DFS (Tarjan's algorithm) compiled with Cython if available,
        otherwise with Numba.
        Input:
          - problem: dict with keys:
              - "num_nodes": int
              - "edges": list of [u, v] with 0 <= u < v < num_nodes
        Output:
          - dict with key "articulation_points": sorted list of articulation point node indices
        """
        n = int(problem.get("num_nodes", 0))
        if n <= 2:
            return {"articulation_points": []}

        edges_list = problem.get("edges", [])
        if not edges_list:
            return {"articulation_points": []}

        edges_arr = np.asarray(edges_list, dtype=np.int32)
        if edges_arr.ndim != 2 or edges_arr.shape[1] != 2:
            return {"articulation_points": []}

        if _ap_cython is not None:
            # Ensure contiguity for memoryviews
            edges_arr = np.ascontiguousarray(edges_arr, dtype=np.int32)
            ap = _ap_cython(n, edges_arr)
            return {"articulation_points": ap.tolist()}

        ap = _articulation_points_numba(n, edges_arr)
        return {"articulation_points": ap.tolist()}