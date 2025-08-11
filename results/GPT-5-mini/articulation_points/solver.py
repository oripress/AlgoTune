from typing import Any

# Try to use the compiled Cython fast path if available; import numpy once to avoid per-call overhead
try:
    import art_cy  # compiled extension providing find_aps_csr
    import numpy as _np
    _HAVE_CY = True
except Exception:
    _HAVE_CY = False
    _np = None

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Find articulation points. Use the compiled Cython routine when available
        with a vectorized NumPy CSR builder, and an optimized iterative Tarjan
        implementation in pure Python otherwise.
        Returns {"articulation_points": sorted_list}.
        """
        n = int(problem.get("num_nodes", 0))
        edges = problem.get("edges", []) or []

        if n <= 0:
            return {"articulation_points": []}

        m_est = len(edges)

        # Use Cython path if available and we have edges
        if _HAVE_CY and m_est > 0:
            try:
                # Convert edge list to numpy array quickly; if shape invalid, fall back
                arr = _np.asarray(edges, dtype=_np.int64)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    raise ValueError("Invalid edges array shape")

                # Extract endpoints as intc for compatibility
                u = arr[:, 0].astype(_np.intc, copy=False)
                v = arr[:, 1].astype(_np.intc, copy=False)

                # Filter out-of-range endpoints (vectorized)
                mask = (u >= 0) & (u < n) & (v >= 0) & (v < n)
                if not mask.all():
                    u = u[mask]
                    v = v[mask]
                    if u.size == 0:
                        return {"articulation_points": []}

                # Degree per node via bincount (fast C implementation)
                deg_u = _np.bincount(u, minlength=n).astype(_np.intc)
                deg_v = _np.bincount(v, minlength=n).astype(_np.intc)
                deg = deg_u + deg_v

                # Build CSR row pointer via cumulative sum
                row_ptr = _np.empty(n + 1, dtype=_np.intc)
                row_ptr[0] = 0
                # deg.cumsum returns array of length n
                row_ptr[1:] = deg.cumsum(dtype=_np.intc)

                total = int(row_ptr[n])
                if total == 0:
                    return {"articulation_points": []}

                # Build directed edge lists (U -> V) then sort by U so neighbors per node are contiguous
                U = _np.concatenate((u, v)).astype(_np.intc, copy=False)
                V = _np.concatenate((v, u)).astype(_np.intc, copy=False)

                # argsort groups entries by U; it's a C implementation and fast for large arrays
                order = _np.argsort(U, kind="quicksort")
                cols = V[order].astype(_np.intc, copy=False)

                # Call the Cython routine which expects CSR arrays
                ap_list = art_cy.find_aps_csr(n, row_ptr, cols)
                ap_list.sort()
                return {"articulation_points": ap_list}
            except Exception:
                # On any error, fall back to pure Python implementation
                pass

        # Pure-Python fallback (optimized iterative Tarjan with parallel stacks)
        adj = [[] for _ in range(n)]
        for e in edges:
            if not e:
                continue
            u = int(e[0]); v = int(e[1])
            if 0 <= u < n and 0 <= v < n:
                adj[u].append(v)
                adj[v].append(u)

        disc = [-1] * n
        low = [0] * n
        parent = [-1] * n
        ap = [False] * n

        # Local aliases for speed
        adj_local = adj
        disc_local = disc
        low_local = low
        parent_local = parent
        ap_local = ap

        time = 0

        # Iterate over all nodes (handle disconnected graphs)
        for start in range(n):
            if disc_local[start] != -1:
                continue

            # Initialize iterative DFS stacks
            node_stack = [start]
            parent_stack = [-1]
            idx_stack = [0]
            child_stack = [0]

            disc_local[start] = time
            low_local[start] = time
            time += 1
            parent_local[start] = -1

            while node_stack:
                u = node_stack[-1]
                p = parent_stack[-1]
                idx = idx_stack[-1]
                children = child_stack[-1]
                neighbors = adj_local[u]

                if idx < len(neighbors):
                    v = neighbors[idx]
                    # advance neighbor index
                    idx_stack[-1] = idx + 1
                    if disc_local[v] == -1:
                        parent_local[v] = u
                        child_stack[-1] = children + 1
                        disc_local[v] = time
                        low_local[v] = time
                        time += 1
                        # push v onto stacks
                        node_stack.append(v)
                        parent_stack.append(u)
                        idx_stack.append(0)
                        child_stack.append(0)
                    elif v != p:
                        dv = disc_local[v]
                        if low_local[u] > dv:
                            low_local[u] = dv
                else:
                    # finished processing u
                    node_stack.pop()
                    parent_stack.pop()
                    idx_stack.pop()
                    child_stack.pop()
                    if p != -1:
                        if low_local[p] > low_local[u]:
                            low_local[p] = low_local[u]
                        if parent_local[p] != -1 and low_local[u] >= disc_local[p]:
                            ap_local[p] = True
                    else:
                        # u is root
                        if children > 1:
                            ap_local[u] = True

        result = [i for i, flag in enumerate(ap_local) if flag]
        result.sort()
        return {"articulation_points": result}