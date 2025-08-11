from typing import Any, Dict

# Optional Numba-accelerated DSU
NUMBA_AVAILABLE = False
try:
    import numpy as np  # type: ignore
    import numba as nb  # type: ignore

    NUMBA_AVAILABLE = True

    @nb.njit(cache=True, fastmath=True)
    def _cc_numba(n: int, edges: np.ndarray) -> int:
        # DSU with single array: par[i] < 0 -> root with size = -par[i]
        par = np.empty(n, dtype=np.int64)
        for i in range(n):
            par[i] = -1
        comp = n

        for k in range(edges.shape[0]):
            ui = edges[k, 0]
            vi = edges[k, 1]

            # Early detect out-of-range nodes; signal sentinel -1
            if ui < 0 or vi < 0 or ui >= n or vi >= n:
                return -1

            # find(ui) with path halving
            x = ui
            while par[x] >= 0:
                px = par[x]
                if par[px] >= 0:
                    par[x] = par[px]
                x = px
            ra = x

            # find(vi) with path halving
            x = vi
            while par[x] >= 0:
                px = par[x]
                if par[px] >= 0:
                    par[x] = par[px]
                x = px
            rb = x

            if ra != rb:
                # union by size (remember: par[root] is negative size)
                if par[ra] > par[rb]:
                    tmp = ra
                    ra = rb
                    rb = tmp
                par[ra] += par[rb]
                par[rb] = ra
                comp -= 1
                # Early exit when fully connected
                if comp == 1:
                    return comp

        return comp

    # Warm up JIT at import time to avoid runtime compilation cost
    try:
        _ = _cc_numba(1, np.empty((0, 2), dtype=np.int64))
    except Exception:
        pass

except Exception:
    NUMBA_AVAILABLE = False
    np = None  # type: ignore

# Ensure numpy is available independently of numba
try:
    if 'np' not in globals() or np is None:  # type: ignore[name-defined]
        import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, int]:
        try:
            bn = int(problem.get("num_nodes", 0))
            edges_obj = problem.get("edges")
            if edges_obj is None:
                return {"number_connected_components": bn}

            # Numba-accelerated path for any input if available
            if NUMBA_AVAILABLE and np is not None:
                try:
                    if isinstance(edges_obj, np.ndarray):
                        arr = edges_obj
                        if arr.size == 0:
                            return {"number_connected_components": bn}
                        if not (arr.ndim == 2 and arr.shape[1] == 2):
                            try:
                                arr = arr.reshape(-1, 2)
                            except Exception:
                                arr = np.asarray(arr, dtype=np.int64)
                                if not (arr.ndim == 2 and arr.shape[1] == 2):
                                    arr = None  # type: ignore
                    else:
                        arr = np.asarray(edges_obj, dtype=np.int64)  # type: ignore
                        if arr.size == 0:
                            return {"number_connected_components": bn}
                        if not (arr.ndim == 2 and arr.shape[1] == 2):
                            arr = None  # type: ignore

                    if arr is not None:
                        if arr.dtype != np.int64:
                            arr = arr.astype(np.int64, copy=False)

                        # First attempt: assume all labels within [0, bn); compiled code detects OOB
                        comp = _cc_numba(bn, arr)  # type: ignore
                        if comp >= 0:
                            return {"number_connected_components": int(comp)}

                        # Out-of-range labels exist: map them to contiguous new indices and retry
                        flat = arr.reshape(-1)
                        mask = (flat < 0) | (flat >= bn)
                        ext = flat[mask]
                        if ext.size:
                            uniq = np.unique(ext)
                            idxs = np.searchsorted(uniq, ext)
                            mapped_flat = flat.copy()
                            mapped_flat[mask] = bn + idxs
                            mapped = mapped_flat.reshape(arr.shape[0], 2)
                            total_n = bn + int(uniq.size)
                        else:
                            mapped = arr
                            total_n = bn
                        comp2 = _cc_numba(total_n, mapped)  # type: ignore
                        return {"number_connected_components": int(comp2)}
                except Exception:
                    # Fall back to Python DSU if any conversion/numba error occurs
                    pass

            # Fallback: fast Python DSU (single-array parent-or-size)
            edges = edges_obj
            # Efficiently handle empty for common types
            if isinstance(edges, (list, tuple)):
                if len(edges) == 0:
                    return {"number_connected_components": bn}
            elif np is not None and isinstance(edges, np.ndarray):  # type: ignore
                if edges.size == 0:
                    return {"number_connected_components": bn}

            # Unified general path with dynamic mapping for any out-of-range labels.
            p = [-1] * bn  # parent-or-size
            comp = bn
            idx: Dict[int, int] = {}
            idx_get = idx.get
            n0 = bn  # local alias

            for u, v in edges:
                # Map u
                if 0 <= u < n0:
                    ui = u
                else:
                    j = idx_get(u)
                    if j is None:
                        j = len(p)
                        idx[u] = j
                        p.append(-1)
                        comp += 1
                    ui = j

                # Map v
                if 0 <= v < n0:
                    vi = v
                else:
                    j = idx_get(v)
                    if j is None:
                        j = len(p)
                        idx[v] = j
                        p.append(-1)
                        comp += 1
                    vi = j

                # find(ui) with path halving
                x = ui
                while p[x] >= 0:
                    px = p[x]
                    if p[px] >= 0:
                        p[x] = p[px]
                    x = px
                ra = x

                # find(vi) with path halving
                x = vi
                while p[x] >= 0:
                    px = p[x]
                    if p[px] >= 0:
                        p[x] = p[px]
                    x = px
                rb = x

                # Union by size
                if ra != rb:
                    if p[ra] > p[rb]:
                        ra, rb = rb, ra
                    p[ra] += p[rb]
                    p[rb] = ra
                    comp -= 1

            return {"number_connected_components": comp}
        except Exception:
            # Use -1 as an unmistakable “solver errored” sentinel
            return {"number_connected_components": -1}