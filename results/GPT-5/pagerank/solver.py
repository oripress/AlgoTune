from typing import Any, Dict, List

import numpy as np

try:
    import numba as nb

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    @nb.njit(cache=True, fastmath=False)
    def _pagerank_iter(in_offsets, in_srcs, in_weights, x, alpha, z, y):
        n = y.shape[0]
        err = 0.0
        for j in range(n):
            acc = 0.0
            start = in_offsets[j]
            end = in_offsets[j + 1]
            for k in range(start, end):
                acc += x[in_srcs[k]] * in_weights[k]
            newval = alpha * acc + z
            y[j] = newval
            err += abs(newval - x[j])
        return err

class Solver:
    def __init__(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> None:
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def solve(self, problem: Dict[str, List[List[int]]], **kwargs: Any) -> Dict[str, List[float]]:
        adj_list = problem["adjacency_list"]
        n = len(adj_list)

        # Edge cases
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        alpha = self.alpha
        max_iter = self.max_iter
        tol = self.tol

        # Build forward edge arrays with deduplication (adjacency_list entries are sorted)
        srcs: List[int] = []
        dsts: List[int] = []
        deg = np.empty(n, dtype=np.int64)

        for u, nbrs in enumerate(adj_list):
            last = -1
            count = 0
            for v in nbrs:
                if v != last:
                    srcs.append(u)
                    dsts.append(v)
                    count += 1
                    last = v
            deg[u] = count

        if dsts:
            edges_src = np.asarray(srcs, dtype=np.int64)
            edges_dst = np.asarray(dsts, dtype=np.int64)
            m = edges_dst.size
        else:
            edges_src = np.empty(0, dtype=np.int64)
            edges_dst = np.empty(0, dtype=np.int64)
            m = 0

        # If no edges at all, stationary distribution is uniform
        if m == 0:
            uniform = [1.0 / n] * n
            return {"pagerank_scores": uniform}

        inv_deg = np.zeros(n, dtype=np.float64)
        nz_mask = deg > 0
        inv_deg[nz_mask] = 1.0 / deg[nz_mask]
        # Precompute per-edge weights scaled by alpha
        edge_w = inv_deg[edges_src] * alpha if m > 0 else np.empty(0, dtype=np.float64)

        # Dangling nodes indices for fast summation
        dangle_idx = np.flatnonzero(~nz_mask)

        # Initialize ranks uniformly
        n_inv = 1.0 / n
        x = np.full(n, n_inv, dtype=np.float64)

        # Heuristic: disable Numba by setting threshold extremely high to avoid JIT overhead
        use_numba = False and NUMBA_AVAILABLE and m >= 1_000_000_000

        if use_numba:
            # Build incoming-edge CSR structure from forward edges
            incounts = np.bincount(edges_dst, minlength=n).astype(np.int64, copy=False)
            in_offsets = np.empty(n + 1, dtype=np.int64)
            np.cumsum(incounts, out=in_offsets[1:])
            in_offsets[0] = 0

            in_srcs = np.empty(m, dtype=np.int64)
            cursor = in_offsets[:-1].copy()
            for e in range(m):
                v = edges_dst[e]
                u = edges_src[e]
                idx = cursor[v]
                in_srcs[idx] = u
                cursor[v] = idx + 1

            in_weights = inv_deg[in_srcs]

            y = np.empty_like(x)
            for _ in range(max_iter):
                dangle_contrib = float(x[dangle_idx].sum()) if dangle_idx.size > 0 else 0.0
                z = n_inv * ((1.0 - alpha) + alpha * dangle_contrib)
                err = _pagerank_iter(in_offsets, in_srcs, in_weights, x, alpha, z, y)
                x, y = y, x  # swap buffers
                if err < n * tol:
                    break

            pagerank_list = [float(val) for val in x.tolist()]
            return {"pagerank_scores": pagerank_list}

        # NumPy fast path using bincount on forward edges
        const_z = (1.0 - alpha) * n_inv if dangle_idx.size == 0 else None

        for _ in range(max_iter):
            if m > 0:
                y = np.bincount(
                    edges_dst,
                    weights=x[edges_src] * edge_w,
                    minlength=n,
                )
            else:
                y = np.zeros(n, dtype=np.float64)

            if const_z is None:
                dangle_sum = float(x[dangle_idx].sum()) if dangle_idx.size > 0 else 0.0
                z = (dangle_sum * alpha + (1.0 - alpha)) * n_inv
            else:
                z = const_z
            y += z

            err = float(np.abs(y - x).sum())
            x = y
            if err < n * tol:
                break

        pagerank_list = [float(val) for val in x.tolist()]
        return {"pagerank_scores": pagerank_list}