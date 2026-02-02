from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, floyd_warshall

# Module-scope SAT solver import and name selection (init time not counted).
from pysat.solvers import Solver as _SATSolver  # type: ignore

_SAT_NAME = "minicard"
try:
    _t = _SATSolver(name=_SAT_NAME)
    _t.delete()
except Exception:
    _SAT_NAME = "MiniCard"

def _build_csr_from_dict(G: dict[str, dict[str, float]], nodes: list[str]) -> csr_matrix:
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for v, adj in G.items():
        i = idx[v]
        for w, d in adj.items():
            j = idx[w]
            if i < j:
                dd = float(d)
                rows.append(i)
                cols.append(j)
                data.append(dd)
                rows.append(j)
                cols.append(i)
                data.append(dd)

    if not rows:
        return csr_matrix((n, n), dtype=np.float64)

    return csr_matrix((np.asarray(data, dtype=np.float64), (rows, cols)), shape=(n, n))

def _apsp_distances(G: dict[str, dict[str, float]], nodes: list[str]) -> np.ndarray:
    mat = _build_csr_from_dict(G, nodes)
    n = mat.shape[0]
    m2 = int(mat.nnz // 2)  # undirected edges count
    if n <= 120 or (n <= 220 and m2 > (n * n) // 12):
        return floyd_warshall(mat, directed=False)
    return dijkstra(mat, directed=False, return_predecessors=False)

def _farthest_first_centers(D: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    n = D.shape[0]
    if k >= n:
        centers = np.arange(n, dtype=np.int32)
        return centers, 0.0

    # Good UB: exact 1-center as start for small n, cheap start for large n.
    if n <= 220:
        first = int(np.argmin(np.max(D, axis=1)))
    else:
        first = 0

    centers = np.empty(k, dtype=np.int32)
    centers[0] = first

    min_to = D[:, first].copy()
    for t in range(1, k):
        nxt = int(np.argmax(min_to))
        centers[t] = nxt
        np.minimum(min_to, D[:, nxt], out=min_to)

    return centers, float(np.max(min_to))

class _Prepared:
    __slots__ = ("nodes", "D", "unique_dists", "sorted_dists", "sorted_vars")

    def __init__(self, nodes: list[str], D: np.ndarray, unique_dists: np.ndarray) -> None:
        self.nodes = nodes
        self.D = D
        self.unique_dists = unique_dists
        # Lazy, only if SAT fallback used.
        self.sorted_dists: np.ndarray | None = None
        self.sorted_vars: np.ndarray | None = None

def _prepare(G: dict[str, dict[str, float]]) -> _Prepared:
    nodes = list(G.keys())
    n = len(nodes)
    if n == 0:
        return _Prepared(nodes, np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64))

    D = _apsp_distances(G, nodes).astype(np.float64, copy=False)

    triu = np.triu_indices(n)
    upper = D[triu]
    upper = upper[np.isfinite(upper)]
    unique = np.unique(upper) if upper.size else np.asarray([0.0], dtype=np.float64)

    return _Prepared(nodes, D, unique)

def _ensure_sorted(prep: _Prepared) -> tuple[np.ndarray, np.ndarray]:
    sd = prep.sorted_dists
    sv = prep.sorted_vars
    if sd is not None and sv is not None:
        return sd, sv
    D = prep.D
    order = np.argsort(D, axis=1, kind="quicksort")
    sd = np.take_along_axis(D, order, axis=1)
    sv = order.astype(np.int32, copy=False) + 1
    prep.sorted_dists = sd
    prep.sorted_vars = sv
    return sd, sv

def _max_dist(D: np.ndarray, centers_idx: np.ndarray) -> float:
    return float(np.max(np.min(D[:, centers_idx], axis=1)))

def _bitset_kcover(D: np.ndarray, k: int, radius: float) -> list[int] | None:
    """
    Exact feasibility for a fixed radius using big-int bitsets.
    Returns chosen centers (0-based) or None.
    Intended for small k/n.
    """
    n = D.shape[0]
    r = float(radius)

    within = D <= r  # (vertex v, center u) boolean

    # Pack columns -> cover mask for each center u: which vertices are covered by choosing u.
    packed_cols = np.packbits(within, axis=0, bitorder="little")  # (nb, n)
    nb = packed_cols.shape[0]
    covers = [0] * n
    for u in range(n):
        covers[u] = int.from_bytes(packed_cols[:, u].tobytes(), "little", signed=False)

    # Pack rows -> candidate centers for each vertex v.
    packed_rows = np.packbits(within, axis=1, bitorder="little")  # (n, nb)
    row_masks = [0] * n
    cand_cnt = [0] * n
    for v in range(n):
        m = int.from_bytes(packed_rows[v].tobytes(), "little", signed=False)
        row_masks[v] = m
        cand_cnt[v] = m.bit_count()

    all_mask = (1 << n) - 1
    best_cache: dict[int, int] = {}

    def pick_vertex(rem: int) -> int:
        # Choose uncovered vertex with smallest number of candidate centers (most constrained).
        m = rem
        best_v = -1
        best_c = 1 << 30
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            c = cand_cnt[v]
            if c < best_c:
                best_c = c
                best_v = v
                if c <= 1:
                    break
        return best_v

    def dfs(rem: int, depth: int) -> list[int] | None:
        if rem == 0:
            return []
        if depth >= k:
            return None

        prev = best_cache.get(rem)
        if prev is not None and prev <= depth:
            return None
        best_cache[rem] = depth

        v = pick_vertex(rem)
        cm = row_masks[v]
        if cm == 0:
            return None

        # Order candidates by how much they reduce the remaining uncovered set (best-first).
        cand: list[tuple[int, int, int]] = []
        m = cm
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            m ^= lsb
            new_rem = rem & ~covers[u]
            cand.append((new_rem.bit_count(), u, new_rem))
        cand.sort(key=lambda x: x[0])

        for _, u, new_rem in cand:
            if new_rem == rem:
                continue
            res = dfs(new_rem, depth + 1)
            if res is not None:
                return [u, *res]
        return None

    return dfs(all_mask, 0)

class _IncSAT:
    __slots__ = ("n", "sat", "sd", "sv", "last_t")

    def __init__(self, prep: _Prepared, k: int) -> None:
        sd, sv = _ensure_sorted(prep)
        n = len(prep.nodes)
        self.n = n
        self.sd = sd
        self.sv = sv
        self.last_t = np.full(n, n + 1, dtype=np.int32)

        sat = _SATSolver(name=_SAT_NAME)
        sat.add_atmost(list(range(1, n + 1)), k)
        self.sat = sat

    def tighten(self, radius: float) -> bool:
        r = float(radius)
        sd = self.sd
        sv = self.sv
        last = self.last_t
        n = self.n

        for v in range(n):
            t = int(np.searchsorted(sd[v], r, side="right"))
            if t <= 0:
                return False
            if t < last[v]:
                last[v] = t
                # Pass numpy array slice directly (avoid tolist()).
                self.sat.add_clause(sv[v, :t])

        return True

    def solve(self) -> list[int] | None:
        if not self.sat.solve():
            return None
        model = self.sat.get_model()
        if model is None:
            return None
        n = self.n
        chosen: list[int] = []
        for lit in model:
            if 1 <= lit <= n:
                chosen.append(lit - 1)
        return chosen

    def delete(self) -> None:
        self.sat.delete()

class Solver:
    def solve(self, problem: tuple[dict[str, dict[str, float]], int], **kwargs) -> Any:
        G, k = problem
        if k <= 0:
            return []

        prep = _prepare(G)
        n = len(prep.nodes)
        if n == 0:
            return []
        if k >= n:
            return prep.nodes[:]

        if k == 1:
            v = int(np.argmin(np.max(prep.D, axis=1)))
            return [prep.nodes[v]]

        heur_idx, ub = _farthest_first_centers(prep.D, k)
        if ub <= 0.0:
            return [prep.nodes[int(i)] for i in heur_idx]

        uniq = prep.unique_dists
        hi = int(np.searchsorted(uniq, ub, side="right")) - 1
        if hi < 0:
            return [prep.nodes[int(i)] for i in heur_idx]

        # Fast exact path for small instances: bitset k-cover + binary search.
        if k <= 6 and n <= 220:
            lo = 0
            best: list[int] | None = None
            while lo < hi:
                mid = (lo + hi) // 2
                sol = _bitset_kcover(prep.D, k, float(uniq[mid]))
                if sol is not None:
                    best = sol
                    hi = mid
                else:
                    lo = mid + 1
            sol = best if best is not None else _bitset_kcover(prep.D, k, float(uniq[lo]))
            if sol is None:
                return [prep.nodes[int(i)] for i in heur_idx]
            return [prep.nodes[i] for i in sol]

        # Fallback: incremental SAT tightening.
        pos = hi - 1
        if pos < 0:
            return [prep.nodes[int(i)] for i in heur_idx]

        inc = _IncSAT(prep, k)
        best2 = heur_idx.astype(np.int32, copy=False)

        try:
            if not inc.tighten(float(uniq[pos])):
                return [prep.nodes[int(i)] for i in best2]

            while True:
                sol = inc.solve()
                if sol is None:
                    break

                best2 = np.asarray(sol, dtype=np.int32)
                obj = _max_dist(prep.D, best2)

                pos = int(np.searchsorted(uniq, obj, side="left")) - 1
                if pos < 0:
                    break
                if not inc.tighten(float(uniq[pos])):
                    break
        finally:
            inc.delete()

        return [prep.nodes[int(i)] for i in best2]