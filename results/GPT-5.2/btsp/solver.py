from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import numba as nb
except Exception:  # pragma: no cover
    nb = None  # type: ignore[assignment]

def _btsp_held_karp_minmax_py(dist: np.ndarray) -> np.ndarray:
    """Pure-Python exact bottleneck TSP via min-max Held–Karp DP (fallback)."""
    n = int(dist.shape[0])
    if n <= 1:
        return np.array([0, 0], dtype=np.int64)
    if n == 2:
        return np.array([0, 1, 0], dtype=np.int64)

    m = n - 1
    full_mask = (1 << m) - 1
    inf = float("inf")

    dp = [[inf] * m for _ in range(full_mask + 1)]
    parent = [[-1] * m for _ in range(full_mask + 1)]

    for j in range(m):
        dp[1 << j][j] = float(dist[0, j + 1])

    for mask in range(1, full_mask + 1):
        for j in range(m):
            bitj = 1 << j
            if (mask & bitj) == 0:
                continue
            prev = mask ^ bitj
            if prev == 0:
                continue
            best = inf
            bestp = -1
            for i in range(m):
                biti = 1 << i
                if (prev & biti) == 0:
                    continue
                val = dp[prev][i]
                w = float(dist[i + 1, j + 1])
                if w > val:
                    val = w
                if val < best:
                    best = val
                    bestp = i
            dp[mask][j] = best
            parent[mask][j] = bestp

    best_val = inf
    last = 0
    for j in range(m):
        val = dp[full_mask][j]
        w = float(dist[j + 1, 0])
        if w > val:
            val = w
        if val < best_val:
            best_val = val
            last = j

    tour = np.empty(n + 1, dtype=np.int64)
    tour[0] = 0
    tour[n] = 0
    idx = n - 1
    cur = last
    mask = full_mask
    while cur != -1:
        tour[idx] = cur + 1
        p = parent[mask][cur]
        mask ^= 1 << cur
        cur = p
        idx -= 1
    return tour

if nb is not None:

    @nb.njit(cache=True, nogil=True, fastmath=True)
    def _btsp_held_karp_minmax_inplace(dist: np.ndarray, dp: np.ndarray, tour: np.ndarray) -> None:
        """
        In-place exact bottleneck TSP DP without storing parent table.

        dist: (n,n) float64
        dp: (2^(n-1), n-1) float64 (workspace; only dp[mask,j] for j in mask is valid)
        tour: (n+1) int64 output
        """
        n = dist.shape[0]
        if n <= 1:
            tour[0] = 0
            tour[1] = 0
            return
        if n == 2:
            tour[0] = 0
            tour[1] = 1
            tour[2] = 0
            return

        m = n - 1
        full_mask = (1 << m) - 1
        inf = 1e300

        # base: 0 -> (j+1)
        for j in range(m):
            dp[1 << j, j] = dist[0, j + 1]

        # DP
        for mask in range(1, full_mask + 1):
            for j in range(m):
                bitj = 1 << j
                if (mask & bitj) == 0:
                    continue
                prev = mask ^ bitj
                if prev == 0:
                    continue

                best = inf
                for i in range(m):
                    biti = 1 << i
                    if (prev & biti) == 0:
                        continue
                    val = dp[prev, i]
                    w = dist[i + 1, j + 1]
                    if w > val:
                        val = w
                    if val < best:
                        best = val
                dp[mask, j] = best

        # close cycle to 0
        best_val = inf
        last = 0
        for j in range(m):
            val = dp[full_mask, j]
            w = dist[j + 1, 0]
            if w > val:
                val = w
            if val < best_val:
                best_val = val
                last = j

        # reconstruct by scanning dp (no parent array)
        tour[0] = 0
        tour[n] = 0

        idx = n - 1
        cur = last
        mask = full_mask
        eps = 1e-12
        while True:
            tour[idx] = cur + 1
            idx -= 1
            prev = mask ^ (1 << cur)
            if prev == 0:
                break

            target = dp[mask, cur] + eps
            nxt = 0
            for i in range(m):
                biti = 1 << i
                if (prev & biti) == 0:
                    continue
                val = dp[prev, i]
                w = dist[i + 1, cur + 1]
                if w > val:
                    val = w
                if val <= target:
                    nxt = i
                    break

            mask = prev
            cur = nxt

else:
    _btsp_held_karp_minmax_inplace = None  # type: ignore[assignment]

# Small LRU cache for workspaces (module-level so it persists across Solver objects).
_WS: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
_WS_ORDER: List[int] = []
_WS_MAX = 3

def _get_workspace(n: int) -> Tuple[np.ndarray, np.ndarray]:
    m = n - 1
    full_mask = (1 << m) - 1
    key = n
    ws = _WS.get(key)
    if ws is not None:
        try:
            _WS_ORDER.remove(key)
        except ValueError:
            pass
        _WS_ORDER.append(key)
        return ws

    dp = np.empty((full_mask + 1, m), dtype=np.float64)
    tour = np.empty(n + 1, dtype=np.int64)
    ws = (dp, tour)
    _WS[key] = ws
    _WS_ORDER.append(key)
    if len(_WS_ORDER) > _WS_MAX:
        old = _WS_ORDER.pop(0)
        _WS.pop(old, None)
    return ws

class Solver:
    _compiled: bool = False

    def __init__(self) -> None:
        # Trigger compilation outside solve() (init time not counted).
        if nb is not None and not Solver._compiled:
            dummy = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
            dp, tour = _get_workspace(2)
            _btsp_held_karp_minmax_inplace(dummy, dp, tour)  # type: ignore[misc]
            Solver._compiled = True

    def solve(self, problem: List[List[float]], **kwargs: Any) -> Any:
        n = len(problem)
        if n <= 1:
            return [0, 0]

        dist = np.asarray(problem, dtype=np.float64)

        if nb is None:
            return _btsp_held_karp_minmax_py(dist).tolist()

        dp, tour = _get_workspace(n)
        _btsp_held_karp_minmax_inplace(dist, dp, tour)  # type: ignore[misc]
        return tour.tolist()