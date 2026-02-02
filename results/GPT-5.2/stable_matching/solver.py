from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

# -------- Numba-accelerated core --------
if njit is not None:

    @njit(cache=True)
    def _gs_numba(prop_prefs: np.ndarray, recv_rank: np.ndarray) -> np.ndarray:
        """
        Proposer-optimal stable matching (Gale–Shapley), nopython.

        prop_prefs: (n, n) int32, receivers in preference order for each proposer
        recv_rank:  (n, n) int32, recv_rank[r, p] = rank of proposer p for receiver r
        Returns: (n,) int32 matching[p] = r
        """
        n = prop_prefs.shape[0]

        next_prop = np.zeros(n, dtype=np.int32)
        recv_match = -np.ones(n, dtype=np.int32)

        # Stack of free proposers (LIFO), avoid np.arange allocation.
        free = np.empty(n, dtype=np.int32)
        for i in range(n):
            free[i] = i
        top = n

        while top > 0:
            top -= 1
            p = free[top]

            i = next_prop[p]
            p_row = prop_prefs[p]
            r = p_row[i]
            next_prop[p] = i + 1

            cur = recv_match[r]
            if cur == -1:
                recv_match[r] = p
            else:
                rank_row = recv_rank[r]
                if rank_row[p] < rank_row[cur]:
                    recv_match[r] = p
                    free[top] = cur
                    top += 1
                else:
                    free[top] = p
                    top += 1

        matching = np.empty(n, dtype=np.int32)
        for r in range(n):
            matching[recv_match[r]] = r
        return matching

def _as_list_of_lists(raw: Any) -> tuple[list[list[int]], int]:
    """Normalize prefs possibly given as dict[int, list[int]]. Avoid copies when already list."""
    if isinstance(raw, dict):
        n = len(raw)
        return [raw[i] for i in range(n)], n
    if isinstance(raw, list):
        return raw, len(raw)
    prefs = list(raw)
    return prefs, len(prefs)

def _as_list_of_lists_fixed(raw: Any, n: int) -> list[list[int]]:
    if isinstance(raw, dict):
        return [raw[i] for i in range(n)]
    if isinstance(raw, list):
        return raw
    return list(raw)

class Solver:
    """
    Fast stable matching solver.

    - Small n: pure-Python stack-based Gale–Shapley (avoids numpy materialization).
    - Larger n: numpy builds receiver rank table; Numba runs core loop.
    """

    _PYTHON_CUTOFF = 48

    def __init__(self) -> None:
        # Force JIT compilation outside timed solve() if possible.
        if njit is not None:
            prop = np.array([[0]], dtype=np.int32)
            rank = np.array([[0]], dtype=np.int32)
            _gs_numba(prop, rank)

    @staticmethod
    def _solve_python(proposer_prefs: list[list[int]], receiver_prefs: list[list[int]]) -> list[int]:
        n = len(proposer_prefs)

        recv_rank = [[0] * n for _ in range(n)]
        for r, prefs in enumerate(receiver_prefs):
            for rank, p in enumerate(prefs):
                recv_rank[r][p] = rank

        next_prop = [0] * n
        recv_match = [-1] * n
        free = list(range(n))  # stack (LIFO)

        while free:
            p = free.pop()
            i = next_prop[p]
            r = proposer_prefs[p][i]
            next_prop[p] = i + 1

            cur = recv_match[r]
            if cur == -1:
                recv_match[r] = p
            else:
                if recv_rank[r][p] < recv_rank[r][cur]:
                    recv_match[r] = p
                    free.append(cur)
                else:
                    free.append(p)

        matching = [0] * n
        for r, p in enumerate(recv_match):
            matching[p] = r
        return matching

    @staticmethod
    def _build_recv_rank(recv_prefs: np.ndarray) -> np.ndarray:
        """Inverse permutation per receiver (row-wise assignment to reduce temporaries)."""
        n = int(recv_prefs.shape[0])
        recv_rank = np.empty((n, n), dtype=np.int32)
        base = np.arange(n, dtype=np.int32)
        for r in range(n):
            recv_rank[r, recv_prefs[r]] = base
        return recv_rank

    def solve(self, problem: Any, **kwargs: Any) -> Any:
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]

        # Numpy fast path.
        if isinstance(prop_raw, np.ndarray):
            prop = np.ascontiguousarray(prop_raw, dtype=np.int32)
            n = int(prop.shape[0])
            if isinstance(recv_raw, np.ndarray):
                recv = np.ascontiguousarray(recv_raw, dtype=np.int32)
            else:
                recv = np.ascontiguousarray(
                    np.asarray(_as_list_of_lists_fixed(recv_raw, n), dtype=np.int32)
                )

            if njit is None:
                return {"matching": self._solve_python(prop.tolist(), recv.tolist())}

            recv_rank = self._build_recv_rank(recv)
            return {"matching": _gs_numba(prop, recv_rank).tolist()}

        proposer_prefs, n = _as_list_of_lists(prop_raw)
        receiver_prefs = _as_list_of_lists_fixed(recv_raw, n)

        if njit is None or n <= self._PYTHON_CUTOFF:
            return {"matching": self._solve_python(proposer_prefs, receiver_prefs)}

        prop = np.ascontiguousarray(np.asarray(proposer_prefs, dtype=np.int32))
        recv = np.ascontiguousarray(np.asarray(receiver_prefs, dtype=np.int32))
        recv_rank = self._build_recv_rank(recv)
        matching = _gs_numba(prop, recv_rank)
        return {"matching": matching.tolist()}