from typing import Any, Dict, List
import numpy as np

# Try to import numba for JIT acceleration
try:
    from numba import njit

    @njit(cache=True, fastmath=False)
    def _gs_numba(proposer_prefs: np.ndarray, receiver_prefs: np.ndarray) -> np.ndarray:
        n = proposer_prefs.shape[0]

        # Build receiver ranking table
        recv_rank = np.empty((n, n), dtype=np.int32)
        for r in range(n):
            for rank in range(n):
                p = receiver_prefs[r, rank]
                recv_rank[r, p] = rank

        next_prop = np.zeros(n, dtype=np.int32)
        recv_match = np.full(n, -1, dtype=np.int32)

        # Use explicit stack for free proposers
        free_stack = np.empty(n, dtype=np.int32)
        top = 0
        for i in range(n):
            free_stack[i] = i
        top = n

        while top > 0:
            top -= 1
            p = free_stack[top]
            r_idx = next_prop[p]
            r = proposer_prefs[p, r_idx]
            next_prop[p] = r_idx + 1

            cur = recv_match[r]
            if cur == -1:
                recv_match[r] = p
            else:
                if recv_rank[r, p] < recv_rank[r, cur]:
                    recv_match[r] = p
                    free_stack[top] = cur
                    top += 1
                else:
                    free_stack[top] = p
                    top += 1

        # Build proposer -> receiver matching
        matching = np.empty(n, dtype=np.int32)
        for r in range(n):
            matching[recv_match[r]] = r
        return matching

    # Pre-compile once at import time to avoid first-call overhead during solve
    try:
        _ = _gs_numba(np.empty((0, 0), dtype=np.int32), np.empty((0, 0), dtype=np.int32))
    except Exception:
        pass

    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[int]]:
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]

        # Normalize to list-of-lists in the same way as the reference
        if isinstance(prop_raw, dict):
            n = len(prop_raw)
            proposer_prefs = [prop_raw[i] for i in range(n)]
        else:
            proposer_prefs = list(prop_raw)
            n = len(proposer_prefs)

        if isinstance(recv_raw, dict):
            receiver_prefs = [recv_raw[i] for i in range(n)]
        else:
            receiver_prefs = list(recv_raw)

        if n == 0:
            return {"matching": []}

        # Use Numba-accelerated path for moderate/large n
        if HAS_NUMBA and n >= 24:
            prop_arr = np.asarray(proposer_prefs, dtype=np.int32)
            recv_arr = np.asarray(receiver_prefs, dtype=np.int32)
            matching = _gs_numba(prop_arr, recv_arr).tolist()
            return {"matching": matching}

        # Optimized pure-Python fallback (uses flat receiver rank array and list as stack)
        # Build flattened receiver ranking tables for O(1) rank queries
        # recv_rank[r*n + p] = rank of proposer p in receiver r's preferences
        rn = n * n
        recv_rank = [0] * rn
        for r in range(n):
            base = r * n
            prefs_r = receiver_prefs[r]
            for rank, p in enumerate(prefs_r):
                recv_rank[base + p] = rank

        next_prop = [0] * n
        recv_match = [-1] * n
        free = list(range(n))  # use as stack (pop from end)

        proposer_prefs_local = proposer_prefs
        recv_rank_local = recv_rank
        recv_match_local = recv_match
        next_prop_local = next_prop
        free_pop = free.pop
        free_append = free.append
        n_local = n

        while free:
            p = free_pop()
            p_next = next_prop_local[p]
            r = proposer_prefs_local[p][p_next]
            next_prop_local[p] = p_next + 1

            cur = recv_match_local[r]
            if cur == -1:
                recv_match_local[r] = p
            else:
                base = r * n_local
                if recv_rank_local[base + p] < recv_rank_local[base + cur]:
                    recv_match_local[r] = p
                    free_append(cur)
                else:
                    free_append(p)

        matching = [0] * n
        for r, p in enumerate(recv_match_local):
            matching[p] = r

        return {"matching": matching}