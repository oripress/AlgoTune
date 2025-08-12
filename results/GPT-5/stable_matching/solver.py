from typing import Any, Dict, List

import numpy as np

# Optional accelerators
CYTHON_AVAILABLE = False
try:
    from gs import gs_solve as _gs_cy  # compiled from gs.pyx if available
    CYTHON_AVAILABLE = True
except Exception:  # pragma: no cover
    _gs_cy = None
    CYTHON_AVAILABLE = False

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[int]]:
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]

        # Normalize to list-of-lists without unnecessary copying
        if isinstance(prop_raw, dict):
            n = len(prop_raw)
            proposer_prefs = [prop_raw[i] for i in range(n)]
        else:
            proposer_prefs = prop_raw
            n = len(proposer_prefs)

        if isinstance(recv_raw, dict):
            receiver_prefs = [recv_raw[i] for i in range(n)]
        else:
            receiver_prefs = recv_raw

        if n == 0:
            return {"matching": []}

        # For small instances, the Python implementation is faster due to lower overhead.
        SMALL_N_CUTOFF = 32
        use_cython = CYTHON_AVAILABLE and n > SMALL_N_CUTOFF

        if use_cython:
            prop_np = np.asarray(proposer_prefs, dtype=np.int32, order="C")
            recv_np = np.asarray(receiver_prefs, dtype=np.int32, order="C")
            matching = _gs_cy(prop_np, recv_np)
            if isinstance(matching, np.ndarray):
                matching = matching.tolist()
            else:
                matching = list(matching)
            return {"matching": matching}

        # Fallback: optimized pure-Python implementation with chained proposals

        # Precompute receiver ranking tables: rank[r][p] = rank of proposer p for receiver r
        recv_rank = [[0] * n for _ in range(n)]
        for r in range(n):
            rank_r = recv_rank[r]
            prefs_r = receiver_prefs[r]
            for rk, p in enumerate(prefs_r):
                rank_r[p] = rk

        next_prop = [0] * n
        recv_match = [-1] * n  # -1 indicates unmatched

        # Local bindings
        proposer_prefs_local = proposer_prefs
        recv_rank_local = recv_rank
        recv_match_local = recv_match
        next_prop_local = next_prop

        # Gale-Shapley algorithm (proposer-optimal) with chained proposals
        for p0 in range(n):
            p = p0
            while True:
                p_next = next_prop_local[p]
                row = proposer_prefs_local[p]
                r = row[p_next]
                next_prop_local[p] = p_next + 1

                cur = recv_match_local[r]
                if cur == -1:
                    recv_match_local[r] = p
                    break
                else:
                    rank_r = recv_rank_local[r]
                    if rank_r[p] < rank_r[cur]:
                        recv_match_local[r] = p
                        p = cur  # displaced proposer continues proposing
                    else:
                        # rejected; keep proposing further without stack ops
                        continue

        # Build proposer->receiver mapping
        matching = [0] * n
        for r in range(n):
            p = recv_match_local[r]
            matching[p] = r

        return {"matching": matching}