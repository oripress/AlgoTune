import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def _gs(prop, recv):
    # Gale-Shapley stable matching with inlined ranking build
    n = prop.shape[0]
    # build receiver ranking table: recv_rank[r, p] = rank of proposer p for receiver r
    recv_rank = np.empty((n, n), np.int32)
    for r in range(n):
        for j in range(n):
            recv_rank[r, recv[r, j]] = j

    # initialize arrays
    next_prop = np.zeros(n, np.int32)
    recv_match = np.full(n, -1, np.int32)
    prop_match = np.full(n, -1, np.int32)
    free_stack = np.empty(n, np.int32)
    for i in range(n):
        free_stack[i] = i
    free_count = n

    # main loop
    while free_count > 0:
        free_count -= 1
        p = free_stack[free_count]
        r = prop[p, next_prop[p]]
        next_prop[p] += 1
        cur = recv_match[r]
        if cur == -1:
            recv_match[r] = p
            prop_match[p] = r
        else:
            # receiver chooses better proposer
            if recv_rank[r, p] < recv_rank[r, cur]:
                recv_match[r] = p
                prop_match[p] = r
                prop_match[cur] = -1
                free_stack[free_count] = cur
                free_count += 1
            else:
                free_stack[free_count] = p
                free_count += 1

    return prop_match

# Pre-compile for caching
_dummy = np.zeros((1, 1), np.int32)
_gs(_dummy, _dummy)

class Solver:
    def solve(self, problem, **kwargs):
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]

        # normalize proposer preferences
        if isinstance(prop_raw, dict):
            n = len(prop_raw)
            prop_prefs = [prop_raw[i] for i in range(n)]
        else:
            prop_prefs = prop_raw
            n = len(prop_prefs)

        # normalize receiver preferences
        if isinstance(recv_raw, dict):
            recv_prefs = [recv_raw[i] for i in range(n)]
        else:
            recv_prefs = recv_raw

        # convert to contiguous int32 arrays
        prop_arr = np.array(prop_prefs, dtype=np.int32)
        recv_arr = np.array(recv_prefs, dtype=np.int32)

        # run JIT-compiled Gale-Shapley
        match = _gs(prop_arr, recv_arr)
        return {"matching": match.tolist()}