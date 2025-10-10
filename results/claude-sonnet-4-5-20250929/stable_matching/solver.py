import numpy as np
from typing import Any
from numba import njit

@njit
def gale_shapley_numba(proposer_prefs, receiver_prefs, n):
    # Build receiver ranking tables
    recv_rank = np.zeros((n, n), dtype=np.int32)
    for r in range(n):
        for rank in range(n):
            recv_rank[r, receiver_prefs[r, rank]] = rank
    
    # Track next proposal index for each proposer
    next_prop = np.zeros(n, dtype=np.int32)
    # Track current match for each receiver (-1 if unmatched)
    recv_match = np.full(n, -1, dtype=np.int32)
    # Track free proposers
    free = np.ones(n, dtype=np.bool_)
    free_count = n
    
    while free_count > 0:
        # Find a free proposer
        p = -1
        for i in range(n):
            if free[i]:
                p = i
                break
        
        free[p] = False
        free_count -= 1
        
        r = proposer_prefs[p, next_prop[p]]
        next_prop[p] += 1
        
        cur = recv_match[r]
        if cur == -1:
            recv_match[r] = p
        else:
            if recv_rank[r, p] < recv_rank[r, cur]:
                recv_match[r] = p
                free[cur] = True
                free_count += 1
            else:
                free[p] = True
                free_count += 1
    
    # Build final matching array
    matching = np.zeros(n, dtype=np.int32)
    for r in range(n):
        matching[recv_match[r]] = r
    
    return matching

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]
        
        # Normalize to numpy arrays
        if isinstance(prop_raw, dict):
            n = len(prop_raw)
            proposer_prefs = np.array([prop_raw[i] for i in range(n)], dtype=np.int32)
        else:
            proposer_prefs = np.array(prop_raw, dtype=np.int32)
            n = len(proposer_prefs)
        
        if isinstance(recv_raw, dict):
            receiver_prefs = np.array([recv_raw[i] for i in range(n)], dtype=np.int32)
        else:
            receiver_prefs = np.array(recv_raw, dtype=np.int32)
        
        matching = gale_shapley_numba(proposer_prefs, receiver_prefs, n)
        
        return {"matching": matching.tolist()}