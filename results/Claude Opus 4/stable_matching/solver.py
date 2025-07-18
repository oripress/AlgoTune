import numpy as np
from typing import Any
from numba import njit

@njit
def gale_shapley_core(proposer_prefs, receiver_prefs, n):
    # Create receiver ranking tables
    recv_rank = np.zeros((n, n), dtype=np.int32)
    for r in range(n):
        for rank in range(n):
            recv_rank[r, receiver_prefs[r, rank]] = rank
    
    # Initialize data structures
    next_prop = np.zeros(n, dtype=np.int32)
    recv_match = np.full(n, -1, dtype=np.int32)
    free = np.arange(n, dtype=np.int32)
    free_count = n
    
    # Gale-Shapley algorithm
    while free_count > 0:
        p = free[free_count - 1]
        free_count -= 1
        
        r = proposer_prefs[p, next_prop[p]]
        next_prop[p] += 1
        
        cur = recv_match[r]
        if cur == -1:
            recv_match[r] = p
        else:
            if recv_rank[r, p] < recv_rank[r, cur]:
                recv_match[r] = p
                free[free_count] = cur
                free_count += 1
            else:
                free[free_count] = p
                free_count += 1
    
    # Build final matching
    matching = np.empty(n, dtype=np.int32)
    for r in range(n):
        matching[recv_match[r]] = r
    
    return matching

class Solver:
    def __init__(self):
        # Warm up JIT compilation
        test_prefs = np.array([[0, 1], [1, 0]], dtype=np.int32)
        gale_shapley_core(test_prefs, test_prefs, 2)
    
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]
        
        # Convert to numpy arrays
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
        
        # Run the core algorithm
        matching = gale_shapley_core(proposer_prefs, receiver_prefs, n)
        
        return {"matching": matching.tolist()}