from typing import Any
import numpy as np
from numba import njit

@njit(cache=True)
def solve_numba(proposer_prefs, receiver_prefs):
    n = proposer_prefs.shape[0]
    
    # Build recv_rank
    recv_rank = np.empty((n, n), dtype=np.int32)
    for r in range(n):
        for rank in range(n):
            p = receiver_prefs[r, rank]
            recv_rank[r, p] = rank
    
    # Main algorithm
    next_prop = np.zeros(n, dtype=np.int32)
    recv_match = np.full(n, -1, dtype=np.int32)
    
    stack = np.empty(n, dtype=np.int32)
    for i in range(n):
        stack[i] = i
    stack_size = n
    
    while stack_size > 0:
        stack_size -= 1
        p = stack[stack_size]
        
        r = proposer_prefs[p, next_prop[p]]
        next_prop[p] += 1
        
        cur = recv_match[r]
        if cur == -1:
            recv_match[r] = p
        else:
            if recv_rank[r, p] < recv_rank[r, cur]:
                recv_match[r] = p
                stack[stack_size] = cur
                stack_size += 1
            else:
                stack[stack_size] = p
                stack_size += 1
    
    matching = np.empty(n, dtype=np.int32)
    for r in range(n):
        matching[recv_match[r]] = r
    
    return matching

def solve_python(proposer_prefs, receiver_prefs, n):
    """Optimized pure Python with stack (O(1) pop) instead of queue (O(n) pop)"""
    recv_rank = [[0] * n for _ in range(n)]
    for r in range(n):
        prefs = receiver_prefs[r]
        rr = recv_rank[r]
        for rank in range(n):
            rr[prefs[rank]] = rank
    
    next_prop = [0] * n
    recv_match = [-1] * n
    free = list(range(n))
    
    while free:
        p = free.pop()  # O(1) vs O(n) in reference
        r = proposer_prefs[p][next_prop[p]]
        next_prop[p] += 1
        
        cur = recv_match[r]
        if cur == -1:
            recv_match[r] = p
        else:
            rr = recv_rank[r]
            if rr[p] < rr[cur]:
                recv_match[r] = p
                free.append(cur)
            else:
                free.append(p)
    
    matching = [0] * n
    for r in range(n):
        matching[recv_match[r]] = r
    
    return matching

class Solver:
    def __init__(self):
        dummy_prefs = np.array([[0]], dtype=np.int32)
        solve_numba(dummy_prefs, dummy_prefs)
    
    def solve(self, problem, **kwargs) -> Any:
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]

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

        # Use Python for small inputs (avoid numpy conversion overhead)
        if n <= 75:
            matching = solve_python(proposer_prefs, receiver_prefs, n)
            return {"matching": matching}
        
        proposer_prefs_arr = np.array(proposer_prefs, dtype=np.int32)
        receiver_prefs_arr = np.array(receiver_prefs, dtype=np.int32)
        
        matching = solve_numba(proposer_prefs_arr, receiver_prefs_arr)
        
        return {"matching": matching.tolist()}