import numpy as np
import numba

@numba.njit(fastmath=True)
def solve_numba(proposer_prefs, receiver_prefs):
    n = proposer_prefs.shape[0]
    
    # Optimized receiver ranking matrix computation
    recv_rank = np.zeros((n, n), dtype=np.int32)
    for r in range(n):
        recv_prefs = receiver_prefs[r]
        for rank in range(n):
            recv_rank[r, recv_prefs[rank]] = rank
    
    next_prop = np.zeros(n, dtype=np.int32)
    recv_match = np.full(n, -1, dtype=np.int32)
    free_stack = np.zeros(n, dtype=np.int32)
    free_ptr = n - 1
    
    # Initialize free stack
    for i in range(n):
        free_stack[i] = i
    
    # Main algorithm loop
    while free_ptr >= 0:
        p = free_stack[free_ptr]
        free_ptr -= 1
        
        # Skip exhausted proposers
        if next_prop[p] >= n:
            continue
            
        # Get next receiver preference
        prop_prefs = proposer_prefs[p]
        r = prop_prefs[next_prop[p]]
        next_prop[p] += 1
        
        cur = recv_match[r]
        if cur == -1:
            recv_match[r] = p
        else:
            # Direct comparison with precomputed ranks
            if recv_rank[r, p] < recv_rank[r, cur]:
                recv_match[r] = p
                free_ptr += 1
                free_stack[free_ptr] = cur
            else:
                free_ptr += 1
                free_stack[free_ptr] = p
    
    # Simplified matching array construction
    matching = np.zeros(n, dtype=np.int32)
    for r in range(n):
        matching[recv_match[r]] = r
    
    return matching

class Solver:
    def solve(self, problem, **kwargs):
        proposer_prefs_raw = problem["proposer_prefs"]
        receiver_prefs_raw = problem["receiver_prefs"]
        
        # Handle input format variations
        n = len(proposer_prefs_raw) if not isinstance(proposer_prefs_raw, dict) else len(proposer_prefs_raw)
        
        # Convert to efficient NumPy arrays
        proposer_prefs = np.array(proposer_prefs_raw if not isinstance(proposer_prefs_raw, dict) 
                                else [proposer_prefs_raw[i] for i in range(n)], dtype=np.int32)
        receiver_prefs = np.array(receiver_prefs_raw if not isinstance(receiver_prefs_raw, dict) 
                                 else [receiver_prefs_raw[i] for i in range(n)], dtype=np.int32)
        
        matching = solve_numba(proposer_prefs, receiver_prefs)
        return {"matching": matching.tolist()}