from typing import Any, Dict, List
import numba
from numba import jit
import numpy as np

@jit(nopython=True)
def gale_shapley_numba(proposer_prefs, receiver_prefs, n):
    # Create receiver rankings using a more efficient approach
    # Instead of storing rankings, we'll store the preference positions directly
    recv_rank = np.empty((n, n), dtype=np.int16)
    for r in range(n):
        for rank in range(n):
            p = receiver_prefs[r, rank]
            recv_rank[r, p] = np.int16(rank)
    
    # Track next proposer to consider for each receiver
    next_prop = np.zeros(n, dtype=np.int16)
    # Track who each receiver is currently matched to (-1 if unmatched)
    recv_match = np.full(n, -1, dtype=np.int16)
    # Free proposers (initially all)
    free_proposers = np.arange(n, dtype=np.int16)
    free_count = np.int16(n)
    
    # Pre-allocate matching array
    matching = np.zeros(n, dtype=np.int16)
    
    # Gale-Shapley algorithm - optimized version
    while free_count > 0:
        # Get a free proposer
        free_count -= 1
        p = free_proposers[free_count]
        
        # Get the next receiver this proposer wants to propose to
        r = proposer_prefs[p, next_prop[p]]
        next_prop[p] += 1
        
        # Check if receiver is free
        current_match = recv_match[r]
        if current_match == -1:
            # Receiver is free, match them
            recv_match[r] = p
        else:
            # Receiver is matched, check if they prefer the new proposer
            # Using direct array access for speed
            new_proposer_rank = recv_rank[r, p]
            current_proposer_rank = recv_rank[r, current_match]
            
            if new_proposer_rank < current_proposer_rank:
                # Receiver prefers new proposer, switch matches
                recv_match[r] = p
                free_proposers[free_count] = current_match  # Old match becomes free
                free_count += 1
            else:
                # Receiver prefers current match, proposer stays free
                free_proposers[free_count] = p
                free_count += 1
    
    # Convert receiver->proposer mapping to proposer->receiver mapping directly
    for r in range(n):
        p = recv_match[r]
        if p != -1:
            matching[p] = r
        
    return matching

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[int]]:
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]
        
        # Normalize to list-of-lists
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
        
        # Convert to numpy arrays with smaller data types for efficiency
        proposer_prefs_np = np.array(proposer_prefs, dtype=np.int16)
        receiver_prefs_np = np.array(receiver_prefs, dtype=np.int16)
        
        # Run optimized Gale-Shapley algorithm
        matching = gale_shapley_numba(proposer_prefs_np, receiver_prefs_np, n)
        
        return {"matching": matching.tolist()}