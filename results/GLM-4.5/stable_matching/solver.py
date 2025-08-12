from typing import Any
import array
from collections import deque

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]
        
        # normalise to list-of-lists
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
        
        # Build receiver ranking using array module for efficiency
        recv_rank = []
        for prefs in receiver_prefs:
            rank_array = array.array('i', [0]) * n
            for rank, p in enumerate(prefs):
                rank_array[p] = rank
            recv_rank.append(rank_array)
        
        # Initialize data structures
        next_prop = [0] * n
        recv_match = [-1] * n
        free = deque(range(n))
        
        # Optimized main loop
        while free:
            p = free.popleft()
            
            # Get next preference
            prefs_p = proposer_prefs[p]
            j = next_prop[p]
            r = prefs_p[j]
            next_prop[p] = j + 1
            
            # Check if receiver is free
            current_match = recv_match[r]
            if current_match == -1:
                recv_match[r] = p
            else:
                # Use array lookup for O(1) access
                rank_r = recv_rank[r]
                rank_p = rank_r[p]
                rank_current = rank_r[current_match]
                if rank_p < rank_current:
                    recv_match[r] = p
                    free.append(current_match)
                else:
                    free.append(p)
        
        # Build result
        matching = [0] * n
        for r, p in enumerate(recv_match):
            matching[p] = r
        
        return {"matching": matching}