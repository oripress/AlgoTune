import numpy as np
from collections import deque
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]
        
        # Convert to numpy arrays for faster access
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
        
        # Build receiver ranking tables - rank of each proposer for each receiver
        recv_rank = np.zeros((n, n), dtype=np.int32)
        for r in range(n):
            recv_rank[r, receiver_prefs[r]] = np.arange(n, dtype=np.int32)
        
        # Initialize data structures
        next_prop = np.zeros(n, dtype=np.int32)  # Next preference index for each proposer
        recv_match = np.full(n, -1, dtype=np.int32)  # Current match for each receiver (-1 = unmatched)
        free = deque(range(n))  # Free proposers (using deque for O(1) pop/append)
        
        # Main Gale-Shapley algorithm
        while free:
            p = free.popleft()
            r = proposer_prefs[p, next_prop[p]]
            next_prop[p] += 1
            
            cur = recv_match[r]
            if cur == -1:
                recv_match[r] = p
            else:
                if recv_rank[r, p] < recv_rank[r, cur]:
                    recv_match[r] = p
                    free.append(cur)
                else:
                    free.append(p)
        
        # Build final matching (proposer -> receiver)
        matching = np.zeros(n, dtype=np.int32)
        for r in range(n):
            matching[recv_match[r]] = r
        
        return {"matching": matching.tolist()}