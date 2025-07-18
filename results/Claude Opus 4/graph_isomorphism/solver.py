from typing import Any
import numpy as np
from collections import defaultdict

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        """
        Find graph isomorphism mapping using degree-based pruning and backtracking.
        """
        n = problem["num_nodes"]
        edges_g1 = problem["edges_g1"]
        edges_g2 = problem["edges_g2"]
        
        # Build adjacency lists
        adj1 = [set() for _ in range(n)]
        adj2 = [set() for _ in range(n)]
        
        for u, v in edges_g1:
            adj1[u].add(v)
            adj1[v].add(u)
            
        for u, v in edges_g2:
            adj2[u].add(v)
            adj2[v].add(u)
        
        # Calculate degrees
        deg1 = [len(adj1[i]) for i in range(n)]
        deg2 = [len(adj2[i]) for i in range(n)]
        
        # Group nodes by degree
        deg_to_nodes1 = defaultdict(list)
        deg_to_nodes2 = defaultdict(list)
        
        for i in range(n):
            deg_to_nodes1[deg1[i]].append(i)
            deg_to_nodes2[deg2[i]].append(i)
        
        # Check if degree sequences match
        if sorted(deg1) != sorted(deg2):
            return {"mapping": [-1] * n}
        
        # For each node in G1, find possible matches in G2 based on degree
        possible_matches = []
        for i in range(n):
            possible_matches.append(deg_to_nodes2[deg1[i]][:])
        
        # Backtracking search
        mapping = [-1] * n
        used = [False] * n
        
        def backtrack(node):
            if node == n:
                return True
            
            for candidate in possible_matches[node]:
                if used[candidate]:
                    continue
                
                # Check if this mapping preserves edges
                valid = True
                for neighbor in adj1[node]:
                    if neighbor < node:  # Already mapped
                        if mapping[neighbor] not in adj2[candidate]:
                            valid = False
                            break
                
                if valid:
                    mapping[node] = candidate
                    used[candidate] = True
                    
                    if backtrack(node + 1):
                        return True
                    
                    used[candidate] = False
                    mapping[node] = -1
            
            return False
        
        if backtrack(0):
            return {"mapping": mapping}
        else:
            return {"mapping": [-1] * n}