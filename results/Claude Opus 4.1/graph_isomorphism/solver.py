import numpy as np
from typing import Any
from collections import defaultdict

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        """
        Fast graph isomorphism solver using degree sequences and adjacency patterns.
        """
        n = problem["num_nodes"]
        edges_g1 = problem["edges_g1"]
        edges_g2 = problem["edges_g2"]
        
        # Build adjacency lists
        adj1 = defaultdict(set)
        adj2 = defaultdict(set)
        
        for u, v in edges_g1:
            adj1[u].add(v)
            adj1[v].add(u)
        
        for u, v in edges_g2:
            adj2[u].add(v)
            adj2[v].add(u)
        
        # Compute degree sequences
        deg1 = [len(adj1[i]) for i in range(n)]
        deg2 = [len(adj2[i]) for i in range(n)]
        
        # Group nodes by degree for faster matching
        nodes_by_deg1 = defaultdict(list)
        nodes_by_deg2 = defaultdict(list)
        
        for i in range(n):
            nodes_by_deg1[deg1[i]].append(i)
            nodes_by_deg2[deg2[i]].append(i)
        
        # Quick check for special case: cycle graph
        if all(d == 2 for d in deg1) and len(edges_g1) == n:
            # It's a cycle - use fast cycle isomorphism
            return self._solve_cycle(n, adj1, adj2)
        
        # Use degree-based refinement with backtracking
        mapping = [-1] * n
        used = [False] * n
        
        # Sort nodes by degree (descending) for better pruning
        nodes_sorted = sorted(range(n), key=lambda x: -deg1[x])
        
        if self._backtrack(0, nodes_sorted, mapping, used, adj1, adj2, deg1, deg2, nodes_by_deg2):
            return {"mapping": mapping}
        
        # Fallback: shouldn't happen for guaranteed isomorphic graphs
        return {"mapping": list(range(n))}
    
    def _solve_cycle(self, n, adj1, adj2):
        """Fast solver for cycle graphs."""
        # Start from node 0 in G1, try mapping to each node in G2
        for start2 in range(n):
            mapping = [-1] * n
            mapping[0] = start2
            
            # Traverse the cycle in G1 and G2 simultaneously
            curr1 = 0
            curr2 = start2
            used = [False] * n
            used[start2] = True
            
            valid = True
            for _ in range(n - 1):
                # Find next node in G1
                next1 = -1
                for neighbor in adj1[curr1]:
                    if mapping[neighbor] == -1:
                        next1 = neighbor
                        break
                
                if next1 == -1:
                    valid = False
                    break
                
                # Find corresponding next node in G2
                next2 = -1
                for neighbor in adj2[curr2]:
                    if not used[neighbor]:
                        next2 = neighbor
                        break
                
                if next2 == -1:
                    valid = False
                    break
                
                mapping[next1] = next2
                used[next2] = True
                curr1 = next1
                curr2 = next2
            
            if valid:
                return {"mapping": mapping}
        
        return {"mapping": list(range(n))}
    
    def _backtrack(self, idx, nodes_sorted, mapping, used, adj1, adj2, deg1, deg2, nodes_by_deg2):
        """Backtracking with degree-based pruning."""
        if idx == len(nodes_sorted):
            return True
        
        node1 = nodes_sorted[idx]
        degree = deg1[node1]
        
        # Only try nodes in G2 with the same degree
        for node2 in nodes_by_deg2[degree]:
            if used[node2]:
                continue
            
            # Check if this mapping is consistent with already mapped neighbors
            valid = True
            for neighbor1 in adj1[node1]:
                if mapping[neighbor1] != -1:
                    if mapping[neighbor1] not in adj2[node2]:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Try this mapping
            mapping[node1] = node2
            used[node2] = True
            
            if self._backtrack(idx + 1, nodes_sorted, mapping, used, adj1, adj2, deg1, deg2, nodes_by_deg2):
                return True
            
            # Backtrack
            mapping[node1] = -1
            used[node2] = False
        
        return False