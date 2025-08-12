import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve MST using Kruskal's algorithm with Union-Find.
        
        :param problem: dict with 'num_nodes', 'edges'
        :return: dict with 'mst_edges'
        """
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]
        
        # Sort edges by weight
        sorted_edges = sorted(edges, key=lambda x: x[2])
        
        # Union-Find data structure
        parent = list(range(num_nodes))
        rank = [0] * num_nodes
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            # Union by rank
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1
            return True
        
        # Kruskal's algorithm
        mst_edges = []
        for u, v, w in sorted_edges:
            if union(u, v):
                # Ensure u < v for consistent ordering
                if u > v:
                    u, v = v, u
                mst_edges.append([u, v, w])
                if len(mst_edges) == num_nodes - 1:
                    break
        
        # Sort by (u, v) as required
        mst_edges.sort(key=lambda x: (x[0], x[1]))
        return {"mst_edges": mst_edges}