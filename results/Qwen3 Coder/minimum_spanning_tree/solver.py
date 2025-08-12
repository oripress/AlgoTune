import heapq
from typing import Any, List, Tuple

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # Already connected
        
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, List[List[float]]]:
        """
        Find MST using Kruskal's algorithm with Union-Find for optimal performance.
        
        :param problem: dict with 'num_nodes', 'edges'
        :return: dict with 'mst_edges'
        """
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]
        
        # Sort edges by weight
        sorted_edges = sorted(edges, key=lambda x: x[2])
        
        # Initialize Union-Find structure
        uf = UnionFind(num_nodes)
        
        # Kruskal's algorithm
        mst_edges = []
        for u, v, weight in sorted_edges:
            if uf.union(u, v):  # If not forming a cycle
                # Ensure u < v for consistency
                if u > v:
                    u, v = v, u
                mst_edges.append([u, v, weight])
                
                # Early termination: MST complete when we have n-1 edges
                if len(mst_edges) == num_nodes - 1:
                    break
        
        # Sort by (u, v) for consistent output
        mst_edges.sort(key=lambda x: (x[0], x[1]))
        
        return {"mst_edges": mst_edges}