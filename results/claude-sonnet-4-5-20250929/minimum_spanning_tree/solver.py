from numba import njit
import numpy as np

@njit
def find_root(parent, x):
    """Find with path compression"""
    root = x
    while parent[root] != root:
        root = parent[root]
    # path compression
    while parent[x] != root:
        next_x = parent[x]
        parent[x] = root
        x = next_x
    return root

@njit
def union_sets(parent, rank, x, y):
    """Union by rank"""
    px = find_root(parent, x)
    py = find_root(parent, y)
    if px == py:
        return False
    if rank[px] < rank[py]:
        px, py = py, px
    parent[py] = px
    if rank[px] == rank[py]:
        rank[px] += 1
    return True

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute MST using Kruskal's algorithm with Union-Find.
        Uses Numba JIT compilation for Union-Find operations.
        
        :param problem: dict with 'num_nodes', 'edges'
        :return: dict with 'mst_edges'
        """
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]
        
        # Union-Find data structure with numpy arrays
        parent = np.arange(num_nodes, dtype=np.int32)
        rank = np.zeros(num_nodes, dtype=np.int32)
        
        # Sort edges by weight
        edges.sort(key=lambda e: e[2])
        
        mst_edges = []
        for u, v, w in edges:
            if union_sets(parent, rank, u, v):
                # Ensure u < v
                if u > v:
                    u, v = v, u
                mst_edges.append([u, v, w])
                if len(mst_edges) == num_nodes - 1:
                    break
        
        # Sort by (u, v)
        mst_edges.sort()
        return {"mst_edges": mst_edges}