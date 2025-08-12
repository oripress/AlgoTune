import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from numba import njit

@njit
def calculate_efficiency_from_distances(dist_matrix, n):
    """Calculate global efficiency from distance matrix using only upper triangle."""
    total_inverse_distance = 0.0
    
    # Only iterate over upper triangle (j > i) to avoid duplicates
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_matrix[i, j]
            if d > 0 and d < np.inf:  # Connected nodes
                total_inverse_distance += 1.0 / d
    
    # We have n*(n-1)/2 pairs, but we only calculated for j > i
    # So we need to multiply by 2 to get all pairs
    return (2.0 * total_inverse_distance) / (n * (n - 1))

class Solver:
    def solve(self, problem: dict[str, list[list[int]]], **kwargs) -> dict[str, float]:
        """
        Calculates the global efficiency of the graph using scipy + numba optimization.
        
        Args:
            problem: A dictionary containing the adjacency list of the graph.
                     {"adjacency_list": adj_list}
        
        Returns:
            A dictionary containing the global efficiency.
            {"global_efficiency": efficiency_value}
        """
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        # Handle edge cases: efficiency is 0 for graphs with 0 or 1 node.
        if n <= 1:
            return {"global_efficiency": 0.0}
        
        # Build sparse matrix more efficiently using pre-allocated arrays
        n_edges = sum(len(neighbors) for neighbors in adj_list)
        
        # Pre-allocate arrays
        row_indices = np.empty(n_edges, dtype=np.int32)
        col_indices = np.empty(n_edges, dtype=np.int32)
        
        # Fill arrays directly
        idx = 0
        for u in range(n):
            for v in adj_list[u]:
                row_indices[idx] = u
                col_indices[idx] = v
                idx += 1
        
        data = np.ones(n_edges, dtype=np.int8)
        adj_sparse = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        
        # Compute shortest paths using scipy's optimized algorithm
        dist_matrix = shortest_path(adj_sparse, directed=False, unweighted=True)
        
        # Calculate global efficiency using numba-optimized function
        efficiency = calculate_efficiency_from_distances(dist_matrix, n)
        
        return {"global_efficiency": float(efficiency)}