import numpy as np
from typing import Any, List, Optional
import numba

# Numba-jitted helper functions are defined at the top level for optimal performance.

@numba.jit(nopython=True)
def _find_hamiltonian_cycle_jit(n: int, adj_matrix: np.ndarray, path: np.ndarray, pos: int, visited: np.ndarray) -> bool:
    """
    A recursive utility to find a Hamiltonian cycle using a simple backtracking search.
    This version is optimized with Numba but does not use complex heuristics,
    prioritizing correctness and simplicity.
    """
    if pos == n:
        # Check if the last vertex is connected to the starting vertex (0)
        return adj_matrix[path[n - 1], path[0]]

    last_vertex = path[pos - 1]
    
    # Iterate through all vertices as potential next candidates
    for v in range(n):
        # Check if vertex v is adjacent to the last vertex and not yet visited
        if adj_matrix[last_vertex, v] and not visited[v]:
            path[pos] = v
            visited[v] = True

            if _find_hamiltonian_cycle_jit(n, adj_matrix, path, pos + 1, visited):
                return True

            # Backtrack
            visited[v] = False
    
    return False

@numba.jit(nopython=True)
def _is_connected(n: int, adj_matrix: np.ndarray) -> bool:
    """
    Checks if the graph is connected using BFS on a dense adjacency matrix.
    """
    if n <= 1:
        return True
    
    q = np.zeros(n, dtype=np.int32)
    q[0] = 0
    head, tail = 0, 1
    
    visited = np.zeros(n, dtype=np.bool_)
    visited[0] = True
    count = 1
    
    while head < tail:
        u = q[head]
        head += 1
        
        for v in range(n):
            if adj_matrix[u, v] and not visited[v]:
                visited[v] = True
                q[tail] = v
                tail += 1
                count += 1
    
    return count == n

class Solver:
    def solve(self, problem: List[List[float]], **kwargs: Any) -> List[int]:
        """
        Solves the Bottleneck Traveling Salesman Problem using a combination of
        binary search on edge weights and a Numba-accelerated backtracking search.
        """
        n = len(problem)
        
        if n == 0:
            return []
        if n <= 2:
            return list(range(n)) + [0]

        problem_np = np.array(problem, dtype=np.float64)

        iu = np.triu_indices(n, k=1)
        unique_weights = np.unique(problem_np[iu])

        low, high = 0, len(unique_weights) - 1
        optimal_tour = []
        
        while low <= high:
            mid_idx = (low + high) // 2
            w_max = unique_weights[mid_idx]
            
            path = self._check_for_hamiltonian_cycle(n, problem_np, w_max)
            
            if path:
                optimal_tour = path
                high = mid_idx - 1
            else:
                low = mid_idx + 1
        
        return optimal_tour + [0] if optimal_tour else []

    def _check_for_hamiltonian_cycle(self, n: int, problem_np: np.ndarray, w_max: float) -> Optional[List[int]]:
        """
        Checks if a Hamiltonian cycle exists for a given max weight w_max.
        Returns the path if found, otherwise None.
        """
        adj_matrix = problem_np <= w_max
        
        # Pruning 1: Degree check. Every vertex must have at least 2 edges.
        if np.any(np.sum(adj_matrix, axis=1) < 2):
            return None
            
        # Pruning 2: Connectivity check.
        if not _is_connected(n, adj_matrix):
            return None

        # If pruning passes, attempt the full backtracking search.
        path = np.full(n, -1, dtype=np.int32)
        visited = np.zeros(n, dtype=np.bool_)
        path[0] = 0
        visited[0] = True
        
        if _find_hamiltonian_cycle_jit(n, adj_matrix, path, 1, visited):
            return path.tolist()
        
        return None
        neighbors_arr = np.empty(n, dtype=np.int32)
        neighbor_degrees_arr = np.empty(n, dtype=np.int32)
        
        if _find_hamiltonian_cycle_jit_heuristic(n, adj_matrix, degrees.astype(np.int32), path, 1, visited, neighbors_arr, neighbor_degrees_arr):
            return path.tolist()
        
        return None