import numpy as np
from typing import Any, List, Set, Tuple
import itertools
import time

class Solver:
    def solve(self, problem: List[List[float]], **kwargs) -> Any:
        """
        Solve the BTSP problem using binary search on edge weights.
        Returns a tour as a list of city indices starting and ending at city 0.
        """
        n = len(problem)
        if n <= 1:
            return [0, 0]
        
        if n == 2:
            return [0, 1, 0]
        
        # Convert to numpy for faster access
        dist = np.array(problem)
        
        # For small instances, use brute force
        if n <= 10:
            return self._brute_force_numpy(dist)
        
        # For larger instances, use binary search approach with timeout
        start_time = time.time()
        result = self._binary_search_bottleneck_numpy(dist, start_time)
        
        # If timeout or no solution, use fast heuristic
        if result is None:
            result = self._fast_bottleneck_heuristic(dist)
        
        return result
    
    def _brute_force_numpy(self, dist: np.ndarray) -> List[int]:
        """Brute force solution for small instances using numpy."""
        n = len(dist)
        cities = list(range(1, n))
        
        best_tour = None
        best_bottleneck = float('inf')
        
        # Try all permutations starting from city 0
        for perm in itertools.permutations(cities):
            tour = [0] + list(perm) + [0]
            
            # Calculate bottleneck efficiently
            bottleneck = 0
            for i in range(n):
                edge_weight = dist[tour[i], tour[i+1]]
                if edge_weight > bottleneck:
                    bottleneck = edge_weight
                    if bottleneck >= best_bottleneck:
                        break  # Early termination
            
            if bottleneck < best_bottleneck:
                best_bottleneck = bottleneck
                best_tour = tour
        
        return best_tour
    
    def _binary_search_bottleneck_numpy(self, dist: np.ndarray, start_time: float) -> List[int]:
        """Use binary search on edge weights to find optimal bottleneck."""
        n = len(dist)
        timeout = 0.5  # 500ms timeout for binary search
        
        # Collect unique edge weights efficiently
        edge_weights = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_weights.append(dist[i, j])
        
        edge_weights = sorted(set(edge_weights))
        
        # Binary search on edge weights
        left, right = 0, len(edge_weights) - 1
        best_tour = None
        
        while left <= right:
            if time.time() - start_time > timeout:
                return best_tour  # Return best found so far
            
            mid = (left + right) // 2
            threshold = edge_weights[mid]
            
            # Check if there's a Hamiltonian cycle with edges <= threshold
            tour = self._find_hamiltonian_cycle_fast(dist, threshold, start_time, timeout)
            
            if tour:
                best_tour = tour
                right = mid - 1
    def _find_hamiltonian_cycle_fast(self, dist: np.ndarray, threshold: float, 
                                    start_time: float, timeout: float) -> List[int]:
        """Find a Hamiltonian cycle using only edges with weight <= threshold."""
        n = len(dist)
        
        # Build adjacency matrix for edges <= threshold
        adj_matrix = (dist <= threshold) & (dist > 0)
        
        # Check degree constraints first
        degrees = np.sum(adj_matrix, axis=1)
        if np.any(degrees < 2):
            return None
        
        # Use DFS with optimizations
        visited = np.zeros(n, dtype=bool)
        path = [0]
        visited[0] = True
        
        if self._dfs_optimized(adj_matrix, visited, path, 0, n, start_time, timeout):
            path.append(0)
            return path
        
        return None
        
        if self._dfs_optimized(adj_matrix, visited, path, 0, n, start_time, timeout):
            path.append(0)
            return path
        
        return None
    
    def _dfs_optimized(self, adj_matrix: np.ndarray, visited: np.ndarray, 
                      path: List[int], curr: int, n: int, start_time: float, timeout: float) -> bool:
        """Optimized DFS to find Hamiltonian path."""
        if time.time() - start_time > timeout:
            return False
            
        if len(path) == n:
            # Check if we can return to start
            return adj_matrix[path[-1], 0]
        
        # Get unvisited neighbors
        neighbors = np.where(adj_matrix[curr] & ~visited)[0]
        
        # Sort by degree (prefer vertices with fewer unvisited neighbors)
        if len(neighbors) > 1:
            unvisited_degrees = []
            for v in neighbors:
                degree = np.sum(adj_matrix[v] & ~visited)
                unvisited_degrees.append((degree, v))
            neighbors = [v for _, v in sorted(unvisited_degrees)]
        
        for next_city in neighbors:
            visited[next_city] = True
            path.append(next_city)
            
            if self._dfs_optimized(adj_matrix, visited, path, next_city, n, start_time, timeout):
                return True
            
            # Backtrack
            path.pop()
            visited[next_city] = False
        
        return False
    
    def _fast_bottleneck_heuristic(self, dist: np.ndarray) -> List[int]:
        """Fast heuristic for BTSP using Christofides-like approach."""
        n = len(dist)
        
        # Start with MST-based approach
        visited = np.zeros(n, dtype=bool)
        tour = [0]
        visited[0] = True
        
        # Build tour greedily, but considering bottleneck
        for _ in range(n - 1):
            curr = tour[-1]
            best_next = -1
            best_score = float('inf')
            
            for next_city in range(n):
                if not visited[next_city]:
                    # Score based on current edge weight and potential future bottleneck
                    edge_weight = dist[curr, next_city]
                    
                    # Look ahead: what's the minimum edge from next_city to unvisited nodes
                    future_min = float('inf')
                    for other in range(n):
                        if not visited[other] and other != next_city:
                            future_min = min(future_min, dist[next_city, other])
                    
                    # Also consider return to start
                    if np.sum(visited) == n - 2:  # This would be the last city
                        future_min = min(future_min, dist[next_city, 0])
                    
                    score = max(edge_weight, future_min)
                    
                    if score < best_score:
                        best_score = score
                        best_next = next_city
            
            tour.append(best_next)
            visited[best_next] = True
        
        tour.append(0)
        return tour