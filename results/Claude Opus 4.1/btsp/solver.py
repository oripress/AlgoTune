import itertools
import numpy as np
import networkx as nx

class Solver:
    def solve(self, problem: list[list[float]]) -> list[int]:
        """
        Solve the BTSP problem.
        Returns a tour as a list of city indices starting and ending at city 0.
        """
        n = len(problem)
        if n <= 1:
            return [0, 0]
        if n == 2:
            return [0, 1, 0]
        
        # Convert to numpy array for faster access
        dist = np.array(problem)
        
        # Use a binary search approach to find the minimum bottleneck
        # Get all unique edge weights
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                edges.append(dist[i][j])
        edges = sorted(set(edges))
        
        left, right = 0, len(edges) - 1
        best_tour = None
        
        while left <= right:
            mid = (left + right) // 2
            threshold = edges[mid]
            
            # Check if we can form a Hamiltonian cycle using only edges <= threshold
            tour = self._find_hamiltonian_cycle_with_threshold(dist, threshold)
            
            if tour is not None:
                best_tour = tour
                right = mid - 1
            else:
                left = mid + 1
        
        return best_tour if best_tour else []
    
    def _find_hamiltonian_cycle_with_threshold(self, dist, threshold):
        """Find a Hamiltonian cycle using only edges with weight <= threshold."""
        n = len(dist)
        
        # Create a graph with only edges <= threshold
        graph = nx.Graph()
        # Add all nodes first
        graph.add_nodes_from(range(n))
        
        # Add edges with weight <= threshold
        for i in range(n):
            for j in range(i+1, n):
                if dist[i][j] <= threshold:
                    graph.add_edge(i, j)
        
        # Check if graph is connected
        if not nx.is_connected(graph):
            return None
        
        # Check if all vertices have degree >= 2 (necessary for Hamiltonian cycle)
        for node in graph.nodes():
            if graph.degree(node) < 2:
                return None
        
        # Try to find a Hamiltonian cycle using DFS
        visited = [False] * n
        path = [0]
        visited[0] = True
        
        if self._hamiltonian_dfs(graph, 0, visited, path, n):
            path.append(0)
            return path
    def _hamiltonian_dfs(self, graph, curr, visited, path, n):
        """DFS to find Hamiltonian path."""
        if len(path) == n:
            # Check if we can return to start
            return 0 in graph.neighbors(curr)
        
        for next_node in graph.neighbors(curr):
            if not visited[next_node]:
                visited[next_node] = True
                path.append(next_node)
                
                if self._hamiltonian_dfs(graph, next_node, visited, path, n):
                    return True
                
                path.pop()
                visited[next_node] = False
        
        return False