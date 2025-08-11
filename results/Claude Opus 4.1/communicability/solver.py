import numpy as np
from scipy.linalg import expm

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Calculates the communicability for the graph using matrix exponential.
        
        Args:
            problem: A dictionary containing the adjacency list of the graph.
                     {"adjacency_list": adj_list}
        
        Returns:
            A dictionary containing the communicability matrix (as dict of dicts).
            {"communicability": comm_dict}
        """
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n == 0:
            return {"communicability": {}}
        
        # Build adjacency matrix directly
        adj_matrix = np.zeros((n, n), dtype=np.float64)
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                adj_matrix[u, v] = 1.0
        
        # Compute matrix exponential
        exp_matrix = expm(adj_matrix)
        
        # Convert to dictionary format
        result_comm_dict = {}
        for u in range(n):
            result_comm_dict[u] = {}
            for v in range(n):
                result_comm_dict[u][v] = float(exp_matrix[u, v])
        
        return {"communicability": result_comm_dict}