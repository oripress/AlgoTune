import numpy as np
from scipy.linalg import expm
from typing import Any
import math

def matrix_exponential_taylor_optimized(A):
    """Compute matrix exponential with optimized Taylor series."""
    n = A.shape[0]
    
    # For very small matrices, use fewer terms
    if n <= 3:
        max_terms = 20
    elif n <= 5:
        max_terms = 30
    elif n <= 10:
        max_terms = 40
    else:
        # For larger matrices, fall back to scipy's expm
        return expm(A)
    
    # Pre-compute factorials more efficiently
    factorials = np.ones(max_terms + 1, dtype=np.float64)
    for k in range(1, max_terms + 1):
        factorials[k] = factorials[k-1] * k
    
    # Use optimized computation with better memory layout
    result = np.eye(n, dtype=np.float64)
    term = np.eye(n, dtype=np.float64)
    
    # Pre-allocate for better performance
    temp = np.empty_like(A)
    
    for k in range(1, max_terms + 1):
        # Use np.dot for potentially better performance
        np.dot(term, A, out=temp)
        term[:] = temp
        result += term / factorials[k]
        
    return result

class Solver:
    def solve(self, problem: dict[str, list[list[int]]], **kwargs) -> dict[str, dict[int, dict[int, float]]]:
        """
        Calculates the communicability with optimized Taylor series for small matrices.
        
        Args:
            problem: A dictionary containing the adjacency list of the graph.
                     {"adjacency_list": adj_list}
        
        Returns:
            A dictionary containing the communicability matrix (as dict of dicts).
            {"communicability": comm_dict}
            where comm_dict[u][v] is the communicability between nodes u and v.
        """
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n == 0:
            return {"communicability": {}}
        
        # Create adjacency matrix more efficiently using direct assignment
        A = np.zeros((n, n), dtype=np.float64)
        for i, neighbors in enumerate(adj_list):
            for j in neighbors:
                A[i, j] = 1.0
        
        # Use optimized Taylor series for small matrices, scipy for larger ones
        expA = matrix_exponential_taylor_optimized(A)
        
        # Convert to required dictionary format
        expA_list = expA.tolist()
        comm_dict = {}
        for i in range(n):
            inner_dict = {}
            row = expA_list[i]
            for j in range(n):
                inner_dict[j] = row[j]
            comm_dict[i] = inner_dict
        
        return {"communicability": comm_dict}