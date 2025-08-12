import scipy.sparse
from scipy.sparse.linalg import expm
import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        # Get the input matrix
        matrix = problem["matrix"]
        
        # Calculate the matrix exponential
        exp_matrix = expm(matrix)
        
        # Convert to CSC format if needed
        result = exp_matrix.tocsc()
            
        return result