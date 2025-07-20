import numpy as np
from scipy import linalg

class Solver:
    def solve(self, problem, **kwargs):
        """Solve the generalized eigenvalue problem A·x = λ·B·x"""
        A, B = problem
        
        # Use scipy's eigh which is optimized for symmetric/hermitian matrices
        # It directly solves the generalized eigenvalue problem
        eigenvalues, eigenvectors = linalg.eigh(A, B)
        
        # Convert to lists in reverse order (descending)
        n = len(eigenvalues)
        eigenvalues_list = eigenvalues[::-1].tolist()
        
        # More efficient eigenvector conversion
        eigenvectors_list = [eigenvectors[:, n-1-i].tolist() for i in range(n)]
        
        return (eigenvalues_list, eigenvectors_list)