import numpy as np
from scipy import sparse

class Solver:
    def solve(self, problem, **kwargs):
        A = problem["matrix"]
        n = A.shape[0]
        
        # Simple Taylor series approximation for matrix exponential
        I = sparse.eye(n, format='csc')
        result = I.copy()
        term = I.copy()
        k = 1
        max_terms = 15
        
        while k < max_terms:
            term = term.dot(A) / k
            result = result + term
            k += 1
            
        return result