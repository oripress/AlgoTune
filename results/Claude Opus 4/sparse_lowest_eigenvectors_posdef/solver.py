class Solver:
    def solve(self, problem):
        import numpy as np
        
        mat = problem["matrix"]
        k = problem["k"]
        
        # Handle sparse matrix
        if hasattr(mat, 'toarray'):
            mat = mat.toarray()
        
        # Get eigenvalues
        vals = np.linalg.eigvalsh(mat)
        
        # Return k smallest
        return list(vals[:k])