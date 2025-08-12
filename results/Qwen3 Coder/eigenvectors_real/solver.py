import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        # Compute eigenvalues and eigenvectors using eigh for symmetric matrices
        eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(problem))
        
        # Reverse order to have descending eigenvalues and corresponding eigenvectors
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
        # Convert to lists using numpy operations
        eigenvalues_list = eigenvalues.tolist()
        eigenvectors_list = eigenvectors.T.tolist()
        return (eigenvalues_list, eigenvectors_list)