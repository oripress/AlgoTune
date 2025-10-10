import numpy as np
from numpy.typing import NDArray
from scipy import linalg

class Solver:
    def solve(self, problem: NDArray) -> tuple[list[float], list[list[float]]]:
        """
        Solve the eigenvalue problem for the given real symmetric matrix.
        """
        # scipy.linalg.eigh with driver='evd' uses divide-and-conquer, often faster
        eigenvalues, eigenvectors = linalg.eigh(problem, driver='evd')
        
        # Reverse to get descending order
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
        # Convert to lists more efficiently
        eigenvalues_list = eigenvalues.tolist()
        eigenvectors_list = eigenvectors.T.tolist()
        
        return (eigenvalues_list, eigenvectors_list)