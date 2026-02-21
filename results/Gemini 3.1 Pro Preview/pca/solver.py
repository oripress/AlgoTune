import numpy as np
import scipy.linalg
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        X = np.array(problem["X"], dtype=float, copy=False)
        n_components = problem["n_components"]
        
        m, n = X.shape
        if n_components == 0:
            return np.zeros((0, n))
            
        X -= np.add.reduce(X, axis=0) / m
        
        if m >= n:
            C = np.dot(X.T, X)
            if n_components < n // 2:
                evals, evecs = scipy.linalg.eigh(C, subset_by_index=[n - n_components, n - 1])
                V = evecs[:, ::-1].T
            else:
                evals, evecs = np.linalg.eigh(C)
                V = evecs[:, -n_components:][:, ::-1].T
        else:
            C = np.dot(X, X.T)
            if n_components < m // 2:
                evals, evecs = scipy.linalg.eigh(C, subset_by_index=[m - n_components, m - 1])
                evecs = evecs[:, ::-1]
            else:
                evals, evecs = np.linalg.eigh(C)
                evecs = evecs[:, -n_components:][:, ::-1]
                
            V = np.dot(X.T, evecs)
            norms = np.sqrt(np.einsum('ij,ij->j', V, V))
            norms[norms == 0] = 1.0
            V /= norms
            V = V.T
            
        return V