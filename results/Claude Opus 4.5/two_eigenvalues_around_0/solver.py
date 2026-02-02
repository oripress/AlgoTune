import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh as scipy_eigvalsh

class Solver:
    def __init__(self):
        # Check if CUDA is available
        self.use_torch = True
        
    def solve(self, problem, **kwargs):
        matrix_list = problem["matrix"]
        n = len(matrix_list)
        
        # For smaller matrices, use scipy which is well optimized
        if n <= 50:
            matrix = np.array(matrix_list, dtype=np.float64)
            eigenvalues = scipy_eigvalsh(matrix, driver='evr')
            abs_eigs = np.abs(eigenvalues)
            idx = np.argpartition(abs_eigs, 1)[:2]
            result = eigenvalues[idx]
            sorted_idx = np.argsort(np.abs(result))
            return [float(result[sorted_idx[0]]), float(result[sorted_idx[1]])]
        
        # For medium matrices, use scipy
        if n <= 200:
            matrix = np.array(matrix_list, dtype=np.float64)
            eigenvalues = scipy_eigvalsh(matrix, driver='evr')
            abs_eigs = np.abs(eigenvalues)
            idx = np.argpartition(abs_eigs, 1)[:2]
            result = eigenvalues[idx]
            sorted_idx = np.argsort(np.abs(result))
            return [float(result[sorted_idx[0]]), float(result[sorted_idx[1]])]
        
        # For larger matrices, use eigsh with shift-invert mode
        matrix = np.array(matrix_list, dtype=np.float64)
        try:
            ncv = min(n - 1, max(20, 10))
            eigenvalues = eigsh(matrix, k=2, sigma=0, which='LM',
                              return_eigenvectors=False, ncv=ncv, tol=1e-7)
            sorted_idx = np.argsort(np.abs(eigenvalues))
            return [float(eigenvalues[sorted_idx[0]]), float(eigenvalues[sorted_idx[1]])]
        except Exception:
            eigenvalues = scipy_eigvalsh(matrix, driver='evr')
            abs_eigs = np.abs(eigenvalues)
            idx = np.argpartition(abs_eigs, 1)[:2]
            result = eigenvalues[idx]
            sorted_idx = np.argsort(np.abs(result))
            return [float(result[sorted_idx[0]]), float(result[sorted_idx[1]])]