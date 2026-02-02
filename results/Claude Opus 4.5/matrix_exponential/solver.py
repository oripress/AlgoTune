import numpy as np
import torch
from scipy.linalg import expm

class Solver:
    def solve(self, problem, **kwargs):
        A = problem["matrix"]
        if not isinstance(A, np.ndarray):
            A = np.array(A, dtype=np.float64)
        
        n = A.shape[0]
        
        # For smaller matrices, try torch's matrix_exp
        if n <= 256:
            try:
                A_torch = torch.from_numpy(A)
                expA_torch = torch.matrix_exp(A_torch)
                expA = expA_torch.numpy()
            except:
                expA = expm(A)
        else:
            expA = expm(A)
        
        return {"exponential": expA}