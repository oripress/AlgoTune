import numpy as np
import scipy.linalg

class Solver:
    def solve(self, problem, **kwargs):
        A = np.array(problem["matrix"], dtype=np.float32)
        n_components = problem["n_components"]
        
        U, s, Vt = self.randomized_svd(A, n_components)
        
        return {"U": U, "S": s, "V": Vt.T}

    def randomized_svd(self, M, n_components, random_state=42):
        n_samples, n_features = M.shape
        n_oversamples = 0
        n_random = n_components + n_oversamples
        n_random = min(n_random, n_samples, n_features)
        
        rng = np.random.default_rng(random_state)
        
        transpose = n_samples < n_features
        if transpose:
            M = M.T
            n_samples, n_features = M.shape
            
        Omega = rng.random(size=(n_features, n_random), dtype=np.float32)
        Omega -= 0.5
        
        Y = M @ Omega
        
        Q, _ = scipy.linalg.qr(Y, mode='economic', check_finite=False, overwrite_a=True)
        
        B = Q.T @ M
        
        Uhat, s, Vt = scipy.linalg.svd(B, full_matrices=False, check_finite=False, overwrite_a=True)
        
        Uhat = Uhat[:, :n_components]
        U = Q @ Uhat
        
        s = s[:n_components]
        Vt = Vt[:n_components, :]
        
        if transpose:
            return Vt.T, s, U.T
        else:
            return U, s, Vt