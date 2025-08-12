import numpy as np
from scipy.linalg import svd
import numba

# Numba-optimized unfolding functions
@numba.njit
def unfold1_numba(X, dim1, dim2, dim3):
    return X.reshape((dim1, dim2 * dim3))

@numba.njit
def unfold2_numba(X, dim1, dim2, dim3):
    out = np.zeros((dim2, dim1 * dim3))
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                out[j, i * dim3 + k] = X[i, j, k]
    return out

@numba.njit
def unfold3_numba(X, dim1, dim2, dim3):
    out = np.zeros((dim3, dim1 * dim2))
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                out[k, i * dim2 + j] = X[i, j, k]
    return out

# Numba-optimized folding functions
@numba.njit
def fold1_numba(mat, dim1, dim2, dim3):
    return mat.reshape((dim1, dim2, dim3))

@numba.njit
def fold2_numba(mat, dim1, dim2, dim3):
    out = np.zeros((dim1, dim2, dim3))
    mat_reshaped = mat.reshape((dim2, dim1, dim3))
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                out[i, j, k] = mat_reshaped[j, i, k]
    return out

@numba.njit
def fold3_numba(mat, dim1, dim2, dim3):
    out = np.zeros((dim1, dim2, dim3))
    mat_reshaped = mat.reshape((dim3, dim1, dim2))
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                out[i, j, k] = mat_reshaped[k, i, j]
    return out

# Numba-optimized SVT
@numba.njit
def SVT_numba(Y, tau):
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0)
    return U @ np.diag(s_thresh) @ Vt

class Solver:
    def solve(self, problem, **kwargs):
        # Extract problem data
        observed_tensor = np.array(problem["tensor"])
        mask = np.array(problem["mask"])
        tensor_dims = observed_tensor.shape
        dim1, dim2, dim3 = tensor_dims
        
        # Precompute observed data
        obs_data = observed_tensor[mask]
        
        # ADMM parameters
        rho = 1.0
        max_iters = 100
        tol = 1e-5
        
        # Initialize variables
        X = np.zeros(tensor_dims)
        Z1 = np.zeros((dim1, dim2 * dim3))
        Z2 = np.zeros((dim2, dim1 * dim3))
        Z3 = np.zeros((dim3, dim1 * dim2))
        U1 = np.zeros_like(Z1)
        U2 = np.zeros_like(Z2)
        U3 = np.zeros_like(Z3)
        
        # Precompute constants
        rho_inv = 1 / rho
        
        # ADMM iterations
        for it in range(max_iters):
            X_prev = X.copy()
            
            # Update X by averaging the three foldings
            term1 = fold1_numba(Z1 - U1, dim1, dim2, dim3)
            term2 = fold2_numba(Z2 - U2, dim1, dim2, dim3)
            term3 = fold3_numba(Z3 - U3, dim1, dim2, dim3)
            
            X = (term1 + term2 + term3) / 3.0
            
            # Project observed entries
            X[mask] = obs_data
            
            # Update Z variables with SVT
            Z1 = SVT_numba(unfold1_numba(X, dim1, dim2, dim3) + U1, rho_inv)
            Z2 = SVT_numba(unfold2_numba(X, dim1, dim2, dim3) + U2, rho_inv)
            Z3 = SVT_numba(unfold3_numba(X, dim1, dim2, dim3) + U3, rho_inv)
            
            # Update dual variables
            U1 += unfold1_numba(X, dim1, dim2, dim3) - Z1
            U2 += unfold2_numba(X, dim1, dim2, dim3) - Z2
            U3 += unfold3_numba(X, dim1, dim2, dim3) - Z3
            
            # Check convergence
            diff = np.linalg.norm(X - X_prev) / (np.linalg.norm(X_prev) + 1e-8)
            if diff < tol:
                break
        
        return {"completed_tensor": X.tolist()}