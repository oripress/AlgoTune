from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from functools import partial

class Solver:
    def __init__(self):
        # Pre-compile JAX function
        self._solve_jax = jax.jit(self._randomized_svd_jax, static_argnums=(2, 3))
    
    @staticmethod
    def _randomized_svd_jax(A, key, n_components, n_iter):
        n, m = A.shape
        n_oversamples = 5
        k = n_components + n_oversamples
        k = min(k, min(n, m))
        
        # Generate random projection matrix
        Omega = random.normal(key, (m, k))
        
        # Project A onto random subspace
        Y = A @ Omega
        
        # QR decomposition
        Q, _ = jnp.linalg.qr(Y, mode='reduced')
        
        # Power iterations
        def power_iter(Q, _):
            Q, _ = jnp.linalg.qr(A @ (A.T @ Q), mode='reduced')
            return Q, None
        
        Q, _ = jax.lax.scan(power_iter, Q, None, length=n_iter)
        
        # Project A onto the subspace
        B = Q.T @ A
        
        # SVD of the small matrix
        U_B, s, Vt = jnp.linalg.svd(B, full_matrices=False)
        
        # Recover U
        U = Q @ U_B
        
        return U[:, :n_components], s[:n_components], Vt[:n_components, :].T
    
    def solve(self, problem, **kwargs):
        A = np.asarray(problem["matrix"], dtype=np.float64)
        n_components = problem["n_components"]
        matrix_type = problem.get("matrix_type", "default")
        
        # Determine number of power iterations based on matrix type
        if matrix_type == "ill_conditioned":
            n_iter = 3
        elif matrix_type == "low_rank":
            n_iter = 1
        else:
            n_iter = 2
        
        key = random.PRNGKey(42)
        A_jax = jnp.array(A)
        
        U, s, V = self._solve_jax(A_jax, key, n_components, n_iter)
        
        return {"U": np.array(U), "S": np.array(s), "V": np.array(V)}