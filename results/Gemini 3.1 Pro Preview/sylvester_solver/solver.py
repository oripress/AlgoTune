from typing import Any
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

@jax.jit
def jax_solve(A, B, Q):
    D_A, V_A = jnp.linalg.eig(A)
    D_B, V_B = jnp.linalg.eig(B)
    
    Y = jnp.linalg.solve(V_A, Q)
    Q_tilde = Y @ V_B
    
    denom = D_A[:, None] + D_B[None, :]
    X_tilde = Q_tilde / denom
    
    Z = V_A @ X_tilde
    X = jnp.linalg.solve(V_B.T, Z.T).T
    return X

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        A, B, Q = problem["A"], problem["B"], problem["Q"]
        
        X = jax_solve(jnp.array(A), jnp.array(B), jnp.array(Q))
        
        return {"X": np.array(X)}