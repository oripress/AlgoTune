from typing import Any
import numpy as np
import numba

@numba.njit(cache=True, fastmath=True)
def _solve_lqr(A, B, Q, R, P, T, x0):
    n, m = B.shape
    K = np.zeros((T, m, n))
    S = P.copy()
    A_T = A.T
    B_T = B.T
    regularization = np.eye(m) * 1e-9

    # Backward pass: Iterate from T-1 down to 0
    # This version is optimized by pre-calculating matrix products
    # to reduce redundant computations within the loop.
    for t in range(T - 1, -1, -1):
        # Pre-calculate products involving S
        S_A = S @ A
        S_B = S @ B
        
        # Calculate terms for the gain K
        M1 = R + B_T @ S_B
        M2 = B_T @ S_A
        
        # Solve for the gain k at time t
        k = np.linalg.solve(M1 + regularization, M2)
        K[t] = k
        
        # Update S for the next iteration (S_{t-1})
        # Reuse pre-calculated terms to update S
        S = Q + A_T @ S_A - k.T @ M2
        
        # Enforce symmetry to prevent numerical drift
        S = (S + S.T) * 0.5

    # Forward pass to compute the optimal control sequence U
    U = np.zeros((T, m))
    x = x0.copy()
    for t in range(T):
        u = -K[t] @ x
        U[t, :] = u.ravel()
        x = A @ x + B @ u
        
    return U

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Computes the optimal LQR control sequence.
        This implementation uses a Numba-jitted function with optimized
        matrix calculations in the core Riccati recursion loop.
        """
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        Q = np.array(problem["Q"], dtype=np.float64)
        R = np.array(problem["R"], dtype=np.float64)
        P = np.array(problem["P"], dtype=np.float64)
        T = problem["T"]
        x0 = np.array(problem["x0"], dtype=np.float64).reshape(-1, 1)

        U = _solve_lqr(A, B, Q, R, P, T, x0)

        return {"U": U.tolist()}