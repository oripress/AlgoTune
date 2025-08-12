from typing import Any
import numpy as np
import numba

@numba.jit(nopython=True, fastmath=True)
def _frank_wolfe_loop(points, u, max_iter, conv_tol, d):
    """
    Numba-jitted core loop for the Frank-Wolfe algorithm.
    This function is compiled to machine code for maximum performance.
    """
    epsilon_eye = 1e-9 * np.eye(d)
    n = points.shape[0]

    for _ in range(max_iter):
        c = points.T @ u
        P_centered = points - c
        M = (P_centered.T * u) @ P_centered

        M_inv = np.linalg.inv(M + epsilon_eye)
        
        # Vectorized calculation of Mahalanobis distances
        g = np.sum((P_centered @ M_inv) * P_centered, axis=1)
        
        j = np.argmax(g)
        g_max = g[j]

        if g_max / d < 1 + conv_tol:
            break

        # Add epsilon to denominator for stability
        step = (g_max / d - 1) / (g_max - 1 + 1e-12)
        
        u *= (1 - step)
        u[j] += step
    return u

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        points = np.array(problem["points"], dtype=np.float64)
        n, d = points.shape

        if n <= d:
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.nan * np.ones((d, d)), "Y": np.nan * np.ones((d,))},
            }

        # Initial weights and parameters for the jitted loop
        u = np.ones(n, dtype=np.float64) / n
        max_iter = 1000
        conv_tol = 1e-5

        try:
            # Call the Numba-jitted core solver
            u = _frank_wolfe_loop(points, u, max_iter, conv_tol, d)

            # After iterations, compute the final ellipsoid from the returned u
            c = points.T @ u
            P_centered = points - c
            M = (P_centered.T * u) @ P_centered
            
            A_ellipsoid = (1/d) * np.linalg.inv(M + 1e-9 * np.eye(d))

            eigvals, eigvecs = np.linalg.eigh(A_ellipsoid)
            
            if np.any(eigvals <= 1e-9):
                raise np.linalg.LinAlgError("Matrix is not positive definite")

            X = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            Y = -X @ c

            log_det_X = 0.5 * np.sum(np.log(eigvals))
            objective_value = -log_det_X

            # Final scaling step to guarantee feasibility
            norms = np.linalg.norm(points @ X + Y, axis=1)
            s_max = np.max(norms)

            if s_max > 1.0:
                X /= s_max
                Y /= s_max
                objective_value += d * np.log(s_max)

            return {
                "objective_value": objective_value,
                "ellipsoid": {"X": X, "Y": Y}
            }

        except (np.linalg.LinAlgError, ValueError):
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.nan * np.ones((d, d)), "Y": np.nan * np.ones((d,))},
            }