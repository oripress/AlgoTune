import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov, solve
import logging

class Solver:
    def solve(self, problem, **kwargs):
        try:
            A = np.array(problem["A"], dtype=float)
            B = np.array(problem["B"], dtype=float)
            n, m = A.shape[0], B.shape[1]

            # Fast stability check using Frobenius norm
            # If ||A||_F < 1, then rho(A) <= ||A||_F < 1
            if np.linalg.norm(A) < 1.0 - 1e-9:
                P = solve_discrete_lyapunov(A.T, np.eye(n))
                K = np.zeros((m, n))
                return {
                    "is_stabilizable": True,
                    "K": K.tolist(),
                    "P": P.tolist()
                }

            # Slower stability check using eigenvalues
            # Check if triangular to speed up eigenvalue computation
            if np.all(np.tril(A, -1) == 0) or np.all(np.triu(A, 1) == 0):
                eig_vals = np.diagonal(A)
            else:
                eig_vals = np.linalg.eigvals(A)
            
            if np.max(np.abs(eig_vals)) < 1.0 - 1e-9:
                P = solve_discrete_lyapunov(A.T, np.eye(n))
                K = np.zeros((m, n))
                return {
                    "is_stabilizable": True,
                    "K": K.tolist(),
                    "P": P.tolist()
                }

            # If not stable, use LQR
            Q_cost = np.eye(n)
            R_cost = np.eye(m)

            P = solve_discrete_are(A, B, Q_cost, R_cost, balanced=False)

            B_T_P = B.T @ P
            R_total = R_cost + B_T_P @ B
            
            # Solve for K
            # If m=1, scalar division is faster
            if m == 1:
                K = -(B_T_P @ A) / R_total[0, 0]
            else:
                # Use scipy.linalg.solve with assume_a='pos' for SPD matrix
                K = -solve(R_total, B_T_P @ A, assume_a='pos')
            
            return {
                "is_stabilizable": True,
                "K": K.tolist(),
                "P": P.tolist()
            }

        except Exception:
            return {"is_stabilizable": False, "K": None, "P": None}
        except Exception:
            # If an error occurs (e.g. not stabilizable), return False
            return {"is_stabilizable": False, "K": None, "P": None}