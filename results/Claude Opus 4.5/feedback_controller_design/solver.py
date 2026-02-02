import numpy as np
from scipy.linalg import solve_discrete_are, solve

class Solver:
    def solve(self, problem, **kwargs):
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        n, m = A.shape[0], B.shape[1]
        
        try:
            In = np.eye(n)
            Im = np.eye(m)
            
            # Solve DARE with balanced=False for speed
            P = solve_discrete_are(A, B, In, Im, balanced=False)
            
            # Compute K = -(R + B'PB)^{-1} B'PA using pos def solver
            BTP = B.T @ P
            K = -solve(Im + BTP @ B, BTP @ A, assume_a='pos')
            
            return {"is_stabilizable": True, "K": K.tolist(), "P": P.tolist()}
        except:
            return {"is_stabilizable": False, "K": None, "P": None}