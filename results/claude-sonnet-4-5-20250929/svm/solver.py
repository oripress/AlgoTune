from typing import Any, Dict
import numpy as np
import osqp
import scipy.sparse as sp

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimized SVM solver using OSQP with highly efficient matrix construction.
        """
        X = np.array(problem["X"], dtype=np.float64)
        y = np.array(problem["y"], dtype=np.float64)
        C = float(problem["C"])
        
        n, p = X.shape
        num_vars = p + 1 + n
        
        # P matrix: diagonal [1,...,1 (p), 0, 0,...,0 (n+1)]
        P_data = np.zeros(num_vars)
        P_data[:p] = 1.0
        P = sp.diags(P_data, format='csc')
        
        # q vector: [0,...,0 (p+1), C,...,C (n)]
        q = np.zeros(num_vars)
        q[p+1:] = C
        
        # Build A in COO format directly
        # Constraint 1: -xi <= 0 => [0 (p+1) | -I (n)]
        # Constraint 2: y*(X@beta + beta0) + xi >= 1 => [yX | y | I]
        
        yX = y.reshape(-1, 1) * X
        
        # A1 indices: -I block at columns p+1 to p+n
        row_A1 = np.arange(n)
        col_A1 = np.arange(p + 1, p + 1 + n)
        data_A1 = -np.ones(n)
        
        # A2 indices
        # yX part: rows n to 2n-1, cols 0 to p-1
        row_A2_yX = np.repeat(np.arange(n, 2*n), p)
        col_A2_yX = np.tile(np.arange(p), n)
        data_A2_yX = yX.ravel()
        
        # y part: rows n to 2n-1, col p
        row_A2_y = np.arange(n, 2*n)
        col_A2_y = np.full(n, p)
        data_A2_y = y
        
        # I part: rows n to 2n-1, cols p+1 to p+n
        row_A2_I = np.arange(n, 2*n)
        col_A2_I = np.arange(p + 1, p + 1 + n)
        data_A2_I = np.ones(n)
        
        # Concatenate all
        row_ind = np.concatenate([row_A1, row_A2_yX, row_A2_y, row_A2_I])
        col_ind = np.concatenate([col_A1, col_A2_yX, col_A2_y, col_A2_I])
        data = np.concatenate([data_A1, data_A2_yX, data_A2_y, data_A2_I])
        
        A = sp.coo_matrix((data, (row_ind, col_ind)), shape=(2*n, num_vars)).tocsc()
        
        # Bounds
        l = np.concatenate([-np.inf * np.ones(n), np.ones(n)])
        u = np.concatenate([np.zeros(n), np.inf * np.ones(n)])
        
        # Setup and solve
        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, verbose=False, eps_abs=1e-5, eps_rel=1e-5, 
                   polish=True, max_iter=10000)
        res = prob.solve()
        
        if res.info.status != 'solved' and res.info.status != 'solved inaccurate':
            return None
        
        # Extract solution
        beta = res.x[:p]
        beta0 = res.x[p]
        
        # Calculate optimal value and misclassification
        optimal_value = 0.5 * np.dot(beta, beta) + C * np.sum(res.x[p+1:])
        pred = X @ beta + beta0
        missclass = np.mean((pred * y) < 0)
        
        return {
            "beta0": float(beta0),
            "beta": beta.tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": float(missclass),
        }