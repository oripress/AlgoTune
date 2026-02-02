import numpy as np
from typing import Any
import highspy
try:
    import solver_utils
except ImportError:
    solver_utils = None

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        X = np.array(problem["X"], dtype=float)
        y = np.array(problem["y"], dtype=float)
        quantile = problem["quantile"]
        fit_intercept = problem["fit_intercept"]
        
        n_samples, n_features = X.shape
        m_constraints = n_features + (1 if fit_intercept else 0)
        
        if solver_utils:
            if not X.flags.c_contiguous:
                X = np.ascontiguousarray(X)
            a_values, a_indices, a_start, col_cost, col_lower, col_upper = solver_utils.prepare_model_data(X, y, quantile, fit_intercept)
        else:
            if fit_intercept:
                X_ext = np.empty((n_samples, m_constraints), dtype=np.float64)
                X_ext[:, :n_features] = X
                X_ext[:, n_features] = 1.0
                a_values = X_ext.ravel()
            else:
                a_values = X.ravel()
                
            a_indices = np.tile(np.arange(m_constraints, dtype=np.int32), n_samples)
            a_start = np.arange(0, (n_samples + 1) * m_constraints, m_constraints, dtype=np.int32)
            col_cost = -y
            col_lower = np.full(n_samples, quantile - 1.0, dtype=np.float64)
            col_upper = np.full(n_samples, quantile, dtype=np.float64)

        num_col = n_samples
        num_row = m_constraints
        a_nnz = len(a_values)
        
        row_lower = np.zeros(num_row, dtype=np.float64)
        row_upper = np.zeros(num_row, dtype=np.float64)
        
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        
        # Ensure types
        a_values = a_values.astype(np.float64, copy=False)
        col_cost = col_cost.astype(np.float64, copy=False)
        col_lower = col_lower.astype(np.float64, copy=False)
        col_upper = col_upper.astype(np.float64, copy=False)
        a_indices = a_indices.astype(np.int32, copy=False)
        a_start = a_start.astype(np.int32, copy=False)
        
        lp = highspy.HighsLp()
        lp.num_col_ = num_col
        lp.num_row_ = num_row
        lp.a_matrix_.start_ = a_start
        lp.a_matrix_.index_ = a_indices
        lp.a_matrix_.value_ = a_values
        lp.col_cost_ = col_cost
        lp.col_lower_ = col_lower
        lp.col_upper_ = col_upper
        lp.row_lower_ = row_lower
        lp.row_upper_ = row_upper
        
        h.passModel(lp)
        
        h.run()
        
        sol = h.getSolution()
        duals = np.array(sol.row_dual)
        coefs_raw = -duals
        
        if fit_intercept:
            coef = coefs_raw[:n_features]
            intercept = coefs_raw[n_features:]
        else:
            coef = coefs_raw
            intercept = np.array([0.0])
            
        predictions = X @ coef + intercept[0]
        
        return {
            "coef": [coef.tolist()],
            "intercept": intercept.tolist(),
            "predictions": predictions.tolist()
        }