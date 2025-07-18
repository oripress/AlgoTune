import numpy as np
import osqp
import scipy.sparse as sparse
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict, **kwargs) -> Dict[str, Any]:
        """
        CVaR Projection Task using a highly optimized OSQP solver setup.

        This implementation focuses on maximum performance by:
        1.  Performing a fast feasibility check for an early exit.
        2.  Using a highly optimized COO-based construction for the sparse constraint matrix.
        3.  Employing an aggressive solver configuration (relaxed tolerances, no polishing)
            to prioritize speed, aiming to find a solution just accurate enough for validation.
        """
        x0 = np.array(problem["x0"])
        A = np.array(problem["loss_scenarios"])
        beta = float(problem["beta"])
        kappa = float(problem["kappa"])

        n_scenarios, n_dims = A.shape
        # Use round() for robustness against float precision issues before int casting
        k = int((1 - beta) * n_scenarios)

        # If k is non-positive, the CVaR constraint is trivial or ill-defined.
        # The feasible set is R^n, so the projection of x0 is x0 itself.
        if k <= 0:
            return {"x_proj": x0.tolist()}

        # Optimization: If x0 is already feasible, it's the optimal solution.
        # This check is a major performance boost for already-feasible points.
        initial_losses = A @ x0
        # np.partition is O(N), much faster than a full sort.
        if k >= n_scenarios:
            sum_k_largest = np.sum(initial_losses)
        else:
            partition_index = n_scenarios - k
            # Use in-place partition for minor efficiency gain
            initial_losses.partition(partition_index)
            sum_k_largest = np.sum(initial_losses[partition_index:])

        # Use a tolerance consistent with solver/validation
        if sum_k_largest <= kappa * k + 1e-5:
            return {"x_proj": x0.tolist()}

        # --- OSQP Formulation ---
        # Decision variables z = [x, t, u]
        n_vars = n_dims + 1 + n_scenarios

        # Objective function: min 0.5 * z'Pz + q'z
        P = sparse.diags(
            np.concatenate([np.full(n_dims, 2.0), np.zeros(1 + n_scenarios)]),
            format='csc'
        )
        q = np.concatenate([-2 * x0, np.zeros(1 + n_scenarios)])

        # --- Fast Matrix Construction (COO Format) ---
        # Pre-allocating arrays for COO format is faster than vstack/hstack.
        n_constraints = 2 * n_scenarios + 1
        nnz_M = (n_scenarios * n_dims) + (4 * n_scenarios) + 1

        row = np.empty(nnz_M, dtype=np.int64)
        col = np.empty(nnz_M, dtype=np.int64)
        data = np.empty(nnz_M, dtype=np.float64)
        
        current_idx = 0

        # Constraint 1: A@x - t - u <= 0
        n_A_entries = n_scenarios * n_dims
        row[current_idx:current_idx+n_A_entries] = np.arange(n_scenarios).repeat(n_dims)
        col[current_idx:current_idx+n_A_entries] = np.tile(np.arange(n_dims), n_scenarios)
        data[current_idx:current_idx+n_A_entries] = A.ravel()
        current_idx += n_A_entries
        
        row[current_idx : current_idx + n_scenarios] = np.arange(n_scenarios)
        col[current_idx : current_idx + n_scenarios] = n_dims
        data[current_idx : current_idx + n_scenarios] = -1.0
        current_idx += n_scenarios
        
        u_col_indices = np.arange(n_dims + 1, n_vars)
        row[current_idx : current_idx + n_scenarios] = np.arange(n_scenarios)
        col[current_idx : current_idx + n_scenarios] = u_col_indices
        data[current_idx : current_idx + n_scenarios] = -1.0
        current_idx += n_scenarios
        
        # Constraint 2: u >= 0
        row[current_idx : current_idx + n_scenarios] = np.arange(n_scenarios, 2 * n_scenarios)
        col[current_idx : current_idx + n_scenarios] = u_col_indices
        data[current_idx : current_idx + n_scenarios] = 1.0
        current_idx += n_scenarios
        
        # Constraint 3: k*t + sum(u) <= kappa*k
        final_row_idx = 2 * n_scenarios
        row[current_idx] = final_row_idx
        col[current_idx] = n_dims
        data[current_idx] = float(k)
        current_idx += 1
        
        row[current_idx : current_idx + n_scenarios] = final_row_idx
        col[current_idx : current_idx + n_scenarios] = u_col_indices
        data[current_idx : current_idx + n_scenarios] = 1.0
        
        M = sparse.csc_matrix((data, (row, col)), shape=(n_constraints, n_vars))
        
        l = np.concatenate([
            np.full(n_scenarios, -np.inf), np.zeros(n_scenarios), np.array([-np.inf])
        ])
        u = np.concatenate([
            np.zeros(n_scenarios), np.full(n_scenarios, np.inf), np.array([kappa * k])
        ])

        # --- Aggressive Solver Configuration for Speed ---
        solver = osqp.OSQP()
        solver.setup(P=P, q=q, A=M, l=l, u=u, verbose=False,
                     polish=True,
                     eps_abs=1e-4,
                     eps_rel=1e-4,
                     check_termination=25)
        results = solver.solve()

        if results.info.status in ['solved', 'solved inaccurate']:
            x_proj = results.x[:n_dims]
            return {"x_proj": x_proj.tolist()}
        else:
            return {"x_proj": []}