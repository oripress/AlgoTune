import numpy as np
import highspy
from scipy.sparse import eye, kron, vstack
from typing import Any

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the EMD problem using the highspy library.
        This version passes the complete model as arguments to passModel.
        """
        a = np.array(problem["source_weights"], dtype=np.float64)
        b = np.array(problem["target_weights"], dtype=np.float64)
        M = np.array(problem["cost_matrix"], dtype=np.float64)

        n, m = M.shape
        num_vars = n * m
        num_cons = n + m

        # Objective function - use flatten() to ensure a contiguous copy
        c = M.flatten()

        # Constraint matrix (CSC format)
        A1 = kron(eye(n), np.ones((1, m)))
        A2 = kron(np.ones((1, n)), eye(m))
        A_eq = vstack([A1, A2]).tocsc()  # Use column-wise format

        # Ensure correct integer types (int32) for HiGHS C++ backend
        a_starts = A_eq.indptr.astype(np.int32)
        a_indices = A_eq.indices.astype(np.int32)
        a_values = A_eq.data
        num_nz = A_eq.nnz

        # Constraint bounds (RHS)
        b_eq = np.concatenate((a, b))

        # Variable bounds
        col_lower = np.zeros(num_vars)
        col_upper = np.full(num_vars, highspy.kHighsInf)

        # --- Instantiate, pass model, and solve with HiGHS ---
        h = highspy.Highs()
        h.setOptionValue('log_to_console', False)

        # Pass the complete model to HiGHS using the full argument list
        h.passModel(
            int(num_vars),
            int(num_cons),
            int(num_nz),
            int(highspy.MatrixFormat.kColwise),  # Specify column-wise
            int(highspy.ObjSense.kMinimize),
            0.0,  # offset
            c,
            col_lower,
            col_upper,
            b_eq,  # row_lower
            b_eq,  # row_upper
            a_starts,
            a_indices,
            a_values
        )

        # Solve the LP
        h.run()

        # Get the solution
        solution = h.getSolution()
        G = np.array(solution.col_value).reshape((n, m))

        return {"transport_plan": G}