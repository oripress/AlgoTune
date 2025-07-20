from typing import Any
import numpy as np
# Using highspy directly to avoid scipy.optimize.linprog overhead.
import highspy
# Use csr_matrix for a more direct conversion from the dense numpy array.
from scipy.sparse import csr_matrix
# Import os to leverage multi-core processing for the solver.
import os

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list]:
        """
        Solve the lp box problem using the highspy library directly.
        This version is optimized by enabling parallel execution in HiGHS
        and using a more direct sparse matrix conversion.
        """
        # Convert lists to numpy arrays for efficient processing.
        c = np.array(problem["c"], dtype=np.float64)
        A = np.array(problem["A"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        
        m, n = A.shape
        
        # Create a Highs instance.
        h = highspy.Highs()

        # --- Optimizations ---
        # 1. Suppress solver log output to avoid I/O overhead.
        h.setOptionValue('output_flag', False)
        
        # 2. Enable parallel execution. Using all available cores significantly
        # speeds up the simplex or interior-point algorithms in HiGHS.
        # This is the most impactful optimization.
        num_threads = os.cpu_count()
        if num_threads:
            h.setOptionValue('threads', num_threads)

        # --- Build the model incrementally ---
        
        # 1. Add variables and their box constraints (0 <= x <= 1).
        h.addVars(n, np.zeros(n), np.ones(n))
        
        # 2. Set the objective function (minimize c'x).
        h.changeObjectiveSense(highspy.ObjSense.kMinimize)
        h.changeColsCost(n, np.arange(n, dtype=np.int32), c)
        
        # 3. Add the constraints (Ax <= b).
        # Directly create a CSR matrix from the dense numpy array. This is
        # more efficient than the previous csc_matrix(A).tocsr() approach.
        A_csr = csr_matrix(A)
        
        # Define row bounds: -infinity <= Ax <= b.
        row_lower = np.full(m, -highspy.kHighsInf)
        
        h.addRows(
            m, 
            row_lower, 
            b,
            A_csr.nnz, 
            A_csr.indptr.astype(np.int32), 
            A_csr.indices.astype(np.int32), 
            A_csr.data
        )
        
        # Run the solver.
        h.run()
        
        # Get the solution.
        solution = h.getSolution()
        
        # The solution object from HiGHS contains the primal solution (col_value)
        # as a standard Python list, so no conversion is necessary.
        return {"solution": solution.col_value}