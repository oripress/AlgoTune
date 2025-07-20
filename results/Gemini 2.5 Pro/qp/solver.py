from typing import Any
import osqp
import numpy as np
from scipy import sparse

class Solver:
    """
    A high-performance QP solver using OSQP with a caching mechanism.
    
    This solver caches pre-factorized KKT systems based on the sparsity
    pattern of the problem matrices. If a problem with the same structure
    is solved again, it uses the fast `update` method instead of the slow
    `setup` method.
    """
    def __init__(self):
        """
        Initializes the solver and its cache.
        """
        self._solvers_cache: dict[tuple, osqp.OSQP] = {}

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves a convex quadratic program using OSQP, with caching.
        """
        # 1. Efficiently extract and prepare problem data
        P_in = problem.get("P", problem.get("Q", []))
        q_in = problem["q"]
        G_in = problem["G"]
        h_in = problem["h"]
        A_in = problem["A"]
        b_in = problem["b"]

        n = len(q_in)

        # Construct sparse matrices directly, providing shape to handle empty inputs
        P = sparse.csc_matrix(P_in, shape=(n, n), dtype=np.float64)
        G = sparse.csc_matrix(G_in, shape=(len(h_in), n), dtype=np.float64)
        A = sparse.csc_matrix(A_in, shape=(len(b_in), n), dtype=np.float64)

        # OSQP uses the upper-triangular part of P.
        if P.nnz > 0:
            P = sparse.triu(P, format='csc')

        q = np.array(q_in, dtype=np.float64)
        h = np.array(h_in, dtype=np.float64)
        b = np.array(b_in, dtype=np.float64)

        # Combine constraints for OSQP format: l <= Ax <= u
        A_osqp = sparse.vstack([G, A], format='csc')
        l = np.concatenate([np.full(G.shape[0], -np.inf), b])
        u = np.concatenate([h, b])

        # 2. Use advanced caching
        # Create a cache key from the matrix shapes and sparsity patterns.
        cache_key = (
            P.shape, A_osqp.shape,
            P.indptr.tobytes(), P.indices.tobytes(),
            A_osqp.indptr.tobytes(), A_osqp.indices.tobytes()
        )

        if cache_key in self._solvers_cache:
            # Cache hit: use the much faster `update` method
            solver = self._solvers_cache[cache_key]
            solver.update(q=q, l=l, u=u, Px=P.data, Ax=A_osqp.data)
        else:
            # Cache miss: perform a one-time setup for this problem structure
            solver = osqp.OSQP()
            solver.setup(P=P, q=q, A=A_osqp, l=l, u=u,
                         verbose=False,
                         # Use strict tolerances for correctness
                         eps_abs=1e-8,
                         eps_rel=1e-8,
                         max_iter=10000,
                         warm_start=True)
            self._solvers_cache[cache_key] = solver

        # 3. Solve the problem and handle return values gracefully
        result = solver.solve()

        solution_values = result.x
        objective_value = result.info.obj_val

        # If solver fails, it might return None for x. The validator expects
        # a list of floats of the correct dimension. We provide a vector of
        # NaNs, which will correctly fail validation without crashing.
        if solution_values is None:
            solution_values = np.full(n, np.nan)

        # The objective can be +/- inf for infeasible problems.
        # The validator expects a float.
        if not np.isfinite(objective_value):
            objective_value = np.nan

        return {
            "solution": solution_values.tolist(),
            "objective": float(objective_value)
        }