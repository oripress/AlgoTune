import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        """
        Solve the optimal advertising problem using CVXPY with CLARABEL solver.
        Reformulated as LP with auxiliary variables for revenue per ad.
        Optimized with vectorized operations, solver prioritization, and post-processing.

        :param problem: Dictionary with problem parameters
        :return: Dictionary with optimal displays and revenue
        """
        try:
            # Convert inputs to numpy arrays
            P = np.array(problem["P"])
            R = np.array(problem["R"])
            B = np.array(problem["B"])
            c = np.array(problem["c"])
            T = np.array(problem["T"])
            
            # Get dimensions
            m, n = P.shape

            # Precompute R * P for efficiency
            RP = R[:, np.newaxis] * P

            # Create variables
            D = cp.Variable((m, n), nonneg=True)
            t = cp.Variable(m, nonneg=True)  # Revenue per ad must be non-negative

            # Objective: maximize total revenue
            total_revenue = cp.sum(t)

            # Constraints - use efficient matrix operations
            constraints = [
                # Revenue per ad constraints
                t <= cp.sum(cp.multiply(RP, D), axis=1),
                t <= B,
                # Traffic capacity per time slot
                cp.sum(D, axis=0) <= T,
                # Minimum display requirements
                cp.sum(D, axis=1) >= c,
            ]

            # Formulate and solve problem
            prob = cp.Problem(cp.Maximize(total_revenue), constraints)
            
            # Try fastest solvers first with tighter tolerances
            solvers = [
                (cp.CLARABEL, {'tol_gap_abs': 1e-6, 'tol_gap_rel': 1e-6, 'tol_feas': 1e-6, 'verbose': False}),
                (cp.ECOS, {'abstol': 1e-6, 'reltol': 1e-6, 'feastol': 1e-6, 'verbose': False}),
                (cp.SCS, {'eps': 1e-6, 'max_iters': 10000, 'verbose': False})
            ]
            for solver, options in solvers:
                if solver in cp.installed_solvers():
                    try:
                        prob.solve(solver=solver, **options)
                        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                            break
                    except:
                        continue

            # Check solution status
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return {"status": prob.status, "optimal": False}
            
            # Post-process to ensure constraints are strictly satisfied
            D_val = np.maximum(D.value, 0)  # Ensure non-negative
            
            # Clip column sums to traffic capacity
            col_sums = D_val.sum(axis=0)
            mask = col_sums > T
            if np.any(mask):
                scale_factors = np.where(mask, T / col_sums, 1.0)
                D_val = D_val * scale_factors

            # Recalculate results with adjusted displays
            clicks = (P * D_val).sum(axis=1)
            revenue = np.minimum(R * clicks, B)
            total_rev = revenue.sum()

            return {
                "displays": D_val.tolist(),
                "clicks": clicks.tolist(),
                "revenue_per_ad": revenue.tolist(),
                "total_revenue": float(total_rev),
                "optimal": True,
                "status": prob.status
            }

        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}