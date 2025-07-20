import numpy as np
import cvxpy as cp
from multiprocessing import Pool

def solve_single(problem):
    c = np.array(problem["c"])
    A = np.array(problem["A"])
    b = np.array(problem["b"])
    n = c.shape[0]
    
    x = cp.Variable(n)
    objective = cp.Minimize(c.T @ x - cp.sum(cp.log(x)))
    constraints = [A @ x == b]
    prob = cp.Problem(objective, constraints)
    
    # Optimized CLARABEL parameters
    try:
        prob.solve(
            solver="CLARABEL",
            max_iter=100,
            tol_gap_abs=1e-6,
            tol_gap_rel=1e-6,
            tol_feas=1e-8,
            verbose=False
        )
        if prob.status == "optimal":
            return x.value.tolist()
    except:
        pass
    
    # Fallback to ECOS if CLARABEL fails
    prob.solve(solver="ECOS", max_iters=100, abstol=1e-8, reltol=1e-8)
    if prob.status == "optimal":
        return x.value.tolist()
    
    # Final fallback
    prob.solve()
    if prob.status != "optimal":
        raise RuntimeError(f"Solver failed: {prob.status}")
    return x.value.tolist()

class Solver:
    def solve(self, problem, **kwargs):
        # If we get a batch of problems, solve in parallel
        if isinstance(problem, list):
            with Pool() as pool:
                solutions = pool.map(solve_single, problem)
            return [{"solution": s} for s in solutions]
        
        # Single problem
        return {"solution": solve_single(problem)}