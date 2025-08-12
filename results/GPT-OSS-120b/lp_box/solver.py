from typing import Any, Dict, List
import numpy as np

try:
    from ortools.linear_solver import pywraplp
except Exception:  # pragma: no cover
    pywraplp = None

try:
    from scipy.optimize import linprog
except Exception:  # pragma: no cover
    linprog = None

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[float]]:
        """
        Solve the LP box problem:
            minimize    c^T x
            subject to  A x <= b
                        0 <= x <= 1
        """
        c = np.asarray(problem["c"], dtype=float)
        A = np.asarray(problem["A"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)

        n = c.shape[0]

        # Try OR-Tools GLOP
        if pywraplp is not None:
            solver = pywraplp.Solver.CreateSolver("GLOP")
            if solver is not None:
                x = [solver.NumVar(0.0, 1.0, f"x{i}") for i in range(n)]
                for row, bound in zip(A, b):
                    ct = solver.RowConstraint(-solver.infinity(), bound)
                    for var, coeff in zip(x, row):
                        ct.SetCoefficient(var, coeff)
                objective = solver.Objective()
                for var, coeff in zip(x, c):
                    objective.SetCoefficient(var, coeff)
                objective.SetMinimization()
                status = solver.Solve()
                if status == pywraplp.Solver.OPTIMAL:
                    return {"solution": [var.solution_value() for var in x]}

        # Fallback to SciPy linprog
        if linprog is not None:
            bounds = [(0.0, 1.0) for _ in range(n)]
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
            if res.success:
                return {"solution": res.x.tolist()}
            raise RuntimeError("LP solver failed to find an optimal solution.")
        raise ImportError("No suitable LP solver available.")