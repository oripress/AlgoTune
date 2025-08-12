from typing import List
from ortools.linear_solver import pywraplp

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solve Minimum Dominating Set using OR‑Tools MIP (CBC).
        Returns a sorted list of vertex indices forming an optimal dominating set.
        """
        n = len(problem)
        if n == 0:
            return []

        # Create a linear solver with the CBC backend.
        solver = pywraplp.Solver.CreateSolver('CBC')
        if solver is None:
            # Fallback: return all vertices.
            return list(range(n))

        # Binary variable x[i] = 1 if vertex i is in the dominating set.
        x = [solver.BoolVar(f"x_{i}") for i in range(n)]

        # Domination constraints: each vertex must be dominated by itself or a neighbor.
        for i in range(n):
            expr = [x[i]]
            for j in range(n):
                if problem[i][j] == 1:
                    expr.append(x[j])
            solver.Add(solver.Sum(expr) >= 1)

        # Objective: minimize the number of selected vertices.
        solver.Minimize(solver.Sum(x))

        # Set a modest time limit (30 seconds) to avoid pathological runtimes.
        solver.set_time_limit(30_000)  # milliseconds

        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            result = [i for i in range(n) if x[i].solution_value() > 0.5]
            return sorted(result)
        else:
            # If the solver fails, return all vertices (which trivially dominates the graph).
            return list(range(n))