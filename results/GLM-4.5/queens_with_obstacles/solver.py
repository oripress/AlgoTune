import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs) -> list:
        """
        Optimized CP-SAT solver for Queens with Obstacles Problem.
        Uses the proven reach-based approach with performance optimizations.
        """
        instance = problem
        n, m = instance.shape
        model = cp_model.CpModel()

        # Decision variables with minimal naming overhead
        q = [[model.NewBoolVar(f"q_{r}_{c}") for c in range(m)] for r in range(n)]

        # Constraint: No queens on obstacles
        for r in range(n):
            for c in range(m):
                if instance[r, c]:
                    model.Add(q[r][c] == 0)

        # Constraint: No two queens attack each other
        for r in range(n):
            for c in range(m):
                if not instance[r, c]:
                    reach = self._queen_reach(instance, (r, c))
                    if reach:
                        model.Add(sum(q[nr][nc] for nr, nc in reach) == 0).only_enforce_if(q[r][c])
        
        # Maximize the number of queens placed
        model.Maximize(sum(q[r][c] for r in range(n) for c in range(m)))

        # Optimize solver parameters
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        solver.parameters.max_time_in_seconds = 10.0
        solver.parameters.num_search_workers = 1
        solver.parameters.cp_model_presolve = True
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [(r, c) for r in range(n) for c in range(m) if solver.Value(q[r][c])]
        else:
            return []
    def _queen_reach(self, instance: np.ndarray, start: tuple[int, int]):
        """Returns all coordinates that would be in reach of the queen (optimized version)."""
        n, m = instance.shape
        r, c = start
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        result = []
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            while 0 <= nr < n and 0 <= nc < m:
                if instance[nr, nc]:
                    break
                result.append((nr, nc))
                nr += dr
                nc += dc
        
        return result