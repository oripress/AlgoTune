from collections.abc import Iterator
import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: np.ndarray, **kwargs) -> list[tuple[int, int]]:
        """
        Solves the Queens with Obstacles Problem using CP-SAT.
        """
        instance = problem
        n, m = instance.shape
        model = cp_model.CpModel()
        
        # Decision variables - use empty string for variable names (faster)
        queens = [[model.NewBoolVar('') for c in range(m)] for r in range(n)]
        
        # Constraint: No queens on obstacles
        for r in range(n):
            for c in range(m):
                if instance[r, c]:
                    model.Add(queens[r][c] == 0)
        
        # Constraint: No two queens attack each other
        for r in range(n):
            for c in range(m):
                if not instance[r, c]:
                    reach_positions = list(self._queen_reach(instance, (r, c)))
                    # If we place a queen at (r, c), ensure no other queens are in reach
                    if reach_positions:
                        model.Add(
                            sum(queens[nr][nc] for nr, nc in reach_positions) == 0
                        ).only_enforce_if(queens[r][c])
        
        # Maximize the number of queens placed
        model.Maximize(sum(queens[r][c] for r in range(n) for c in range(m)))
        
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        solver.parameters.max_time_in_seconds = 10.0
        solver.parameters.num_search_workers = 4  # Use parallel workers
        solver.parameters.cp_model_presolve = True
        solver.parameters.linearization_level = 2
        solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [(r, c) for r in range(n) for c in range(m) if solver.Value(queens[r][c])]
        else:
            return []
    
    def _queen_reach(self, instance: np.ndarray, start: tuple[int, int]) -> Iterator[tuple[int, int]]:
        """
        Yields all coordinates that would be in reach of the queen.
        """
        n, m = instance.shape
        r, c = start
        
        # Unrolled loop for better performance
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            while 0 <= nr < n and 0 <= nc < m:
                if instance[nr, nc]:
                    break
                yield (nr, nc)
                nr += dr
                nc += dc