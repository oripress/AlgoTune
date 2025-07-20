from collections.abc import Iterator
import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def queen_reach(self, instance: np.ndarray, start: tuple[int, int]) -> Iterator[tuple[int, int]]:
        """
        Yields all coordinates that would be in reach of the queen, including the own position.
        
        Parameters:
            instance (np.ndarray): The chessboard matrix with obstacles.
            start (tuple): The starting position (row, column) of the queen.
        
        Yields:
            tuple: Coordinates (row, column) that the queen can reach.
        """
        n, m = instance.shape
        r, c = start
        directions = [
            (-1, -1), (-1, 0), (-1, 1),  # Up-left, Up, Up-right
            (0, -1), (0, 1),             # Left, Right
            (1, -1), (1, 0), (1, 1),     # Down-left, Down, Down-right
        ]
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            while 0 <= nr < n and 0 <= nc < m:
                if instance[nr, nc]:  # Stop if there's an obstacle
                    break
                yield (nr, nc)
                nr += dr
                nc += dc
    
    def solve(self, problem: np.ndarray, **kwargs) -> list[tuple[int, int]]:
        """
        Solves the Queens with Obstacles Problem using CP-SAT.
        
        Parameters:
            problem (np.ndarray): The chessboard matrix with obstacles.
        
        Returns:
            list: A list of tuples representing the positions (row, column) of the placed queens.
        """
        instance = problem
        n, m = instance.shape
        model = cp_model.CpModel()
        
        # Decision variables
        queens = [[model.NewBoolVar(f"queen_{r}_{c}") for c in range(m)] for r in range(n)]
        
        # Constraint: No queens on obstacles
        for r in range(n):
            for c in range(m):
                if instance[r, c]:
                    model.Add(queens[r][c] == 0)
        
        # Pre-compute reach positions for all cells to avoid redundant computation
        reach_map = {}
        for r in range(n):
            for c in range(m):
                if not instance[r, c]:
                    reach_map[(r, c)] = list(self.queen_reach(instance, (r, c)))
        
        # Constraint: No two queens attack each other
        for r in range(n):
            for c in range(m):
                if not instance[r, c]:
                    reach_positions = reach_map[(r, c)]
                    # If we place a queen at (r, c), ensure no other queens are in reach
                    model.Add(
                        sum(queens[nr][nc] for nr, nc in reach_positions) == 0
                    ).only_enforce_if(queens[r][c])
        
        # Maximize the number of queens placed
        model.Maximize(sum(queens[r][c] for r in range(n) for c in range(m)))
        
        solver = cp_model.CpSolver()
        # Optimize solver parameters for speed
        solver.parameters.log_search_progress = False
        solver.parameters.num_search_workers = 8
        solver.parameters.linearization_level = 0
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [(r, c) for r in range(n) for c in range(m) if solver.Value(queens[r][c])]
        else:
            return []