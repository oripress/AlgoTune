import numpy as np
from typing import List, Tuple
from ortools.sat.python import cp_model
import numba

@numba.jit(nopython=True)
def compute_attacks(problem, valid_positions):
    """Pre-compute attack relationships using numba for speed."""
    n, m = problem.shape
    num_valid = len(valid_positions)
    attacks_matrix = np.zeros((num_valid, num_valid), dtype=np.bool_)
    
    directions = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
    
    for i in range(num_valid):
        r1, c1 = valid_positions[i]
        
        for d in range(8):
            dr, dc = directions[d]
            nr, nc = r1 + dr, c1 + dc
            
            while 0 <= nr < n and 0 <= nc < m:
                if problem[nr, nc]:  # Stop at obstacle
                    break
                    
                # Check if (nr, nc) is in valid_positions
                for j in range(num_valid):
                    r2, c2 = valid_positions[j]
                    if r2 == nr and c2 == nc:
                        attacks_matrix[i, j] = True
                        break
                        
                nr += dr
                nc += dc
    
    return attacks_matrix

class Solver:
    def solve(self, problem: np.ndarray) -> List[Tuple[int, int]]:
        """
        Solves the Queens with Obstacles Problem using optimized CP-SAT with numba.
        
        Parameters:
            problem (np.ndarray): The chessboard matrix with obstacles.
        
        Returns:
            list: A list of tuples representing the positions (row, column) of the placed queens.
        """
        n, m = problem.shape
        
        # Find all valid positions
        valid_positions = []
        for r in range(n):
            for c in range(m):
                if not problem[r, c]:
                    valid_positions.append((r, c))
        
        if not valid_positions:
            return []
        
        # Convert to numpy array for numba
        valid_pos_array = np.array(valid_positions, dtype=np.int32)
        
        # Fast attack computation with numba
        attacks_matrix = compute_attacks(problem, valid_pos_array)
        
        # Create CP-SAT model
        model = cp_model.CpModel()
        
        # Decision variables
        queens = [model.NewBoolVar(f"q_{i}") for i in range(len(valid_positions))]
        
        # Add constraints: no two queens attack each other
        for i in range(len(valid_positions)):
            for j in range(i + 1, len(valid_positions)):
                if attacks_matrix[i, j] or attacks_matrix[j, i]:
                    model.AddBoolOr([queens[i].Not(), queens[j].Not()])
        
        # Maximize the number of queens placed
        model.Maximize(sum(queens))
        
        # Solve with optimized parameters
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        solver.parameters.num_search_workers = 1
        solver.parameters.linearization_level = 2
        solver.parameters.max_time_in_seconds = 1.0  # Quick timeout
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [valid_positions[i] for i in range(len(valid_positions)) if solver.Value(queens[i])]
        else:
            return []