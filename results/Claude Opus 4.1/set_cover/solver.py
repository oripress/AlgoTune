from ortools.linear_solver import pywraplp
from typing import Any, List

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solves the set cover problem using OR-Tools' Integer Linear Programming.
        Returns 1-indexed indices of selected subsets.
        """
        if not problem:
            return []
        
        # Find the universe
        universe = set()
        for subset in problem:
            universe.update(subset)
        
        if not universe:
            return []
        
        n_sets = len(problem)
        
        # Create the ILP solver
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            return []
        
        # Binary variables for each subset (1 if selected, 0 otherwise)
        x = []
        for i in range(n_sets):
            x.append(solver.IntVar(0, 1, f'x_{i}'))
        
        # Constraints: each element must be covered by at least one selected subset
        for element in universe:
            covering_sets = []
            for i, subset in enumerate(problem):
                if element in subset:
                    covering_sets.append(x[i])
            
            if covering_sets:
                solver.Add(solver.Sum(covering_sets) >= 1)
        
        # Objective: minimize the number of selected subsets
        solver.Minimize(solver.Sum(x))
        
        # Solve the problem
        status = solver.Solve()
        
        # Extract the solution
        if status == pywraplp.Solver.OPTIMAL:
            solution = []
            for i in range(n_sets):
                if x[i].solution_value() > 0.5:
                    solution.append(i + 1)  # Convert to 1-indexed
            return solution
        
        return []