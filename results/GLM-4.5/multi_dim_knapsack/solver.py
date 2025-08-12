from typing import Any, List, Tuple, NamedTuple
from ortools.sat.python import cp_model
import numpy as np

class MultiDimKnapsackInstance(NamedTuple):
    value: List[float]
    demand: List[List[float]]
    supply: List[float]

MultiKnapsackSolution = List[int]

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """Your implementation goes here."""
        if not isinstance(problem, MultiDimKnapsackInstance):
            try:
                problem = MultiDimKnapsackInstance(*problem)
            except Exception as e:
                return []

        n: int = len(problem.value)
        k: int = len(problem.supply)

        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Add constraints more efficiently
        for r in range(k):
            # Pre-compute the constraint expression
            constraint_expr = []
            for i in range(n):
                if problem.demand[i][r] > 0:  # Only add non-zero terms
                    constraint_expr.append(x[i] * problem.demand[i][r])
            if constraint_expr:
                model.Add(sum(constraint_expr) <= problem.supply[r])
        
        # Objective function
        objective_terms = []
        for i in range(n):
            if problem.value[i] > 0:  # Only add positive value terms
                objective_terms.append(x[i] * problem.value[i])
        model.Maximize(sum(objective_terms) if objective_terms else 0)

        solver = cp_model.CpSolver()
        
        # Aggressive optimization parameters
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = min(5.0, n * k * 0.002)  # Maximum aggressive time limit
        solver.parameters.log_search_progress = False
        solver.parameters.cp_model_presolve = True
        # Size-based optimization
        if n <= 20:
            # Small problems: use default settings but with presolve
            solver.parameters.linearization_level = 0
        elif n <= 50:
            # Medium problems: balanced approach
            solver.parameters.linearization_level = 1
            solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
        else:
            # Large problems: aggressive optimization
            solver.parameters.linearization_level = 2
            solver.parameters.search_branching = cp_model.FIXED_SEARCH
            solver.parameters.use_dynamic_symmetry_breaking = True
            solver.parameters.use_no_overlap_lns = True
            solver.parameters.use_interval_lns = True
            
            # Removed symmetry breaking to reduce overhead
        
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Extract solution more efficiently
            solution = []
            for i in range(n):
                if solver.Value(x[i]):
                    solution.append(i)
            return solution
        return []