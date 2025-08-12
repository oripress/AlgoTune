from typing import Any, List, Tuple
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """Solve the set cover with conflicts problem with optimized constraint formulation."""
        # Parse the problem
        if isinstance(problem, tuple):
            n, sets, conflicts = problem
        else:
            n, sets, conflicts = problem.n, problem.sets, problem.conflicts
            
        # Precompute which sets cover each object
        object_to_sets = [[] for _ in range(n)]
        for i, s in enumerate(sets):
            for obj in s:
                object_to_sets[obj].append(i)
        
        # Create the model
        model = cp_model.CpModel()
        
        # Create binary variables for each set
        set_vars = [model.NewBoolVar(f'set_{i}') for i in range(len(sets))]
        
        # Constraint: Each object must be covered
        # Add constraints in order of objects with fewest covering sets first (better propagation)
        obj_order = sorted(range(n), key=lambda x: len(object_to_sets[x]))
        for obj in obj_order:
            model.Add(sum(set_vars[i] for i in object_to_sets[obj]) >= 1)
        
        # Constraint: No conflicting sets can be selected together
        # Sort conflicts by size (larger conflicts first for better pruning)
        conflicts_sorted = sorted(conflicts, key=lambda x: len(x), reverse=True)
        for conflict in conflicts_sorted:
            if len(conflict) > 1:
                model.AddAtMostOne([set_vars[i] for i in conflict])
        
        # Objective: Minimize the number of selected sets
        model.Minimize(sum(set_vars))
        
        # Solve with optimized parameters
        solver = cp_model.CpSolver()
        
        # Aggressive parameters for speed
        solver.parameters.max_time_in_seconds = 9.5
        solver.parameters.log_search_progress = False
        solver.parameters.cp_model_probing_level = 2
        solver.parameters.linearization_level = 0
        solver.parameters.cp_model_presolve = True
        solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
        solver.parameters.cp_model_use_sat_presolve = True
        solver.parameters.optimize_with_core = True
        solver.parameters.max_deterministic_time = 4.0
        solver.parameters.num_search_workers = 1
        
        status = solver.Solve(model)
        
        # Extract solution
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solution = [i for i in range(len(sets)) if solver.Value(set_vars[i]) == 1]
            return solution
        else:
            raise ValueError("No feasible solution found.")