import random
from typing import NamedTuple
from ortools.sat.python import cp_model

class Instance(NamedTuple):
    n: int
    sets: list[list[int]]
    conflicts: list[list[int]]

class Solver:
    def solve(self, problem: Instance | tuple) -> list[int]:
        """
        Solve the set cover with conflicts problem.

        Args:
            problem: A tuple (n, sets, conflicts) where:
                - n is the number of objects
                - sets is a list of sets (each set is a list of integers)
                - conflicts is a list of conflicts (each conflict is a list of set indices)

        Returns:
            A list of set indices that form a valid cover, or None if no solution exists

        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        if not isinstance(problem, Instance):
            problem = Instance(*problem)
        n, sets, conflicts = problem
        
        num_sets = len(sets)
        
        # Pre-compute object-to-sets mapping for efficiency
        obj_to_sets = [[] for _ in range(n)]
        for set_idx, set_obj in enumerate(sets):
            for obj in set_obj:
                obj_to_sets[obj].append(set_idx)
        
        # Create CP model
        model = cp_model.CpModel()
        
        # Create binary variables for each set
        set_vars = [model.NewBoolVar(f"set_{i}") for i in range(num_sets)]
        
        # Add coverage constraints with optimization
        for obj in range(n):
            covering_sets = obj_to_sets[obj]
            if len(covering_sets) == 1:
                # If only one set covers this object, it must be selected
                model.Add(set_vars[covering_sets[0]] == 1)
            else:
                # Use sum constraint for multiple covering sets
                model.Add(sum(set_vars[i] for i in covering_sets) >= 1)
        
        # Add conflict constraints
        for conflict in conflicts:
            model.AddAtMostOne(set_vars[i] for i in conflict)
        
        # Objective: minimize number of selected sets
        model.Minimize(sum(set_vars))
        
        # Solve with optimized parameters - back to 2.15x config with small tweak
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 1.3
        solver.parameters.num_search_workers = 4
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        solver.parameters.linearization_level = 0
        solver.parameters.use_phase_saving = True
        solver.parameters.cp_model_presolve = True
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = [i for i in range(num_sets) if solver.Value(set_vars[i]) == 1]
            return solution
        else:
            raise ValueError("No feasible solution found.")