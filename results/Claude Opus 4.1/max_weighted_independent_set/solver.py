import highspy
import numpy as np

class Solver:
    def __init__(self):
        # Pre-initialize solver to avoid overhead
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("presolve", "on")
        self.h.setOptionValue("parallel", "on")
        self.h.setOptionValue("threads", 4)
        self.h.setOptionValue("mip_heuristic_effort", 0.1)
        self.h.setOptionValue("mip_rel_gap", 1e-6)
        self.h.setOptionValue("time_limit", 5.0)
    
    def solve(self, problem: dict[str, list]) -> list[int]:
        """
        Solves the MWIS problem using HiGHS with optimized settings.
        
        :param problem: dict with 'adj_matrix' and 'weights'
        :return: list of selected node indices.
        """
        adj_matrix = problem["adj_matrix"]
        weights = problem["weights"]
        n = len(weights)
        
        # Clear previous model
        self.h.clear()
        
        # Add all variables at once (more efficient)
        lower = [0.0] * n
        upper = [1.0] * n
        self.h.addVars(n, lower, upper)
        
        # Set variable types to binary
        integrality = [highspy.HighsVarType.kInteger] * n
        self.h.changeColsIntegrality(n, range(n), integrality)
        
        # Set objective (negative for maximization)
        obj_coeffs = [-float(w) for w in weights]
        self.h.changeColsCost(n, range(n), obj_coeffs)
        self.h.changeObjectiveSense(highspy.ObjSense.kMinimize)
        
        # Batch add constraints
        # Build constraint data
        constraint_indices = []
        constraint_values = []
        constraint_starts = [0]
        constraint_lower = []
        constraint_upper = []
        
        current_nnz = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i][j]:
                    # Add constraint x[i] + x[j] <= 1
                    constraint_indices.extend([i, j])
                    constraint_values.extend([1.0, 1.0])
                    current_nnz += 2
                    constraint_starts.append(current_nnz)
                    constraint_lower.append(0.0)
                    constraint_upper.append(1.0)
        
        # Add all constraints at once if there are any
        if len(constraint_lower) > 0:
            num_constraints = len(constraint_lower)
            self.h.addRows(
                num_constraints,
                constraint_lower,
                constraint_upper,
                current_nnz,
                constraint_starts,
                constraint_indices,
                constraint_values
            )
        
        # Solve
        self.h.run()
        
        # Extract solution
        solution = self.h.getSolution()
        if solution.value_valid:
            return [i for i in range(n) if solution.col_value[i] > 0.5]
        else:
            return []