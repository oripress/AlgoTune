from typing import Any
import numpy as np
import highspy

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the Capacitated Facility Location Problem using HiGHS MIP solver.
        """
        fixed_costs = np.array(problem["fixed_costs"])
        capacities = np.array(problem["capacities"])
        demands = np.array(problem["demands"])
        transportation_costs = np.array(problem["transportation_costs"])
        
        n_facilities = len(fixed_costs)
        n_customers = len(demands)
        
        # Create HiGHS instance
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("log_to_console", False)
        h.setOptionValue("threads", 8)
        h.setOptionValue("mip_rel_gap", 0.001)  # Allow 0.1% optimality gap
        
        # Total number of variables: n_facilities (y) + n_facilities*n_customers (x)
        n_y = n_facilities
        n_x = n_facilities * n_customers
        n_vars = n_y + n_x
        
        # Variable bounds: all binary
        col_lower = np.zeros(n_vars)
        col_upper = np.ones(n_vars)
        
        # Objective coefficients
        obj_coeffs = np.zeros(n_vars)
        obj_coeffs[:n_y] = fixed_costs
        obj_coeffs[n_y:] = transportation_costs.flatten()
        
        # Build constraint matrix efficiently using lists
        aindex = []
        avalue = []
        astart = [0]
        row_lower = []
        row_upper = []
        
        # Constraint 1: Each customer served exactly once
        for j in range(n_customers):
            row_lower.append(1.0)
            row_upper.append(1.0)
            for i in range(n_facilities):
                aindex.append(n_y + i * n_customers + j)
                avalue.append(1.0)
            astart.append(len(aindex))
        
        # Constraint 2: Capacity limits
        for i in range(n_facilities):
            row_lower.append(-highspy.kHighsInf)
            row_upper.append(0.0)
            aindex.append(i)
            avalue.append(-capacities[i])
            for j in range(n_customers):
                aindex.append(n_y + i * n_customers + j)
                avalue.append(demands[j])
            astart.append(len(aindex))
        
        # Constraint 3: Assignment only if facility open
        for i in range(n_facilities):
            for j in range(n_customers):
                row_lower.append(-highspy.kHighsInf)
                row_upper.append(0.0)
                aindex.append(n_y + i * n_customers + j)
                avalue.append(1.0)
                aindex.append(i)
                avalue.append(-1.0)
                astart.append(len(aindex))
        
        n_constraints = len(row_lower)
        
        # Set up the model
        h.addVars(n_vars, col_lower, col_upper)
        h.changeColsCost(n_vars, np.arange(n_vars, dtype=np.int32), obj_coeffs)
        h.changeColsIntegrality(n_vars, np.arange(n_vars, dtype=np.int32), 
                                np.ones(n_vars, dtype=np.int32))
        
        h.addRows(n_constraints, 
                  np.array(row_lower), 
                  np.array(row_upper),
                  len(aindex),
                  np.array(astart, dtype=np.int32),
                  np.array(aindex, dtype=np.int32),
                  np.array(avalue))
        
        # Solve
        h.run()
        
        solution = h.getSolution()
        status = h.getModelStatus()
        
        if status != highspy.HighsModelStatus.kOptimal:
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }
        
        # Extract solution
        col_values = solution.col_value
        facility_status = [bool(col_values[i] > 0.5) for i in range(n_facilities)]
        assignments = [[float(col_values[n_y + i * n_customers + j]) 
                       for j in range(n_customers)]
                       for i in range(n_facilities)]
        
        return {
            "objective_value": float(h.getObjectiveValue()),
            "facility_status": facility_status,
            "assignments": assignments,
        }