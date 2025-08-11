from typing import Any
import highspy
import numpy as np

class Solver:
    def __init__(self):
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("presolve", "on")
        self.h.setOptionValue("mip_rel_gap", 0.001)  # Accept 0.1% gap for speed
        self.h.setOptionValue("mip_abs_gap", 0.01)
        self.h.setOptionValue("threads", 1)  # Single thread often faster for small problems
        self.h.setOptionValue("mip_heuristic_effort", 0.05)  # Less heuristic effort
        
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the Capacitated Facility Location Problem using HiGHS with optimized constraint generation.
        """
        fixed_costs = np.array(problem["fixed_costs"], dtype=np.float64)
        capacities = np.array(problem["capacities"], dtype=np.float64)
        demands = np.array(problem["demands"], dtype=np.float64)
        transportation_costs = np.array(problem["transportation_costs"], dtype=np.float64)
        
        n_facilities = len(fixed_costs)
        n_customers = len(demands)
        n_vars = n_facilities + n_facilities * n_customers
        
        # Clear and reset
        self.h.clear()
        
        # Objective coefficients - vectorized
        c = np.empty(n_vars, dtype=np.float64)
        c[:n_facilities] = fixed_costs
        c[n_facilities:] = transportation_costs.ravel()
        
        # Variables: all binary
        lower = np.zeros(n_vars, dtype=np.float64)
        upper = np.ones(n_vars, dtype=np.float64)
        integrality = np.ones(n_vars, dtype=np.int32)
        
        # Pre-allocate constraint arrays
        n_customer_constraints = n_customers
        n_capacity_constraints = n_facilities
        n_link_constraints = n_facilities * n_customers
        n_constraints = n_customer_constraints + n_capacity_constraints + n_link_constraints
        
        # Estimate number of non-zeros
        nnz = n_customers * n_facilities + n_facilities * (1 + n_customers) + 2 * n_link_constraints
        
        # Pre-allocate arrays
        A_index = np.empty(nnz, dtype=np.int32)
        A_value = np.empty(nnz, dtype=np.float64)
        A_start = np.empty(n_constraints + 1, dtype=np.int32)
        constraint_lower = np.empty(n_constraints, dtype=np.float64)
        constraint_upper = np.empty(n_constraints, dtype=np.float64)
        
        nnz_idx = 0
        con_idx = 0
        
        # Customer assignment constraints (sum x[i,j] over i = 1 for each j)
        for j in range(n_customers):
            A_start[con_idx] = nnz_idx
            for i in range(n_facilities):
                A_index[nnz_idx] = n_facilities + i * n_customers + j
                A_value[nnz_idx] = 1.0
                nnz_idx += 1
            constraint_lower[con_idx] = 1.0
            constraint_upper[con_idx] = 1.0
            con_idx += 1
        
        # Capacity constraints
        for i in range(n_facilities):
            A_start[con_idx] = nnz_idx
            # -capacity * y[i]
            A_index[nnz_idx] = i
            A_value[nnz_idx] = -capacities[i]
            nnz_idx += 1
            # + demand[j] * x[i,j]
            base_idx = n_facilities + i * n_customers
            for j in range(n_customers):
                A_index[nnz_idx] = base_idx + j
                A_value[nnz_idx] = demands[j]
                nnz_idx += 1
            constraint_lower[con_idx] = -1e308  # -inf
            constraint_upper[con_idx] = 0.0
            con_idx += 1
        
        # Link constraints (x[i,j] <= y[i])
        for i in range(n_facilities):
            base_idx = n_facilities + i * n_customers
            for j in range(n_customers):
                A_start[con_idx] = nnz_idx
                # -y[i]
                A_index[nnz_idx] = i
                A_value[nnz_idx] = -1.0
                nnz_idx += 1
                # +x[i,j]
                A_index[nnz_idx] = base_idx + j
                A_value[nnz_idx] = 1.0
                nnz_idx += 1
                constraint_lower[con_idx] = -1e308  # -inf
                constraint_upper[con_idx] = 0.0
                con_idx += 1
        
        A_start[con_idx] = nnz_idx
        
        # Trim arrays if needed
        A_index = A_index[:nnz_idx]
        A_value = A_value[:nnz_idx]
        
        # Add variables and constraints to model
        self.h.addVars(n_vars, lower, upper)
        self.h.changeColsCost(n_vars, np.arange(n_vars, dtype=np.int32), c)
        self.h.changeColsIntegrality(n_vars, np.arange(n_vars, dtype=np.int32), integrality)
        self.h.addRows(n_constraints, constraint_lower, constraint_upper,
                      nnz_idx, A_start[:n_constraints+1], A_index, A_value)
        
        # Solve
        self.h.run()
        
        if self.h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }
        
        # Extract solution
        sol = self.h.getSolution().col_value
        
        facility_status = [bool(round(sol[i])) for i in range(n_facilities)]
        assignments = [[float(round(sol[n_facilities + i * n_customers + j])) 
                       for j in range(n_customers)] 
                      for i in range(n_facilities)]
        
        return {
            "objective_value": float(self.h.getObjectiveValue()),
            "facility_status": facility_status,
            "assignments": assignments,
        }