from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Solves the Capacitated Facility Location Problem using the CP-SAT solver.
        This version incorporates a state-of-the-art hybrid heuristic and a refined model:
        1. Refined Model: Uses a direct linear capacity constraint for better propagation.
        2. Hybrid Heuristic: A two-stage hint generator (Greedy + Enhanced Local Search).
        3. Preprocessing: Eliminates impossible assignments upfront.
        4. Solver Tuning: Uses aggressive linearization and parallel search.
        """
        fixed_costs = problem["fixed_costs"]
        capacities = problem["capacities"]
        demands = problem["demands"]
        transportation_costs = problem["transportation_costs"]
        
        n_facilities = len(fixed_costs)
        n_customers = len(demands)

        precision_factor = 1_000_000
        scaled_fixed_costs = [int(c * precision_factor) for c in fixed_costs]
        scaled_capacities = [int(c * precision_factor) for c in capacities]
        scaled_demands = [int(d * precision_factor) for d in demands]
        scaled_transportation_costs = [[int(c * precision_factor) for c in row] for row in transportation_costs]

        model = cp_model.CpModel()

        y = [model.NewBoolVar(f'y_{i}') for i in range(n_facilities)]
        x = [[model.NewBoolVar(f'x_{i}_{j}') for j in range(n_customers)] for i in range(n_facilities)]

        impossible_assignments = set()
        for i in range(n_facilities):
            for j in range(n_customers):
                if scaled_capacities[i] < scaled_demands[j]:
                    model.Add(x[i][j] == 0)
                    impossible_assignments.add((i, j))

        for j in range(n_customers):
            model.AddExactlyOne([x[i][j] for i in range(n_facilities)])

        for i in range(n_facilities):
            facility_load = cp_model.LinearExpr.WeightedSum(x[i], scaled_demands)
            model.Add(facility_load <= scaled_capacities[i] * y[i])

        total_fixed_cost = cp_model.LinearExpr.WeightedSum(y, scaled_fixed_costs)
        flat_x = [x[i][j] for i in range(n_facilities) for j in range(n_customers)]
        flat_transport_costs = [scaled_transportation_costs[i][j] for i in range(n_facilities) for j in range(n_customers)]
        total_transport_cost = cp_model.LinearExpr.WeightedSum(flat_x, flat_transport_costs)
        model.Minimize(total_fixed_cost + total_transport_cost)

        # --- Hybrid Heuristic: Greedy Construction + Enhanced Local Search ---
        # 1. Greedy Construction
        customer_indices = sorted(range(n_customers), key=lambda j: demands[j], reverse=True)
        hint_y = [0] * n_facilities
        hint_x = [[0] * n_customers for _ in range(n_facilities)]
        remaining_capacity = list(scaled_capacities)
        current_assignments = {}
        for j in customer_indices:
            best_facility_idx, min_marginal_cost = -1, float('inf')
            for i in range(n_facilities):
                if (i, j) not in impossible_assignments and remaining_capacity[i] >= scaled_demands[j]:
                    fixed_cost_comp = scaled_fixed_costs[i] if hint_y[i] == 0 else 0
                    marginal_cost = scaled_transportation_costs[i][j] + fixed_cost_comp
                    if marginal_cost < min_marginal_cost:
                        min_marginal_cost, best_facility_idx = marginal_cost, i
            if best_facility_idx != -1:
                i = best_facility_idx
                hint_x[i][j], hint_y[i] = 1, 1
                remaining_capacity[i] -= scaled_demands[j]
                current_assignments[j] = i

        # 2. Enhanced Local Search Refinement
        hint_customers_per_facility = [sum(hint_x[i]) for i in range(n_facilities)]
        while True:
            made_improvement = False
            customer_costs = sorted(current_assignments.keys(), key=lambda j: scaled_transportation_costs[current_assignments[j]][j], reverse=True)
            
            for j in customer_costs:
                i_current = current_assignments[j]
                cost_reduction = scaled_transportation_costs[i_current][j]
                if hint_customers_per_facility[i_current] == 1:
                    cost_reduction += scaled_fixed_costs[i_current]

                best_new_i, min_add_cost = -1, float('inf')
                for k in range(n_facilities):
                    if k == i_current or (k, j) in impossible_assignments: continue
                    if remaining_capacity[k] >= scaled_demands[j]:
                        add_cost = scaled_transportation_costs[k][j]
                        if hint_customers_per_facility[k] == 0:
                            add_cost += scaled_fixed_costs[k]
                        if add_cost < min_add_cost:
                            min_add_cost, best_new_i = add_cost, k
                
                if best_new_i != -1 and min_add_cost < cost_reduction:
                    k = best_new_i
                    remaining_capacity[i_current] += scaled_demands[j]
                    remaining_capacity[k] -= scaled_demands[j]
                    hint_x[i_current][j], hint_x[k][j] = 0, 1
                    current_assignments[j] = k
                    hint_customers_per_facility[i_current] -= 1
                    hint_customers_per_facility[k] += 1
                    if hint_customers_per_facility[i_current] == 0: hint_y[i_current] = 0
                    hint_y[k] = 1
                    made_improvement = True
            
            if not made_improvement: break

        for i in range(n_facilities):
            model.AddHint(y[i], hint_y[i])
            for j in range(n_customers):
                model.AddHint(x[i][j], hint_x[i][j])

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        solver.parameters.linearization_level = 2
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return {
                "objective_value": solver.ObjectiveValue() / precision_factor,
                "facility_status": [solver.Value(var) == 1 for var in y],
                "assignments": [[float(solver.Value(var)) for var in row] for row in x],
            }
        else:
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }