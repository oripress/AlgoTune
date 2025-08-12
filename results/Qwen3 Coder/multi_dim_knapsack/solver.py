from typing import Any, List, Tuple
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        # Parse input
        if isinstance(problem, (list, tuple)):
            if len(problem) == 3:
                value, demand, supply = problem
            else:
                return []
        else:
            # Assume it's a named tuple or object with attributes
            value, demand, supply = problem.value, problem.demand, problem.supply

        n = len(value)  # number of items
        k = len(supply)  # number of resources

        # Preprocessing: Remove items that exceed any single resource capacity
        valid_items = []
        for i in range(n):
            if all(demand[i][r] <= supply[r] for r in range(k)):
                valid_items.append(i)

        # If no valid items, return empty solution
        if not valid_items:
            return []

        # For very small problems, use simple greedy
        if len(valid_items) <= 8:
            # Calculate efficiency for each item
            efficiency = []
            for i in valid_items:
                total_demand = sum(demand[i])
                if total_demand > 0:
                    eff = value[i] / total_demand
                else:
                    eff = float('inf') if value[i] > 0 else 0
                efficiency.append(eff)

            # Sort items by efficiency (descending)
            sorted_items = sorted(valid_items, key=lambda i: efficiency[valid_items.index(i)], reverse=True)

            # Greedy selection
            selected = []
            resource_usage = [0] * k

            for i in sorted_items:
                # Check if item can be added
                can_add = True
                for r in range(k):
                    if resource_usage[r] + demand[i][r] > supply[r]:
                        can_add = False
                        break

                if can_add:
                    selected.append(i)
                    for r in range(k):
                        resource_usage[r] += demand[i][r]

            return sorted(selected)
        
        # For medium problems, use improved greedy with local search
        elif len(valid_items) <= 20:
            # Sort by value density
            items_with_density = []
            for i in valid_items:
                total_demand = sum(demand[i])
                density = value[i] / max(total_demand, 1e-10)  # Avoid division by zero
                items_with_density.append((i, density))
            
            items_with_density.sort(key=lambda x: x[1], reverse=True)
            
            # Greedy selection with local improvement
            selected = []
            resource_usage = [0] * k
            
            for i, _ in items_with_density:
                # Check if item can be added
                can_add = True
                for r in range(k):
                    if resource_usage[r] + demand[i][r] > supply[r]:
                        can_add = False
                        break
                
                if can_add:
                    selected.append(i)
                    for r in range(k):
                        resource_usage[r] += demand[i][r]
            
            # Simple local search improvement
            improved = True
            while improved and len(selected) > 0:
                improved = False
                # Try to improve by swapping one item in with one item out
                for out_idx in range(len(selected)):
                    item_out = selected[out_idx]
                    temp_usage = [resource_usage[r] - demand[item_out][r] for r in range(k)]
                    
                    # Try to add better items
                    for i, _ in items_with_density:
                        if i not in selected:
                            can_add = True
                            for r in range(k):
                                if temp_usage[r] + demand[i][r] > supply[r]:
                                    can_add = False
                                    break
                            
                            if can_add:
                                # Check if this improves value
                                current_value = sum(value[j] for j in selected)
                                new_value = current_value - value[item_out] + value[i]
                                
                                if new_value > current_value:
                                    # Make the swap
                                    selected[out_idx] = i
                                    for r in range(k):
                                        resource_usage[r] = temp_usage[r] + demand[i][r]
                                    improved = True
                                    break
                    
                    if improved:
                        break
                        
            return sorted(selected)
        
        # For all other problems, use optimized CP-SAT solver
        # Create model
        model = cp_model.CpModel()

        # Create binary variables only for valid items
        x = {}
        for i in valid_items:
            x[i] = model.NewBoolVar(f'x[{i}]')

        # Add constraints for each resource
        for r in range(k):
            model.Add(sum(demand[i][r] * x[i] for i in valid_items) <= supply[r])

        # Set objective (maximize)
        model.Maximize(sum(value[i] * x[i] for i in valid_items))

        # Create solver
        solver = cp_model.CpSolver()
        
        # Set solver parameters for speed
        solver.parameters.max_time_in_seconds = 0.5
        solver.parameters.num_search_workers = 1
        solver.parameters.log_search_progress = False
        solver.parameters.cp_model_probing_level = 0
        solver.parameters.cp_model_presolve = False
        solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
        solver.parameters.linearization_level = 0
        solver.parameters.optimize_with_core = False

        # Solve
        status = solver.Solve(model)

        # Extract solution
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            solution = [i for i in valid_items if solver.Value(x[i]) == 1]
            return solution
        else:
            return []