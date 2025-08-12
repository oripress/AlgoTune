import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[Any]]:
        """
        Solves the minimum cost maximum flow problem using OR-Tools with optimized numpy operations.
        """
        capacity = problem["capacity"]
        cost = problem["cost"]
        s = problem["s"]
        t = problem["t"]
        n = len(capacity)
        
        # Convert to numpy arrays for vectorized operations
        capacity_np = np.array(capacity, dtype=np.int64)
        cost_np = np.array(cost, dtype=np.int64)
        
        # Use numpy to find all edges with capacity > 0
        mask = capacity_np > 0
        rows, cols = np.where(mask)
        
        # Extract the values using numpy advanced indexing
        start_nodes = rows.tolist()
        end_nodes = cols.tolist()
        capacities = capacity_np[mask].tolist()
        unit_costs = cost_np[mask].tolist()
        
        # Use OR-Tools for the actual min cost flow computation
        from ortools.graph.python import min_cost_flow
        
        # Instantiate a SimpleMinCostFlow solver.
        smcf = min_cost_flow.SimpleMinCostFlow()
        
        # Add all arcs at once using the batch method
        smcf.add_arcs_with_capacity_and_unit_cost(
            start_nodes, end_nodes, capacities, unit_costs
        )
        
        # Add node supplies efficiently
        supplies = [0] * n
        supplies[s] = n * 1000  # Large supply at source
        supplies[t] = -n * 1000  # Large demand at sink
        
        smcf.set_nodes_supplies(list(range(n)), supplies)
        
        # Find the maximum flow with minimum cost.
        status = smcf.solve()
        
        # Convert solution to flow matrix efficiently
        if status == smcf.OPTIMAL:
            flow_matrix = [[0] * n for _ in range(n)]
            # Get the flow for each arc
            num_arcs = smcf.num_arcs()
            for i in range(num_arcs):
                flow_matrix[start_nodes[i]][end_nodes[i]] = smcf.flow(i)
        else:
            # Return zero matrix if no solution found
            flow_matrix = [[0] * n for _ in range(n)]
        
        return flow_matrix