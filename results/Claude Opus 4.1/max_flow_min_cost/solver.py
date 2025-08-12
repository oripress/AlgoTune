from typing import Any
from ortools.graph.python import min_cost_flow

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[list[Any]]:
        """Optimized Maximum Flow Min Cost using OR-Tools."""
        capacity = problem["capacity"]
        cost = problem["cost"]
        s = problem["s"]
        t = problem["t"]
        n = len(capacity)
        
        # Create solver
        mcf = min_cost_flow.SimpleMinCostFlow()
        
        # Add edges efficiently - avoid numpy conversion overhead
        max_flow = 0
        for i in range(n):
            row_cap = capacity[i]
            row_cost = cost[i]
            for j in range(n):
                cap = row_cap[j]
                if cap > 0:
                    mcf.add_arc_with_capacity_and_unit_cost(i, j, cap, row_cost[j])
                    if i == s:
                        max_flow += cap
        
        # Set supplies
        mcf.set_node_supply(s, max_flow)
        mcf.set_node_supply(t, -max_flow)
        
        # Solve
        if mcf.solve() != mcf.OPTIMAL:
            return [[0] * n for _ in range(n)]
        
        # Build solution efficiently
        solution = [[0] * n for _ in range(n)]
        num_arcs = mcf.num_arcs()
        for arc in range(num_arcs):
            flow = mcf.flow(arc)
            if flow > 0:
                solution[mcf.tail(arc)][mcf.head(arc)] = flow
        
        return solution