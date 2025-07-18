from typing import Any
from ortools.graph.python import min_cost_flow

class Solver:
    def solve(self, problem, **kwargs) -> list[int]:
        T = problem["T"]
        N = problem["N"]
        prices = problem["prices"]
        capacities = problem["capacities"]
        probs = problem["probs"]
        
        if T == 0:
            return []
        
        # Create min cost flow solver
        mcf = min_cost_flow.SimpleMinCostFlow()
        
        # Define node indices
        source = 0
        sink = 1
        product_nodes = [2 + i for i in range(N)]   # Products: 2 to 2+N-1
        period_nodes = [2 + N + t for t in range(T)]  # Periods: 2+N to 2+N+T-1
        
        # Store arc indices for product->period edges
        product_period_arcs = {}
        
        # Scaling factor to convert floats to integers
        SCALE = 10**9
        
        # Add arcs from source to products
        for i in range(N):
            mcf.add_arc_with_capacity_and_unit_cost(
                source, product_nodes[i], capacities[i], 0)
        
        # Add arcs from products to periods
        for i in range(N):
            for t in range(T):
                rev = prices[i] * probs[t][i]
                # Scale revenue to integer and use negative for min-cost flow
                scaled_cost = int(-rev * SCALE)
                arc_idx = mcf.add_arc_with_capacity_and_unit_cost(
                    product_nodes[i], period_nodes[t], 1, scaled_cost)
                product_period_arcs[(i, t)] = arc_idx
        
        # Add idle arcs (source -> periods)
        for t in range(T):
            mcf.add_arc_with_capacity_and_unit_cost(
                source, period_nodes[t], 1, 0)
        
        # Add arcs from periods to sink
        for t in range(T):
            mcf.add_arc_with_capacity_and_unit_cost(
                period_nodes[t], sink, 1, 0)
        
        # Set supplies
        mcf.set_node_supply(source, T)
        mcf.set_node_supply(sink, -T)
        
        # Solve
        status = mcf.solve()
        
        if status != mcf.OPTIMAL:
            return [-1] * T
        
        # Build solution
        solution = [-1] * T
        for (i, t), arc_idx in product_period_arcs.items():
            if mcf.flow(arc_idx) == 1:
                solution[t] = i
                
        return solution