from typing import Any
import numpy as np
from ortools.graph.python import min_cost_flow
from collections import deque
from itertools import repeat

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[int]:
        T = problem["T"]
        N = problem["N"]
        
        if T == 0:
            return []
            
        prices = np.array(problem["prices"], dtype=np.float64)
        capacities = np.array(problem["capacities"], dtype=np.int64)
        probs = np.array(problem["probs"], dtype=np.float64)
        
        SCALE = 1_000_000.0
        
        # Calculate potential revenues
        revenues = probs * prices
        
        # Filter arcs: revenue must be significant and product must have capacity
        cap_mask = (capacities > 0)
        mask = (revenues * SCALE >= 0.5) & cap_mask[None, :]
        
        rows, cols = np.nonzero(mask)
        selected_costs = -(revenues[rows, cols] * SCALE).astype(np.int64)
        
        smcf = min_cost_flow.SimpleMinCostFlow()
        add_arc = smcf.add_arc_with_capacity_and_unit_cost
        
        # Node indices
        source = 0
        dummy = T + N + 1
        sink = T + N + 2
        
        # Helper to consume iterators efficiently
        consume = deque(maxlen=0).extend
        
        # Add Source -> Periods and Period -> Dummy
        range_t = range(1, T + 1)
        
        # Source -> Periods
        consume(map(add_arc, repeat(source), range_t, repeat(1), repeat(0)))
        # Periods -> Dummy
        consume(map(add_arc, range_t, repeat(dummy), repeat(1), repeat(0)))
            
        # Add Periods -> Products
        u_nodes = rows + 1
        v_nodes = cols + (T + 1)
        
        u_list = u_nodes.tolist()
        v_list = v_nodes.tolist()
        c_list = selected_costs.tolist()
        
        consume(map(add_arc, u_list, v_list, repeat(1), c_list))
            
        # Add Products -> Sink
        valid_indices = np.nonzero(capacities > 0)[0]
        product_nodes = (valid_indices + (T + 1)).tolist()
        product_caps = capacities[valid_indices].tolist()
        
        consume(map(add_arc, product_nodes, repeat(sink), product_caps, repeat(0)))
            
        # Add Dummy -> Sink
        add_arc(dummy, sink, T, 0)
        
        # Set Supplies
        smcf.set_node_supply(source, T)
        smcf.set_node_supply(sink, -T)
        
        # Solve
        status = smcf.solve()
        
        if status != smcf.OPTIMAL:
            return [-1] * T
            
        solution = [-1] * T
        
        # Extract solution
        base_arc = 2 * T
        get_flow = smcf.flow
        T_plus_1 = T + 1
        
        for k, u in enumerate(u_list):
            if get_flow(base_arc + k) == 1:
                solution[u - 1] = v_list[k] - T_plus_1
                        
        return solution