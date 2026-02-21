from typing import Any

class Solver:
    def __init__(self):
        import ortools.graph.python.min_cost_flow as mcf
        import ortools.graph.python.max_flow as mf
        import numpy as np
        self.mcf = mcf
        self.mf = mf
        self.np = np

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        np = self.np
        
        capacity = np.asarray(problem["capacity"], dtype=np.int32)
        cost = np.asarray(problem["cost"], dtype=np.int32)
        s = problem["s"]
        t = problem["t"]
        n = len(capacity)
        
        mask = capacity > 0
        tails, heads = np.nonzero(mask)
        capacities = capacity[mask]
        costs = cost[mask]
        
        # Find max flow value
        smf = self.mf.SimpleMaxFlow()
        smf.add_arcs_with_capacity(tails, heads, capacities)
        status_mf = smf.solve(s, t)
        flow_value = smf.optimal_flow()
        smcf = self.mcf.SimpleMinCostFlow()
        
        
        smcf.add_arcs_with_capacity_and_unit_cost(tails, heads, capacities, costs)
        
        smcf.set_node_supply(s, flow_value)
        smcf.set_node_supply(t, -flow_value)
        
        status = smcf.solve()
        
        solution = np.zeros((n, n), dtype=int)
        if status == smcf.OPTIMAL:
            flows = smcf.flows(np.arange(len(tails)))
            solution[tails, heads] = flows
                
        return solution.tolist()