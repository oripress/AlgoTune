from ortools.graph.python import min_cost_flow

class Solver:
    def solve(self, problem, **kwargs):
        capacity = problem["capacity"]
        cost = problem["cost"]
        s = problem["s"]
        t = problem["t"]
        n = len(capacity)
        
        # Create min cost flow solver
        smcf = min_cost_flow.SimpleMinCostFlow()
        INF = 10**18
        
        # Add arcs
        for u in range(n):
            for v in range(n):
                cap = capacity[u][v]
                if cap > 0:
                    smcf.add_arc_with_capacity_and_unit_cost(u, v, cap, cost[u][v])
        
        # Set supply at source and sink
        smcf.set_node_supply(s, INF)
        smcf.set_node_supply(t, -INF)
        
        # Solve
        status = smcf.solve()
        
        # Extract flow solution
        flow = [[0] * n for _ in range(n)]
        if status == smcf.OPTIMAL:
            for arc in range(smcf.num_arcs()):
                u = smcf.tail(arc)
                v = smcf.head(arc)
                f = smcf.flow(arc)
                if f > 0:  # Only update non-zero flows
                    flow[u][v] = f
        
        return flow