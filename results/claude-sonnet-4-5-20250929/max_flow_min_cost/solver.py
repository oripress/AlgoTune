from ortools.graph.python import min_cost_flow, max_flow

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solves the maximum flow minimum cost problem using OR-Tools.
        
        :param problem: A dictionary with keys "capacity", "cost", "s", "t"
        :return: A 2D list containing the flow for each edge
        """
        capacity = problem["capacity"]
        cost = problem["cost"]
        s = problem["s"]
        t = problem["t"]
        n = len(capacity)
        
        
        # Build edge list using list comprehension (faster than loop)
        edges = [(i, j, capacity[i][j], cost[i][j]) 
                 for i in range(n) for j in range(n) if capacity[i][j] > 0]

        # Calculate maximum possible flow from source
        max_possible_flow = sum(capacity[s][j] for j in range(n))

        # Use min cost flow directly - it will find max feasible flow automatically
        mcf = min_cost_flow.SimpleMinCostFlow()
        for i, j, cap, c in edges:
            mcf.add_arc_with_capacity_and_unit_cost(i, j, cap, c)
        
        # Set supply to maximum possible - solver will find actual max flow
        mcf.set_node_supply(s, max_possible_flow)
        mcf.set_node_supply(t, -max_possible_flow)
        
        status = mcf.solve()
        
        # Extract solution
        solution = [[0] * n for _ in range(n)]
        if status == mcf.OPTIMAL:
            for arc in range(mcf.num_arcs()):
                solution[mcf.tail(arc)][mcf.head(arc)] = mcf.flow(arc)
        
        return solution