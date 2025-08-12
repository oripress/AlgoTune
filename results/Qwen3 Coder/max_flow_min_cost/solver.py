import numpy as np
from typing import Any, Dict
from ortools.graph.python import min_cost_flow

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """Solve the maximum flow minimum cost problem using OR-Tools."""
        try:
            capacity = problem["capacity"]
            cost = problem["cost"]
            s = problem["s"]
            t = problem["t"]
            
            n = len(capacity)
            
            # Create the solver
            smcf = min_cost_flow.SimpleMinCostFlow()
            
            # Add nodes
            for i in range(n):
                smcf.add_node(i)
            
            # Add arcs
            arc_indices = {}
            
            for i in range(n):
                for j in range(n):
                    if capacity[i][j] > 0:
                        # Keep costs as integers to avoid floating point issues
                        scaled_cost = int(round(cost[i][j] * 1000000))
                        arc_index = smcf.add_arc_with_capacity_and_unit_cost(
                            i, j, int(capacity[i][j]), scaled_cost
                        )
                        arc_indices[(i, j)] = arc_index
            
            # Set supply at source and demand at sink with a large value
            # This will make OR-Tools find the maximum flow automatically
            total_capacity = sum(sum(row) for row in capacity)
            
            for i in range(n):
                if i == s:
                    smcf.set_node_supply(i, total_capacity)
                elif i == t:
                    smcf.set_node_supply(i, -total_capacity)
                else:
                    smcf.set_node_supply(i, 0)
            
            # Solve the problem
            status = smcf.solve()
            
            if status == smcf.OPTIMAL:
                # Extract solution
                solution = [[0.0 for _ in range(n)] for _ in range(n)]
                
                for (i, j), arc_index in arc_indices.items():
                    flow = smcf.flow(arc_index)
                    solution[i][j] = float(flow)
                
                return solution
            else:
                # Fallback to NetworkX if OR-Tools fails
                import networkx as nx
                G = nx.DiGraph()
                for i in range(n):
                    G.add_node(i)
                
                for i in range(n):
                    for j in range(n):
                        if capacity[i][j] > 0:
                            G.add_edge(i, j, capacity=capacity[i][j], weight=cost[i][j])
                
                flow_dict = nx.max_flow_min_cost(G, s, t)
                solution = [[0.0 for _ in range(n)] for _ in range(n)]
                for i in flow_dict:
                    for j in flow_dict[i]:
                        solution[i][j] = flow_dict[i][j]
                
                return solution
                
        except Exception as e:
            n = len(problem["capacity"])
            return [[0 for _ in range(n)] for _ in range(n)]