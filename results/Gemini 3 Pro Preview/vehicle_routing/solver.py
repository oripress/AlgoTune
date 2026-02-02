from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)
        
        # Identify customers
        customers = [i for i in range(n) if i != depot]
        num_customers = len(customers)
        
        # If fewer customers than vehicles, we cannot have K routes visiting distinct customers.
        # The reference implementation enforces K departures to distinct nodes (since x_ii is not defined).
        if num_customers < K:
            return []
            
        if K == 0:
            return []

        model = cp_model.CpModel()
        
        # Nodes in the expanded graph:
        # 0 .. num_customers-1 : customers
        # num_customers .. num_customers+K-1 : depot copies
        total_nodes = num_customers + K
        
        # Mapping from expanded node index to original node index
        node_to_orig = {}
        for i in range(num_customers):
            node_to_orig[i] = customers[i]
        for i in range(K):
            node_to_orig[num_customers + i] = depot
            
        arcs = []
        arc_literals = {}
        obj_terms = []
        
        # Create variables and objective
        for i in range(total_nodes):
            for j in range(total_nodes):
                if i == j:
                    continue
                
                # Forbid depot -> depot edges to ensure each route has at least one customer
                if i >= num_customers and j >= num_customers:
                    continue
                
                lit = model.NewBoolVar(f'x_{i}_{j}')
                arc_literals[(i, j)] = lit
                arcs.append((i, j, lit))
                
                u = node_to_orig[i]
                v = node_to_orig[j]
                cost = D[u][v]
                
                if cost > 0:
                    obj_terms.append(lit * cost)
                    
        # Enforce a single Hamiltonian cycle on the expanded graph
        model.AddCircuit(arcs)
        
        # Minimize total distance
        model.Minimize(sum(obj_terms))
        
        solver = cp_model.CpSolver()
        # Use multiple workers for speedup
        solver.parameters.num_search_workers = 8
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            # Reconstruct routes from the cycle
            next_node = {}
            for (i, j), lit in arc_literals.items():
                if solver.Value(lit):
                    next_node[i] = j
            
            # Traverse the cycle starting from the first depot copy
            path = []
            curr = num_customers # First depot copy index
            
            # We expect to visit all total_nodes
            for _ in range(total_nodes):
                path.append(curr)
                curr = next_node[curr]
            
            # Parse the path into K routes
            solution = []
            current_route = [depot]
            
            # The path starts with a depot. The next node is a customer (since D->D forbidden).
            # We iterate from the second node.
            for node in path[1:]:
                if node >= num_customers:
                    # Found a depot, close current route and start new one
                    current_route.append(depot)
                    solution.append(current_route)
                    current_route = [depot]
                else:
                    # Found a customer
                    current_route.append(node_to_orig[node])
            
            # Close the last route
            current_route.append(depot)
            solution.append(current_route)
            
            return solution
        else:
            return []