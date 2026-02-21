from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[list[int]]:
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)
        
        model = cp_model.CpModel()
        
        num_nodes = n + K - 1
        
        arcs = []
        arc_vars = {}
        
        # Precompute original indices and depot flags
        orig = [depot if (i == depot or i >= n) else i for i in range(num_nodes)]
        is_depot = [True if (i == depot or i >= n) else False for i in range(num_nodes)]
        
        obj_terms = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                
                if is_depot[i] and is_depot[j]:
                    continue
                
                var = model.NewBoolVar("")
                arcs.append((i, j, var))
                arc_vars[(i, j)] = var
                
                dist = D[orig[i]][orig[j]]
                if dist != 0:
                    obj_terms.append(dist * var)
                
        model.AddCircuit(arcs)
        model.Minimize(sum(obj_terms))
        
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Build next_node mapping
            next_node = {}
            for (i, j), var in arc_vars.items():
                if solver.Value(var):
                    next_node[i] = j
            
            routes = []
            for d in range(num_nodes):
                if is_depot[d]:
                    if d in next_node:
                        route = [depot]
                        curr = next_node[d]
                        while not is_depot[curr]:
                            route.append(curr)
                            curr = next_node[curr]
                        route.append(depot)
                        routes.append(route)
            return routes
            
        return []