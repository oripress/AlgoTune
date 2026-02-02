from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        adj_matrix = problem["adj_matrix"]
        weights = problem["weights"]
        n = len(adj_matrix)
        
        if n == 0:
            return []
        
        if n == 1:
            return [0] if weights[0] > 0 else []
        
        # Build neighbors list
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i][j]:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        
        # Find connected components
        visited = [False] * n
        components = []
        
        for start in range(n):
            if visited[start]:
                continue
            comp = []
            stack = [start]
            visited[start] = True
            while stack:
                v = stack.pop()
                comp.append(v)
                for u in neighbors[v]:
                    if not visited[u]:
                        visited[u] = True
                        stack.append(u)
            components.append(comp)
        
        result = []
        for comp in components:
            if len(comp) == 1:
                if weights[comp[0]] > 0:
                    result.append(comp[0])
            else:
                # Check isolated nodes (no edges in component)
                has_edge = any(neighbors[v] for v in comp)
                if not has_edge:
                    result.extend(v for v in comp if weights[v] > 0)
                else:
                    result.extend(self._solve_cpsat(comp, neighbors, weights))
        
        return result
    
    def _solve_cpsat(self, component, neighbors, weights):
        m = len(component)
        idx = {v: i for i, v in enumerate(component)}
        
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(m)]
        
        edges = []
        for v in component:
            i = idx[v]
            for u in neighbors[v]:
                if u in idx:
                    j = idx[u]
                    if i < j:
                        edges.append((i, j))
                        model.Add(nodes[i] + nodes[j] <= 1)
        
        local_weights = [weights[component[i]] for i in range(m)]
        model.Maximize(sum(int(local_weights[i] * 1000000) * nodes[i] for i in range(m)))
        
        solver = cp_model.CpSolver()
        if m > 50:
            solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [component[i] for i in range(m) if solver.Value(nodes[i])]
        return []