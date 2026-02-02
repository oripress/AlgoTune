from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n == 0:
            return []
        if n == 1:
            return [0]
        
        # Precompute closed neighborhoods (including itself)
        closed_neighbors = []
        for i in range(n):
            cn = {i}
            for j in range(n):
                if problem[i][j] == 1:
                    cn.add(j)
            closed_neighbors.append(cn)
        
        # Check for universal nodes (covers entire graph)
        for i in range(n):
            if len(closed_neighbors[i]) == n:
                return [i]
        
        # Greedy solution for upper bound and hint
        dominated = set()
        greedy_set = []
        remaining = set(range(n))
        
        while len(dominated) < n:
            best_v = -1
            best_count = -1
            for v in remaining:
                count = len(closed_neighbors[v] - dominated)
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1:
                break
            greedy_set.append(best_v)
            dominated |= closed_neighbors[best_v]
        
        upper_bound = len(greedy_set)
        
        # Quick check: if greedy gives size 1, it's optimal
        if upper_bound == 1:
            return greedy_set
        
        # Build model
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        
        # Add greedy solution as hint
        greedy_set_set = set(greedy_set)
        for i in range(n):
            model.AddHint(nodes[i], 1 if i in greedy_set_set else 0)
        
        # Domination constraints
        cn_lists = [list(closed_neighbors[i]) for i in range(n)]
        for i in range(n):
            model.Add(sum(nodes[j] for j in cn_lists[i]) >= 1)
        
        # Upper bound from greedy
        total = sum(nodes)
        model.Add(total <= upper_bound)
        model.Minimize(total)
        
        # Solver with optimized parameters
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        solver.parameters.log_search_progress = False
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(n) if solver.Value(nodes[i]) == 1]
        return greedy_set