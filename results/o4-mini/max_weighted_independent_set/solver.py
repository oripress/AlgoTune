class Solver:
    def solve(self, problem, **kwargs):
        adj = problem["adj_matrix"]
        weights = problem["weights"]
        n = len(weights)
        # Order vertices by descending weight
        verts = list(range(n))
        verts.sort(key=lambda i: weights[i], reverse=True)
        total_sum = sum(weights)
        best_weight = 0
        best_solution = []
        curr_solution = []

        def dfs(cands, sum_cands, curr_weight):
            nonlocal best_weight, best_solution, curr_solution
            # Bound: prune if potential max <= best found
            if curr_weight + sum_cands <= best_weight:
                return
            if not cands:
                best_weight = curr_weight
                best_solution = curr_solution.copy()
                return
            # consider next vertex
            v = cands[0]
            wv = weights[v]
            # Branch: include v
            # filter out neighbors of v
            adj_v = adj[v]
            new_cands = []
            new_sum = 0
            for u in cands[1:]:
                if adj_v[u] == 0:
                    new_cands.append(u)
                    new_sum += weights[u]
            curr_solution.append(v)
            dfs(new_cands, new_sum, curr_weight + wv)
            curr_solution.pop()
            # Branch: exclude v
            dfs(cands[1:], sum_cands - wv, curr_weight)

        dfs(verts, total_sum, 0)
        return sorted(best_solution)