from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, list], **kwargs):
        """
        Optimized solver for Maximum Weighted Independent Set.
        """
        adj_matrix = problem["adj_matrix"]
        weights = problem["weights"]
        n = len(adj_matrix)

        # Quick check for trivial cases
        if n == 0:
            return []
        if n == 1:
            return [0] if weights[0] > 0 else []

        # Build adjacency list and edge list
        adj_list = [[] for _ in range(n)]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i][j]:
                    edges.append((i, j))
                    adj_list[i].append(j)
                    adj_list[j].append(i)

        # If no edges, return all nodes with positive weights
        if not edges:
            return [i for i in range(n) if weights[i] > 0]

        # Use CP-SAT with greedy hints
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Add constraints for edges
        for i, j in edges:
            model.Add(nodes[i] + nodes[j] <= 1)

        # Maximize weighted sum
        model.Maximize(sum(weights[i] * nodes[i] for i in range(n)))

        # Get greedy hint
        greedy_solution = self._greedy_heuristic(adj_list, weights, n)
        for i in range(n):
            model.AddHint(nodes[i], 1 if i in greedy_solution else 0)

        # Configure solver for speed
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        solver.parameters.num_search_workers = 1

        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(nodes[i])]
        else:
            return []
    
    def _greedy_heuristic(self, adj_list, weights, n):
        """Fast greedy heuristic for initial solution."""
        # Sort by weight/degree ratio
        degrees = [len(adj_list[i]) if adj_list[i] else 1 for i in range(n)]
        nodes_sorted = sorted(range(n), key=lambda i: -weights[i] / degrees[i] if weights[i] > 0 else float('-inf'))
        
        selected = []
        excluded = set()
        
        for node in nodes_sorted:
            if node not in excluded and weights[node] > 0:
                selected.append(node)
                excluded.update(adj_list[node])
        
        return selected