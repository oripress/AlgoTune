class Solver:
    def solve(self, problem, **kwargs):
        """
        Solves the Maximum Weighted Independent Set problem.
        
        :param problem: dict with 'adj_matrix' and 'weights'
        :return: list of selected node indices
        """
        adj_matrix = problem["adj_matrix"]
        weights = problem["weights"]
        n = len(adj_matrix)
        
        # For small graphs, use exact dynamic programming
        if n <= 20:
            return self._solve_dp(adj_matrix, weights, n)
        else:
            # For larger graphs, use branch and bound
            return self._solve_branch_and_bound(adj_matrix, weights, n)
    
    def _solve_dp(self, adj_matrix, weights, n):
        """Dynamic programming solution for small graphs"""
        # dp[mask] = maximum weight of independent set from nodes in mask
        dp = {}
        parent = {}
        
        # Base case
        dp[0] = 0
        parent[0] = -1
        
        # Try all possible subsets
        for mask in range(1, 1 << n):
            dp[mask] = 0
            parent[mask] = -1
            
            # Check if this is a valid independent set
            valid = True
            for i in range(n):
                if mask & (1 << i):
                    for j in range(i + 1, n):
                        if (mask & (1 << j)) and adj_matrix[i][j]:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Calculate weight
                weight = sum(weights[i] for i in range(n) if mask & (1 << i))
                dp[mask] = weight
        
        # Find the mask with maximum weight
        best_mask = max(dp.keys(), key=lambda x: dp[x])
        
        # Extract nodes from best mask
        result = [i for i in range(n) if best_mask & (1 << i)]
        return sorted(result)
    
    def _solve_branch_and_bound(self, adj_matrix, weights, n):
        """Branch and bound solution for larger graphs"""
        # Start with greedy solution as initial bound
        initial_solution = self._greedy_solution(adj_matrix, weights, n)
        best_weight = sum(weights[i] for i in initial_solution)
        best_solution = initial_solution[:]
        
        # Branch and bound
        def branch_and_bound(current, remaining, current_weight):
            nonlocal best_weight, best_solution
            
            if not remaining:
                if current_weight > best_weight:
                    best_weight = current_weight
                    best_solution = current[:]
                return
            
            # Upper bound: current weight + sum of remaining weights
            upper_bound = current_weight + sum(weights[i] for i in remaining)
            if upper_bound <= best_weight:
                return  # Prune
            
            # Try including the next node
            node = remaining[0]
            new_remaining = remaining[1:]
            
            # Check if we can include this node
            can_include = True
            for selected in current:
                if adj_matrix[node][selected]:
                    can_include = False
                    break
            
            if can_include:
                # Include node
                new_current = current + [node]
                # Remove neighbors from remaining
                new_remaining_filtered = [i for i in new_remaining if not adj_matrix[node][i]]
                branch_and_bound(new_current, new_remaining_filtered, current_weight + weights[node])
            
            # Try not including the node
            branch_and_bound(current, new_remaining, current_weight)
        
        # Sort nodes by weight in descending order for better pruning
        nodes = list(range(n))
        nodes.sort(key=lambda x: weights[x], reverse=True)
        
        branch_and_bound([], nodes, 0)
        return sorted(best_solution)
    
    def _greedy_solution(self, adj_matrix, weights, n):
        """Get initial greedy solution"""
        # Sort nodes by weight in descending order
        nodes = list(range(n))
        nodes.sort(key=lambda x: weights[x], reverse=True)
        
        selected = []
        for node in nodes:
            # Check if we can add this node
            can_add = True
            for s in selected:
                if adj_matrix[node][s]:
                    can_add = False
                    break
            if can_add:
                selected.append(node)
        
        return selected