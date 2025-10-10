import pulp

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solves the minimum dominating set problem using PuLP with CBC solver.
        Optimized with greedy warm start.
        
        :param problem: A 2d adjacency matrix representing the graph.
        :return: A list of node indices included in the minimum dominating set.
        """
        n = len(problem)
        
        # Pre-compute adjacency lists
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if problem[i][j] == 1:
                    neighbors[i].append(j)
        
        # Get greedy solution as warm start
        greedy_sol = self._greedy_dominating_set(n, neighbors)
        
        # Create the LP problem
        prob = pulp.LpProblem("MinDominatingSet", pulp.LpMinimize)
        
        # Create binary variables with warm start
        nodes = []
        for i in range(n):
            var = pulp.LpVariable(f"x_{i}", cat='Binary')
            if i in greedy_sol:
                var.setInitialValue(1)
            else:
                var.setInitialValue(0)
            nodes.append(var)
        
        # Objective: minimize number of selected nodes
        prob += pulp.lpSum(nodes)
        
        # Constraints: each node must be dominated (optimized)
        for i in range(n):
            prob += nodes[i] + pulp.lpSum(nodes[j] for j in neighbors[i]) >= 1
        
        # Solve using CBC with optimized parameters
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=8, warmStart=True, threads=1)
        prob.solve(solver)
        
        if prob.status == pulp.LpStatusOptimal:
            return [i for i in range(n) if nodes[i].varValue == 1]
        else:
            return []
    
    def _greedy_dominating_set(self, n, neighbors):
        """Greedy algorithm to get initial solution."""
        dominated = [False] * n
        solution = set()
        
        while not all(dominated):
            # Find node that dominates most undominated nodes
            best_node = -1
            best_count = -1
            
            for i in range(n):
                if i in solution:
                    continue
                count = 0
                if not dominated[i]:
                    count += 1
                for j in neighbors[i]:
                    if not dominated[j]:
                        count += 1
                
                if count > best_count:
                    best_count = count
                    best_node = i
            
            if best_node == -1:
                break
            
            # Add node to solution
            solution.add(best_node)
            dominated[best_node] = True
            for j in neighbors[best_node]:
                dominated[j] = True
        
        return solution