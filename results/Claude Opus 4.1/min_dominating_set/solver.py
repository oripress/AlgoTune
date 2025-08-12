from typing import Any
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solves the minimum dominating set problem using PuLP with optimizations.
        
        :param problem: A 2d adjacency matrix representing the graph.
        :return: A list of node indices included in the minimum dominating set.
        """
        n = len(problem)
        
        # Convert to numpy for faster operations
        adj_matrix = np.array(problem, dtype=np.int8)
        
        # Quick greedy heuristic for upper bound
        greedy_solution = self._greedy_dominating_set(adj_matrix)
        upper_bound = len(greedy_solution)
        
        # Create the LP problem
        prob = LpProblem("MinDominatingSet", LpMinimize)
        
        # Decision variables: x_i = 1 if vertex i is in the dominating set
        x = [LpVariable(f"x_{i}", cat='Binary') for i in range(n)]
        
        # Objective: minimize the number of vertices in the dominating set
        prob += lpSum(x)
        
        # Constraints: each vertex must be dominated
        for i in range(n):
            # Vertex i is dominated if it's in the set or at least one neighbor is
            neighbors = [x[i]]
            for j in range(n):
                if adj_matrix[i][j] == 1:
                    neighbors.append(x[j])
            prob += lpSum(neighbors) >= 1
        
        # Add upper bound constraint
        prob += lpSum(x) <= upper_bound
        
        # Solve with CBC
        solver = PULP_CBC_CMD(msg=0, timeLimit=5)
        prob.solve(solver)
        
        # Extract solution
        if prob.status == 1:  # Optimal
            solution = [i for i in range(n) if x[i].varValue == 1]
            return solution
        else:
            # Fall back to greedy if solver fails
            return greedy_solution
    
    def _greedy_dominating_set(self, adj_matrix):
        """Fast greedy heuristic for dominating set."""
        n = len(adj_matrix)
        uncovered = set(range(n))
        dominating_set = []
        
        # Compute degrees (including self)
        coverage = np.sum(adj_matrix, axis=1) + 1
        
        while uncovered:
            # Find vertex that covers most uncovered vertices
            best_vertex = -1
            best_count = 0
            
            for v in range(n):
                if v in uncovered:
                    count = 1  # covers itself
                else:
                    count = 0
                    
                # Count uncovered neighbors
                for u in uncovered:
                    if u != v and adj_matrix[v][u] == 1:
                        count += 1
                
                if count > best_count:
                    best_count = count
                    best_vertex = v
            
            if best_vertex == -1:
                # No progress, add remaining uncovered
                dominating_set.extend(uncovered)
                break
            
            dominating_set.append(best_vertex)
            
            # Mark covered vertices
            covered = {best_vertex}
            for u in range(n):
                if adj_matrix[best_vertex][u] == 1:
                    covered.add(u)
            uncovered -= covered
        
        return dominating_set