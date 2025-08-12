import random
from ortools.sat.python import cp_model
import numpy as np
from typing import List, Any

class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> Any:
        """
        Solves the minimum dominating set problem using OR-Tools CP-SAT.
        
        :param problem: A 2d adjacency matrix representing the graph.
        :return: A list of node indices included in the minimum dominating set.
        """
        n = len(problem)
        model = cp_model.CpModel()
        
        # Create a boolean variable for each vertex: 1 if included in the dominating set, 0 otherwise.
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        
        # Add domination constraints: For every node i, at least one of {i} ∪ neighbors(i) must be selected.
        for i in range(n):
            neighbors = [nodes[i]]  # Node i can dominate itself
            for j in range(n):
                if problem[i][j] == 1:
                    neighbors.append(nodes[j])
            model.Add(sum(neighbors) >= 1)
        
        # Objective: Minimize the number of vertices chosen.
        model.Minimize(sum(nodes))
        
        # Solve the model.
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL:
            # Extract and return nodes with value 1.
            selected = [i for i in range(n) if solver.Value(nodes[i]) == 1]
            return selected
        else:
            return []