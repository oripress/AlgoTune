import random
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solves the minimum dominating set problem using the CP-SAT solver.
        :param problem: A 2d adjacency matrix representing the graph.
        :return: A list of node indices included in the minimum dominating set.
        """
        n = len(problem)
        model = cp_model.CpModel()
        # Boolean variable for each vertex: 1 if included in the dominating set
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        # Domination constraints
        for i in range(n):
            neigh_vars = [nodes[i]]
            for j in range(n):
                if problem[i][j] == 1:
                    neigh_vars.append(nodes[j])
            model.Add(sum(neigh_vars) >= 1)
        # Objective: minimize the number of selected vertices
        model.Minimize(sum(nodes))
        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL:
            return [i for i in range(n) if solver.Value(nodes[i]) == 1]
        else:
            return []