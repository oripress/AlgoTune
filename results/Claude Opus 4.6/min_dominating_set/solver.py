from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n == 0:
            return []
        if n == 1:
            return [0]

        # Build adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n):
            row = problem[i]
            for j in range(n):
                if row[j]:
                    adj[i].append(j)

        # Variables: x_i = i+1 (1-indexed for SAT)
        wcnf = WCNF()

        # Hard clauses: for each vertex i, at least one of {i} ∪ N(i) must be selected
        for i in range(n):
            clause = [i + 1] + [j + 1 for j in adj[i]]
            wcnf.append(clause)

        # Soft clauses: prefer NOT selecting each vertex (minimize set size)
        for i in range(n):
            wcnf.append([-(i + 1)], weight=1)

        with RC2(wcnf) as solver:
            model = solver.compute()
            if model is not None:
                selected = [v - 1 for v in model if v > 0 and v <= n]
                return selected
            else:
                return list(range(n))