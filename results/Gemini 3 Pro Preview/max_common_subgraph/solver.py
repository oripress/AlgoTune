import time
import sys
from solver_cython import MCSolver

sys.setrecursionlimit(5000)

class Solver:
    def solve(self, problem, **kwargs):
        A = problem["A"]
        B = problem["B"]
        n = len(A)
        m = len(B)

        # Precompute adjacency bitmasks
        adj_a = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if A[i][j]:
                    mask |= (1 << j)
            adj_a[i] = mask

        adj_b = [0] * m
        for i in range(m):
            mask = 0
            for j in range(m):
                if B[i][j]:
                    mask |= (1 << j)
            adj_b[i] = mask

        solver = MCSolver(adj_a, adj_b, n, m)
        return solver.solve()