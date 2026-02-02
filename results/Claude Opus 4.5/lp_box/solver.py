import numpy as np
from scipy.optimize import linprog
from numba import njit, prange

@njit(cache=True)
def check_constraints(A, x, b):
    m = A.shape[0]
    for i in range(m):
        s = 0.0
        for j in range(A.shape[1]):
            s += A[i, j] * x[j]
        if s > b[i] + 1e-9:
            return False
    return True

@njit(cache=True)
def get_unconstrained_box_solution(c):
    n = len(c)
    x = np.empty(n)
    for i in range(n):
        if c[i] < 0:
            x[i] = 1.0
        else:
            x[i] = 0.0
    return x

class Solver:
    def solve(self, problem, **kwargs):
        c = np.asarray(problem["c"], dtype=np.float64)
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        
        # For box-constrained LP without other constraints, optimal solution is:
        # x_i = 0 if c_i >= 0, x_i = 1 if c_i < 0
        x_unconstrained = get_unconstrained_box_solution(c)
        
        # Check if this solution satisfies Ax <= b
        if check_constraints(A, x_unconstrained, b):
            return {"solution": x_unconstrained.tolist()}
        
        # Otherwise, solve full LP
        result = linprog(c, A_ub=A, b_ub=b, bounds=(0, 1), method='highs')
        
        return {"solution": result.x.tolist()}