import numpy as np
from ortools.linear_solver import pywraplp

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        P = np.array(problem["P"], dtype=float)
        R = np.array(problem["R"], dtype=float)
        B = np.array(problem["B"], dtype=float)
        c = np.array(problem["c"], dtype=float)
        T = np.array(problem["T"], dtype=float)
        
        m, n = P.shape
        
        solver = pywraplp.Solver.CreateSolver('GLOP')
        if not solver:
            return {"status": "error", "optimal": False}
            
        inf = solver.infinity()
        
        # Variables
        # D_it: m x n matrix of variables
        D = [[solver.NumVar(0, inf, '') for _ in range(n)] for _ in range(m)]
            
        # r_i: m variables for revenue
        r = [solver.NumVar(0, B[i], '') for i in range(m)]
            
        # Constraints
        # 1. r_i <= R_i * sum_t P_it * D_it
        for i in range(m):
            constraint = solver.Constraint(-inf, 0)
            constraint.SetCoefficient(r[i], 1.0)
            R_i = R[i]
            P_i = P[i]
            D_i = D[i]
            for t in range(n):
                constraint.SetCoefficient(D_i[t], -R_i * P_i[t])
                
        # 2. sum_i D_it <= T_t
        for t in range(n):
            constraint = solver.Constraint(-inf, T[t])
            for i in range(m):
                constraint.SetCoefficient(D[i][t], 1.0)
                
        # 3. sum_t D_it >= c_i
        for i in range(m):
            constraint = solver.Constraint(c[i], inf)
            D_i = D[i]
            for t in range(n):
                constraint.SetCoefficient(D_i[t], 1.0)
                
        # Objective: maximize sum(r_i)
        objective = solver.Objective()
        for i in range(m):
            objective.SetCoefficient(r[i], 1.0)
        objective.SetMaximization()
        
        status = solver.Solve()
        
        if status != pywraplp.Solver.OPTIMAL:
            return {"status": "error", "optimal": False}
            
        D_val = np.zeros((m, n))
        for i in range(m):
            D_i = D[i]
            for t in range(n):
                D_val[i, t] = D_i[t].solution_value()
                
        clicks = np.sum(P * D_val, axis=1)
        revenue = np.minimum(R * clicks, B)
        
        return {
            "status": "optimal",
            "optimal": True,
            "displays": D_val.tolist(),
            "clicks": clicks.tolist(),
            "revenue_per_ad": revenue.tolist(),
            "total_revenue": float(np.sum(revenue)),
            "objective_value": float(objective.Value())
        }