import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the optimal advertising problem using scipy.optimize.linprog.
        
        :param problem: Dictionary with problem parameters
        :return: Dictionary with optimal displays and revenue
        """
        # Extract problem parameters
        P = np.array(problem["P"])
        R = np.array(problem["R"])
        B = np.array(problem["B"])
        c = np.array(problem["c"])
        T = np.array(problem["T"])
        
        # Derive m and n from P matrix
        m, n = P.shape
        
        # Variables: [D_11, D_12, ..., D_mn, z_1, ..., z_m]
        # Total variables: m*n + m
        num_vars = m * n + m
        
        # Objective: maximize sum(z_i) => minimize -sum(z_i)
        c_obj = np.zeros(num_vars)
        c_obj[m*n:] = -1  # Coefficients for z variables
        
        # Inequality constraints: A_ub @ x <= b_ub
        # Preallocate constraint matrix
        num_constraints = n + m + m + m  # Traffic + Revenue + Budget + Min display
        A_ub = np.zeros((num_constraints, num_vars))
        b_ub = np.zeros(num_constraints)
        
        row = 0
        
        # Traffic capacity constraints: sum_i D_ij <= T_j for each j
        for j in range(n):
            for i in range(m):
                A_ub[row, i * n + j] = 1
            b_ub[row] = T[j]
            row += 1
        
        # Revenue constraints: -R_i * sum_j P_ij * D_ij + z_i <= 0 for each i
        for i in range(m):
            for j in range(n):
                A_ub[row, i * n + j] = -R[i] * P[i, j]
            A_ub[row, m * n + i] = 1  # z_i coefficient
            b_ub[row] = 0
            row += 1
        
        # Budget constraints: z_i <= B_i for each i
        for i in range(m):
            A_ub[row, m * n + i] = 1
            b_ub[row] = B[i]
            row += 1
        
        # Minimum display requirements: -sum_j D_ij <= -c_i
        for i in range(m):
            for j in range(n):
                A_ub[row, i * n + j] = -1
            b_ub[row] = -c[i]
            row += 1
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds: All variables >= 0
        bounds = [(0, None) for _ in range(num_vars)]
        
        # Solve
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            return {"status": "failed", "optimal": False}
        
        # Extract solution
        x = result.x
        D_val = x[:m*n].reshape((m, n))
        z_val = x[m*n:]
        
        # Calculate clicks and revenue
        clicks = np.sum(P * D_val, axis=1)
        revenue = np.minimum(R * clicks, B)
        
        return {
            "status": "optimal",
            "optimal": True,
            "displays": D_val.tolist(),
            "clicks": clicks.tolist(),
            "revenue_per_ad": revenue.tolist(),
            "total_revenue": float(np.sum(revenue)),
            "objective_value": float(-result.fun),
        }