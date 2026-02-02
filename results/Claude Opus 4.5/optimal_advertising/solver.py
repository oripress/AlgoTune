import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack, hstack, diags

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        """Solve optimal advertising problem using LP reformulation."""
        P = np.array(problem["P"], dtype=np.float64)
        R = np.array(problem["R"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        c_req = np.array(problem["c"], dtype=np.float64)
        T = np.array(problem["T"], dtype=np.float64)
        
        m, n = P.shape
        num_D = m * n
        num_vars = num_D + m
        
        # Objective: maximize sum(z) -> minimize -sum(z)
        c_obj = np.zeros(num_vars)
        c_obj[num_D:] = -1.0
        
        # Build constraint matrices efficiently
        # Variables: [D[0,0], D[0,1], ..., D[0,n-1], D[1,0], ..., D[m-1,n-1], z[0], z[1], ..., z[m-1]]
        
        # Constraint 1: z_i - R_i * sum_t P[i,t] * D[i,t] <= 0  (m constraints)
        rows = []
        cols = []
        data = []
        for i in range(m):
            for t in range(n):
                rows.append(i)
                cols.append(i * n + t)
                data.append(-R[i] * P[i, t])
        A1_D = csr_matrix((data, (rows, cols)), shape=(m, num_D))
        A1_z = diags([1.0] * m, 0, shape=(m, m), format='csr')
        A1 = hstack([A1_D, A1_z], format='csr')
        b1 = np.zeros(m)
        
        # Constraint 2: sum_i D[i,t] <= T[t]  (n constraints)
        rows = []
        cols = []
        data = []
        for t in range(n):
            for i in range(m):
                rows.append(t)
                cols.append(i * n + t)
                data.append(1.0)
        A2_D = csr_matrix((data, (rows, cols)), shape=(n, num_D))
        A2_z = csr_matrix((n, m))
        A2 = hstack([A2_D, A2_z], format='csr')
        b2 = T.copy()
        
        # Constraint 3: -sum_t D[i,t] <= -c_req[i]  (m constraints)
        rows = []
        cols = []
        data = []
        for i in range(m):
            for t in range(n):
                rows.append(i)
                cols.append(i * n + t)
                data.append(-1.0)
        A3_D = csr_matrix((data, (rows, cols)), shape=(m, num_D))
        A3_z = csr_matrix((m, m))
        A3 = hstack([A3_D, A3_z], format='csr')
        b3 = -c_req.copy()
        
        A_ub = vstack([A1, A2, A3], format='csr')
        b_ub = np.concatenate([b1, b2, b3])
        
        # Bounds: D >= 0, 0 <= z <= B
        bounds = [(0, None) for _ in range(num_D)] + [(0, B[i]) for i in range(m)]
        
        # Solve using HiGHS with tighter tolerances
        options = {'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9}
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
        
        if result.success:
            x = result.x
            D_val = x[:num_D].reshape((m, n))
            
            # Ensure non-negativity
            D_val = np.maximum(D_val, 0)
            
            # Hard enforce traffic constraints - clip any violations
            traffic_usage = np.sum(D_val, axis=0)
            for t in range(n):
                if traffic_usage[t] > T[t]:
                    if traffic_usage[t] > 1e-10:
                        scale = T[t] / traffic_usage[t]
                        D_val[:, t] *= scale
            
            # Verify and ensure minimum display constraints
            displays_per_ad = np.sum(D_val, axis=1)
            for i in range(m):
                if displays_per_ad[i] < c_req[i]:
                    # Find slots with remaining capacity and add displays
                    deficit = c_req[i] - displays_per_ad[i]
                    remaining_cap = T - np.sum(D_val, axis=0)
                    for t in range(n):
                        if deficit <= 0:
                            break
                        add = min(deficit, remaining_cap[t])
                        if add > 0:
                            D_val[i, t] += add
                            remaining_cap[t] -= add
                            deficit -= add
            
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
        else:
            return {"status": result.message, "optimal": False}