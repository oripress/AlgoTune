import numpy as np
from scipy import sparse
import ecos

class Solver:
    def solve(self, problem, **kwargs):
        p0 = np.array(problem["p0"], dtype=float)
        v0 = np.array(problem["v0"], dtype=float)
        p_target = np.array(problem["p_target"], dtype=float)
        g = float(problem["g"])
        m = float(problem["m"])
        h = float(problem["h"])
        K = int(problem["K"])
        F_max = float(problem["F_max"])
        gamma_val = float(problem["gamma"])
        
        n = 4 * K  # variables: F (3K) + s (K)
        
        # Objective: minimize gamma * sum(s_t)
        c = np.zeros(n)
        c[3*K:] = gamma_val
        
        # === Equality constraints (6 rows) ===
        t_idx = np.arange(K, dtype=np.int64)
        coeff_pos = (2*K - 2*t_idx - 1).astype(float)
        
        A_rows_arr = np.concatenate([
            np.full(K, 0, dtype=np.int64), np.full(K, 1, dtype=np.int64), np.full(K, 2, dtype=np.int64),
            np.full(K, 3, dtype=np.int64), np.full(K, 4, dtype=np.int64), np.full(K, 5, dtype=np.int64)
        ])
        A_cols_arr = np.concatenate([
            3*t_idx, 3*t_idx + 1, 3*t_idx + 2,
            3*t_idx, 3*t_idx + 1, 3*t_idx + 2
        ])
        A_vals_arr = np.concatenate([
            np.ones(K), np.ones(K), np.ones(K),
            coeff_pos, coeff_pos, coeff_pos
        ])
        
        b_eq = np.zeros(6)
        b_eq[0] = -m * v0[0] / h
        b_eq[1] = -m * v0[1] / h
        b_eq[2] = m * K * g - m * v0[2] / h
        
        scale = 2 * m / (h * h)
        b_eq[3] = scale * (p_target[0] - p0[0] - K * h * v0[0])
        b_eq[4] = scale * (p_target[1] - p0[1] - K * h * v0[1])
        b_eq[5] = scale * (p_target[2] - p0[2] - K * h * v0[2] + K**2 * h**2 * g / 2)
        
        A_eq = sparse.csc_matrix((A_vals_arr, (A_rows_arr, A_cols_arr)), shape=(6, n))
        
        # === Inequality constraints ===
        n_height = K - 1
        n_lin = n_height + K  # height + s >= 0 (rewritten)
        n_soc_rows = 4 * K
        total_ineq = n_lin + n_soc_rows
        
        h_ineq = np.zeros(total_ineq)
        
        hh2m = h * h / (2 * m)
        
        all_G_rows = []
        all_G_cols = []
        all_G_vals = []
        
        if K > 1:
            # Height constraint: P[t,2] >= 0 for t=1..K-1
            # P[t,2] = p0[2] + t*h*v0[2] - t^2*h^2*g/2 + (h^2/(2m)) * sum_{r=0}^{t-1} (2t-2r-1)*F[r,2]
            # We need -P[t,2] <= 0, i.e. G*x <= h where G has negative coefficients
            for t in range(1, K):
                row = t - 1
                for r in range(t):
                    coeff = 2*t - 2*r - 1
                    all_G_rows.append(row)
                    all_G_cols.append(3*r + 2)
                    all_G_vals.append(-hh2m * coeff)
                h_ineq[row] = p0[2] + t * h * v0[2] - t**2 * h**2 * g / 2
        
        # s_t <= F_max  =>  s_t - F_max <= 0
        s_offset = n_height
        s_idx = np.arange(K, dtype=np.int64)
        all_G_rows.extend((s_offset + s_idx).tolist())
        all_G_cols.extend((3*K + s_idx).tolist())
        all_G_vals.extend(np.ones(K).tolist())
        h_ineq[s_offset:s_offset+K] = F_max
        
        # SOC constraints: ||F[t]|| <= s_t
        soc_offset = n_lin
        for t in range(K):
            base = soc_offset + 4 * t
            # First row: -s_t
            all_G_rows.append(base)
            all_G_cols.append(3*K + t)
            all_G_vals.append(-1.0)
            # Rows 1-3: -F[t,0], -F[t,1], -F[t,2]
            for j in range(3):
                all_G_rows.append(base + 1 + j)
                all_G_cols.append(3*t + j)
                all_G_vals.append(-1.0)
        
        G_rows = np.array(all_G_rows, dtype=np.int64)
        G_cols = np.array(all_G_cols, dtype=np.int64)
        G_vals = np.array(all_G_vals, dtype=float)
        
        G = sparse.csc_matrix((G_vals, (G_rows, G_cols)), shape=(total_ineq, n))
        
        dims = {'l': n_lin, 'q': [4] * K}
        
        sol = ecos.solve(c, G, h_ineq, dims, A_eq, b_eq, verbose=False)
        
        if sol['info']['exitFlag'] not in (0, 1, 10):
            return {"position": [], "velocity": [], "thrust": []}
        
        x = sol['x']
        F_sol = x[:3*K].reshape(K, 3)
        
        # Reconstruct V and P from dynamics
        delta_V = np.zeros((K, 3))
        delta_V[:, :2] = h * F_sol[:, :2] / m
        delta_V[:, 2] = h * (F_sol[:, 2] / m - g)
        
        V = np.zeros((K + 1, 3))
        V[0] = v0
        V[1:] = v0 + np.cumsum(delta_V, axis=0)
        
        P = np.zeros((K + 1, 3))
        P[0] = p0
        delta_P = h / 2 * (V[:-1] + V[1:])
        P[1:] = p0 + np.cumsum(delta_P, axis=0)
        
        fuel = gamma_val * np.sum(np.linalg.norm(F_sol, axis=1))
        
        return {
            "position": P.tolist(),
            "velocity": V.tolist(),
            "thrust": F_sol.tolist(),
            "fuel_consumption": float(fuel),
        }