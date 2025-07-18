import numpy as np
import scipy.sparse as sp
import ecos

class Solver:
    def solve(self, problem, **kwargs):
        # Extract problem parameters
        p0 = np.array(problem["p0"], dtype=float)
        v0 = np.array(problem["v0"], dtype=float)
        p_target = np.array(problem["p_target"], dtype=float)
        g = float(problem["g"])
        m = float(problem["m"])
        h = float(problem["h"])
        K = int(problem["K"])
        F_max = float(problem["F_max"])
        gamma = float(problem["gamma"])

        # Variable indices and counts
        nF = 3 * K
        nP = 3 * (K + 1)
        nV = 3 * (K + 1)
        nu = K
        idx_F = 0
        idx_P = idx_F + nF
        idx_V = idx_P + nP
        idx_u = idx_V + nV
        n = idx_u + nu

        # Objective: minimize gamma * sum(u)
        c = np.zeros(n, dtype=float)
        if K > 0:
            c[idx_u:idx_u + K] = gamma

        # Equality constraints Ax = b
        eq_rows = []
        eq_cols = []
        eq_data = []
        b = []
        eq_r = 0

        # Initial velocity V[0] = v0
        for d in range(3):
            eq_rows.append(eq_r); eq_cols.append(idx_V + d); eq_data.append(1.0)
            b.append(v0[d]); eq_r += 1
        # Initial position P[0] = p0
        for d in range(3):
            eq_rows.append(eq_r); eq_cols.append(idx_P + d); eq_data.append(1.0)
            b.append(p0[d]); eq_r += 1

        # Dynamics: velocity updates
        for t in range(K):
            # x, y components
            for d in range(2):
                eq_rows.append(eq_r); eq_cols.append(idx_V + 3*(t+1) + d); eq_data.append(1.0)
                eq_rows.append(eq_r); eq_cols.append(idx_V + 3*t + d); eq_data.append(-1.0)
                eq_rows.append(eq_r); eq_cols.append(idx_F + 3*t + d); eq_data.append(-h/m)
                b.append(0.0); eq_r += 1
            # z component with gravity
            eq_rows.append(eq_r); eq_cols.append(idx_V + 3*(t+1) + 2); eq_data.append(1.0)
            eq_rows.append(eq_r); eq_cols.append(idx_V + 3*t + 2); eq_data.append(-1.0)
            eq_rows.append(eq_r); eq_cols.append(idx_F + 3*t + 2); eq_data.append(-h/m)
            b.append(-h * g); eq_r += 1

        # Dynamics: position updates
        for t in range(K):
            for d in range(3):
                eq_rows.append(eq_r); eq_cols.append(idx_P + 3*(t+1) + d); eq_data.append(1.0)
                eq_rows.append(eq_r); eq_cols.append(idx_P + 3*t + d); eq_data.append(-1.0)
                eq_rows.append(eq_r); eq_cols.append(idx_V + 3*t + d); eq_data.append(-h/2)
                eq_rows.append(eq_r); eq_cols.append(idx_V + 3*(t+1) + d); eq_data.append(-h/2)
                b.append(0.0); eq_r += 1

        # Terminal conditions V[K] = 0
        for d in range(3):
            eq_rows.append(eq_r); eq_cols.append(idx_V + 3*K + d); eq_data.append(1.0)
            b.append(0.0); eq_r += 1
        # Terminal conditions P[K] = p_target
        for d in range(3):
            eq_rows.append(eq_r); eq_cols.append(idx_P + 3*K + d); eq_data.append(1.0)
            b.append(p_target[d]); eq_r += 1

        # Build sparse A and b
        A = sp.coo_matrix((eq_data, (eq_rows, eq_cols)), shape=(eq_r, n)).tocsc()
        b = np.array(b, dtype=float)

        # Inequality and SOC constraints: Gx + s = h, s in cones
        l = K + 1
        q = [4] * (2 * K)
        total_rows = l + sum(q)
        G_rows = []
        G_cols = []
        G_data = []
        h_vec = np.zeros(total_rows, dtype=float)

        # Linear height constraints: P[t,2] >= 0 -> -P[t,2] <= 0
        for t in range(K + 1):
            r = t
            G_rows.append(r); G_cols.append(idx_P + 3*t + 2); G_data.append(-1.0)
            h_vec[r] = 0.0

        # SOC: u_t >= ||F_t||
        for t in range(K):
            base = l + 4*t
            # u slack
            G_rows.append(base); G_cols.append(idx_u + t); G_data.append(-1.0)
            h_vec[base] = 0.0
            # F components
            for d in range(3):
                r = base + 1 + d
                G_rows.append(r); G_cols.append(idx_F + 3*t + d); G_data.append(1.0)
                h_vec[r] = 0.0

        # SOC: ||F_t|| <= F_max
        start = l + 4*K
        for t in range(K):
            base = start + 4*t
            # constant in h
            h_vec[base] = F_max
            # F components
            for d in range(3):
                r = base + 1 + d
                G_rows.append(r); G_cols.append(idx_F + 3*t + d); G_data.append(1.0)
                h_vec[r] = 0.0

        G = sp.coo_matrix((G_data, (G_rows, G_cols)), shape=(total_rows, n)).tocsc()
        dims = {"l": l, "q": q, "s": []}

        # Solve conic problem
        sol = ecos.solve(c, G, h_vec, dims, A=A, b=b, verbose=False)
        x = sol["x"]

        # Extract solution components
        F_flat = x[idx_F:idx_F + nF]
        P_flat = x[idx_P:idx_P + nP]
        V_flat = x[idx_V:idx_V + nV]

        F = F_flat.reshape(K, 3)
        P = P_flat.reshape(K + 1, 3)
        V = V_flat.reshape(K + 1, 3)

        # Compute fuel consumption
        fuel = gamma * np.sum(np.linalg.norm(F, axis=1))

        return {
            "position": P.tolist(),
            "velocity": V.tolist(),
            "thrust": F.tolist(),
            "fuel_consumption": float(fuel),
        }