from typing import Any, Dict
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the rocket landing optimization problem using a reduced SOCP:
        - Optimize only over thrust F[t] with analytically eliminated V and P
        - Enforce terminal and height constraints as linear functions of F
        - Reconstruct full trajectories post-solve

        Returns:
            dict with keys: position, velocity, thrust, fuel_consumption
        """
        # Extract problem parameters
        try:
            p0 = np.asarray(problem["p0"], dtype=float)
            v0 = np.asarray(problem["v0"], dtype=float)
            p_target = np.asarray(problem["p_target"], dtype=float)
            g = float(problem["g"])
            m = float(problem["m"])
            h = float(problem["h"])
            K = int(problem["K"])
            F_max = float(problem["F_max"])
            gamma = float(problem["gamma"])
        except Exception:
            return {"position": [], "velocity": [], "thrust": []}

        # Basic validation
        if (
            p0.shape != (3,)
            or v0.shape != (3,)
            or p_target.shape != (3,)
            or not (K >= 1 and m > 0 and h > 0 and F_max >= 0 and gamma >= 0)
        ):
            return {"position": [], "velocity": [], "thrust": []}

        # Optimization variable: thrust over K steps (K, 3)
        F = cp.Variable((K, 3))

        # Precompute constants
        # Terminal velocity: V[K] = 0 =>
        # sum(F, axis=0) = [-m/h * v0_x, -m/h * v0_y, g*m*K - m/h * v0_z]
        sumF_target = np.array(
            [
                -m / h * v0[0],
                -m / h * v0[1],
                g * m * K - m / h * v0[2],
            ],
            dtype=float,
        )

        # Final position P[K] equality:
        # P[K] = p0 + h*K*v0 + (h^2/m) * sum_{i=0}^{K-1} (K - i - 0.5) * F[i]
        #        + [0, 0, -0.5*h^2*g*K^2]
        w = (np.arange(K, dtype=float)[::-1] + 0.5)  # w_i = K - i - 0.5
        pK_const = p0 + h * K * v0 + np.array([0.0, 0.0, -0.5 * h * h * g * K * K])
        pos_final_expr = pK_const + (h * h / m) * cp.sum(cp.multiply(w[:, None], F), axis=0)

        # Height constraints: for all t=0..K, P_z[t] >= 0
        # P_z[t] = p0_z + h*t*v0_z + (h^2/m) * sum_{i=0}^{t-1} (t - i - 0.5) * F[i,2]
        #          - 0.5*h^2*g*t^2
        t_vec = np.arange(K + 1, dtype=float)  # 0..K
        W = np.zeros((K + 1, K), dtype=float)
        for t in range(1, K + 1):
            W[t, :t] = (t - 0.5) - np.arange(t, dtype=float)

        Pz_vec_expr = (
            p0[2]
            + h * t_vec * v0[2]
            + (h * h / m) * cp.matmul(W, F[:, 2])
            - 0.5 * h * h * g * (t_vec ** 2)
        )

        # Constraints list
        constraints = []

        # Terminal velocity constraint via sum(F) equalities
        constraints.append(cp.sum(F, axis=0) == sumF_target)

        # Final position constraint
        constraints.append(pos_final_expr == p_target)

        # Height constraints element-wise
        constraints.append(Pz_vec_expr >= 0)

        # Maximum thrust constraint
        if F_max < np.inf:
            constraints.append(cp.norm(F, 2, axis=1) <= F_max)

        # Objective: minimize fuel consumption
        fuel_expr = gamma * cp.sum(cp.norm(F, axis=1))
        objective = cp.Minimize(fuel_expr)

        # Solve the SOCP with ECOS, fallback to SCS if needed
        prob = cp.Problem(objective, constraints)
        solved = False
        try:
            prob.solve(solver=cp.ECOS, verbose=False, warm_start=True)
            solved = prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        except Exception:
            solved = False

        if not solved:
            try:
                prob.solve(solver=cp.SCS, verbose=False, warm_start=True)
                solved = prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
            except Exception:
                solved = False

        if not solved or F.value is None:
            return {"position": [], "velocity": [], "thrust": []}

        # Retrieve optimized thrust
        F_opt = np.asarray(F.value, dtype=float)

        # Reconstruct velocity and position trajectories to satisfy dynamics exactly
        # V[0] = v0; for t>=1:
        # V[t,:2] = v0[:2] + (h/m) * sum_{i=0}^{t-1} F[i,:2]
        # V[t,2]  = v0[2] + (h/m) * sum_{i=0}^{t-1} F[i,2] - h*g*t
        S = np.vstack([np.zeros((1, 3), dtype=float), np.cumsum(F_opt, axis=0)])  # cumulative sum with S[0]=0
        V = np.empty((K + 1, 3), dtype=float)
        V[:, :2] = v0[:2] + (h / m) * S[:, :2]
        V[:, 2] = v0[2] + (h / m) * S[:, 2] - h * g * np.arange(K + 1, dtype=float)

        # P[0] = p0; P[t+1] = P[t] + h/2 (V[t] + V[t+1])
        P = np.empty((K + 1, 3), dtype=float)
        P[0] = p0
        for t in range(K):
            P[t + 1] = P[t] + 0.5 * h * (V[t] + V[t + 1])

        # Compute fuel consumption
        fuel_consumption = float(gamma * np.sum(np.linalg.norm(F_opt, axis=1)))

        return {
            "position": P.tolist(),
            "velocity": V.tolist(),
            "thrust": F_opt.tolist(),
            "fuel_consumption": fuel_consumption,
        }