from typing import Any
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Solve the rocket landing optimization by optimizing thrust vectors F (K x 3)
        using a reduced convex formulation (only F variables) and reconstructing
        positions and velocities from the discrete dynamics. Includes a least-squares
        fallback if the CVXPY solver fails.
        """
        # Parse inputs
        try:
            p0 = np.asarray(problem["p0"], dtype=float).reshape(3)
            v0 = np.asarray(problem["v0"], dtype=float).reshape(3)
            p_target = np.asarray(problem["p_target"], dtype=float).reshape(3)
            g = float(problem["g"])
            m = float(problem["m"])
            h = float(problem["h"])
            K = int(problem["K"])
            F_max = float(problem["F_max"])
            gamma = float(problem["gamma"])
        except Exception:
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": 0.0}

        # Trivial checks
        if K < 0:
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": 0.0}
        if K == 0:
            if np.allclose(p0, p_target, atol=1e-9) and np.linalg.norm(v0) < 1e-9:
                return {
                    "position": [p0.tolist()],
                    "velocity": [v0.tolist()],
                    "thrust": [],
                    "fuel_consumption": 0.0,
                }
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": 0.0}

        # Gravity acceleration vector (z is -g)
        g_vec = np.array([0.0, 0.0, -g], dtype=float)

        # Decision variable: thrust at each timestep
        F = cp.Variable((K, 3))

        constraints = []

        # Final velocity equality:
        # V[K] = v0 + (h/m) * sum_i F[i] + h*K*g_vec == 0
        sumF_target = -(m / h) * (v0 + h * K * g_vec)
        constraints.append(cp.sum(F, axis=0) == sumF_target)

        # Final position equality:
        # P[K] = p0 + h*K*v0 + (h^2/m)*sum_i F[i]*(K - i - 0.5) + h^2*g_vec*(K^2/2)
        coeffs = (h * h / m) * ((K - 0.5) - np.arange(K, dtype=float))  # shape (K,)
        rhs_p = p_target - p0 - h * K * v0 - (h * h) * g_vec * (K * K / 2.0)
        constraints.append(cp.sum(cp.multiply(F, coeffs.reshape((K, 1))), axis=0) == rhs_p)

        # Height constraints: P[t,2] >= 0 for t = 1..K
        # P[t] = p0 + h*t*v0 + (h^2/m)*sum_{i=0}^{t-1} F[i]*(t - i - 0.5) + h^2*g_vec*(t^2/2)
        for t in range(1, K + 1):
            a = (h * h / m) * ((t - 0.5) - np.arange(t, dtype=float))  # length t
            const_term = p0[2] + h * t * v0[2] + (h * h) * g_vec[2] * (t * t / 2.0)
            constraints.append(cp.sum(cp.multiply(F[:t, 2], a)) + const_term >= 0.0)

        # Thrust magnitude limits per timestep
        constraints.append(cp.norm(F, 2, axis=1) <= F_max)

        # Objective: minimize total fuel consumption (sum of thrust magnitudes)
        objective = cp.Minimize(gamma * cp.sum(cp.norm(F, axis=1)))
        prob = cp.Problem(objective, constraints)

        # Try to solve with ECOS, fallback to SCS
        solved = False
        try:
            prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)
            solved = True
        except Exception:
            try:
                prob.solve(solver=cp.SCS, verbose=False)
                solved = True
            except Exception:
                solved = False

        def reconstruct(Fval: np.ndarray):
            V = np.zeros((K + 1, 3), dtype=float)
            P = np.zeros((K + 1, 3), dtype=float)
            V[0] = v0.copy()
            P[0] = p0.copy()
            for t in range(K):
                # Velocity update: x,y and z separately
                V[t + 1, :2] = V[t, :2] + h * (Fval[t, :2] / m)
                V[t + 1, 2] = V[t, 2] + h * (Fval[t, 2] / m - g)
                # Position update (trapezoidal)
                P[t + 1] = P[t] + (h / 2.0) * (V[t] + V[t + 1])
            fuel = float(gamma * np.sum(np.linalg.norm(Fval, axis=1)))
            return P, V, fuel

        # If solver failed or returned no solution, fallback to least-squares on equalities
        if (not solved) or (prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}) or (F.value is None):
            try:
                # Build linear system A x = b for equality constraints (6 eqns)
                A = np.zeros((6, 3 * K), dtype=float)
                b = np.zeros(6, dtype=float)
                # Velocity equalities: sum(F) = sumF_target
                for d in range(3):
                    for i in range(K):
                        A[d, 3 * i + d] = 1.0
                    b[d] = sumF_target[d]
                # Position equalities: sum(coeffs[i] * F[i]) = rhs_p
                for d in range(3):
                    for i in range(K):
                        A[3 + d, 3 * i + d] = coeffs[i]
                    b[3 + d] = rhs_p[d]
                x, *_ = np.linalg.lstsq(A, b, rcond=None)
                if x.size != 3 * K:
                    return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": 0.0}
                Fval = x.reshape((K, 3))
                # Clip to F_max per timestep (preserve direction)
                norms = np.linalg.norm(Fval, axis=1)
                for i in range(K):
                    if norms[i] > F_max and norms[i] > 0:
                        Fval[i] = Fval[i] * (F_max / norms[i])
                P, V, fuel = reconstruct(Fval)
                if np.isnan(P).any() or np.isnan(V).any() or np.isnan(Fval).any():
                    return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": 0.0}
                return {
                    "position": P.tolist(),
                    "velocity": V.tolist(),
                    "thrust": Fval.tolist(),
                    "fuel_consumption": float(fuel),
                }
            except Exception:
                return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": 0.0}

        # Use CVXPY solution
        Fval = np.asarray(F.value, dtype=float)
        Fval[np.abs(Fval) < 1e-12] = 0.0

        P, V, fuel = reconstruct(Fval)

        if (np.isnan(P).any() or np.isnan(V).any() or np.isnan(Fval).any()
                or np.isinf(P).any() or np.isinf(V).any() or np.isinf(Fval).any()):
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": 0.0}

        return {
            "position": P.tolist(),
            "velocity": V.tolist(),
            "thrust": Fval.tolist(),
            "fuel_consumption": float(fuel),
        }