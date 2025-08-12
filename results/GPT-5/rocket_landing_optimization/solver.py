from typing import Any, Dict

import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the rocket landing optimization problem using a reduced SOCP:
        - Optimize only over thrust F[0..K-1] in R^3
        - Enforce terminal velocity and position as linear equalities in F
        - Enforce height (z >= 0) as linear inequalities in F
        - Enforce per-step thrust norm bound and minimize sum of thrust norms
        - Reconstruct velocity and position post-solve

        :param problem: Dictionary with problem parameters
        :return: Dictionary with position, velocity, thrust, and fuel consumption
        """
        try:
            # Extract problem parameters
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
            # Invalid input formatting
            return {"position": [], "velocity": [], "thrust": []}

        if not (np.isfinite(p0).all() and np.isfinite(v0).all() and np.isfinite(p_target).all()):
            return {"position": [], "velocity": [], "thrust": []}
        # Decision variable: thrust over K steps (shape K x 3)
        F_var = cp.Variable((K, 3))

        constraints = []

        # Thrust magnitude constraints per step
        constraints.append(cp.norm(F_var, axis=1) <= F_max)

        # XY dynamics variables and constraints
        V_xy = cp.Variable((K + 1, 2))
        P_xy = cp.Variable((K + 1, 2))

        # Initial XY conditions
        constraints.append(V_xy[0] == v0[:2])
        constraints.append(P_xy[0] == p0[:2])

        # XY velocity dynamics: V_xy[t+1] = V_xy[t] + (h/m) * F_xy[t]
        constraints.append(V_xy[1:] == V_xy[:-1] + (h / m) * F_var[:, :2])

        # XY position dynamics (trapezoidal)
        constraints.append(P_xy[1:] == P_xy[:-1] + (h / 2.0) * (V_xy[:-1] + V_xy[1:]))

        # Terminal XY conditions
        constraints.append(V_xy[K] == 0.0)
        constraints.append(P_xy[K] == p_target[:2])

        # Z-dynamics variables to enforce height and terminal conditions efficiently
        v_z = cp.Variable(K + 1)
        p_z = cp.Variable(K + 1)

        # Initial Z conditions
        constraints.append(v_z[0] == v0[2])
        constraints.append(p_z[0] == p0[2])

        # Velocity z dynamics: v_z[t+1] = v_z[t] + (h/m) * F_z[t] - h*g
        constraints.append(v_z[1:] == v_z[:-1] + (h / m) * F_var[:, 2] - h * g)

        # Position z dynamics (trapezoidal)
        constraints.append(p_z[1:] == p_z[:-1] + (h / 2.0) * (v_z[:-1] + v_z[1:]))

        # Height non-negativity
        constraints.append(p_z >= 0.0)

        # Terminal z conditions
        constraints.append(v_z[K] == 0.0)
        constraints.append(p_z[K] == p_target[2])

        # Objective: minimize fuel consumption = gamma * sum ||F_t||
        objective = cp.Minimize(gamma * cp.sum(cp.norm(F_var, axis=1)))

        # Solve the SOCP
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=False, warm_start=True,
                       abstol=1e-9, reltol=1e-9, feastol=1e-9, max_iters=200)
        except Exception:
            return {"position": [], "velocity": [], "thrust": []}

        # If terminal conditions are not tightly satisfied, try a stricter re-solve
        try:
            if (
                V_xy.value is not None
                and P_xy.value is not None
                and v_z.value is not None
                and p_z.value is not None
            ):
                tight_eps = 1e-8
                if (
                    np.linalg.norm(np.asarray(V_xy.value)[-1]) > tight_eps
                    or abs(float(v_z.value[-1])) > tight_eps
                    or np.linalg.norm(np.asarray(P_xy.value)[-1] - p_target[:2]) > tight_eps
                    or abs(float(p_z.value[-1]) - float(p_target[2])) > tight_eps
                ):
                    prob.solve(solver=cp.ECOS, verbose=False, warm_start=True,
                               abstol=1e-12, reltol=1e-12, feastol=1e-12, max_iters=1000)
        except Exception:
            pass

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or F_var.value is None:
            return {"position": [], "velocity": [], "thrust": []}

        # Retrieve thrust and reconstruct velocity and position trajectories
        F = np.asarray(F_var.value, dtype=float)
        if not np.isfinite(F).all():
            return {"position": [], "velocity": [], "thrust": []}
        # Use optimized state variables directly to avoid accumulation error
        if (
            V_xy.value is None
            or P_xy.value is None
            or v_z.value is None
            or p_z.value is None
        ):
            return {"position": [], "velocity": [], "thrust": []}

        V_xy_val = np.asarray(V_xy.value, dtype=float)
        P_xy_val = np.asarray(P_xy.value, dtype=float)
        v_z_val = np.asarray(v_z.value, dtype=float).reshape(-1)
        p_z_val = np.asarray(p_z.value, dtype=float).reshape(-1)

        if not (
            np.isfinite(V_xy_val).all()
            and np.isfinite(P_xy_val).all()
            and np.isfinite(v_z_val).all()
            and np.isfinite(p_z_val).all()
        ):
            return {"position": [], "velocity": [], "thrust": []}

        V = np.zeros((K + 1, 3), dtype=float)
        P = np.zeros((K + 1, 3), dtype=float)
        V[:, :2] = V_xy_val
        V[:, 2] = v_z_val
        P[:, :2] = P_xy_val
        P[:, 2] = p_z_val

        # Fuel consumption
        fuel_consumption = float(gamma * np.sum(np.linalg.norm(F, axis=1)))

        # Final sanity checks: finiteness
        if not (np.isfinite(P).all() and np.isfinite(V).all()):
            return {"position": [], "velocity": [], "thrust": []}

        return {
            "position": P.tolist(),
            "velocity": V.tolist(),
            "thrust": F.tolist(),
            "fuel_consumption": fuel_consumption,
        }