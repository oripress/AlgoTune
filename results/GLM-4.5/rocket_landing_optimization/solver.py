from typing import Any
import cvxpy as cp
import numpy as np
import pyximport
pyximport.install()

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the rocket landing optimization problem using CVXPY.

        :param problem: Dictionary with problem parameters
        :return: Dictionary with position, velocity, and thrust trajectories
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        # Extract problem parameters
        p0 = np.array(problem["p0"])
        v0 = np.array(problem["v0"])
        p_target = np.array(problem["p_target"])
        g = float(problem["g"])
        m = float(problem["m"])
        h = float(problem["h"])
        K = int(problem["K"])
        F_max = float(problem["F_max"])
        gamma = float(problem["gamma"])
        
        # Variables - use more efficient naming and structure
        V = cp.Variable((K + 1, 3))  # Velocity
        P = cp.Variable((K + 1, 3))  # Position
        F = cp.Variable((K, 3))  # Thrust

        # Pre-compute constants for efficiency
        h_over_2 = h / 2
        h_over_m = h / m
        g_h = g * h

        # Constraints
        constraints = []

        # Initial conditions
        constraints.append(V[0] == v0)
        constraints.append(P[0] == p0)

        # Terminal conditions
        constraints.append(V[K] == 0)  # Zero final velocity
        constraints.append(P[K] == p_target)  # Target position

        # Height constraint (always positive)
        constraints.append(P[:, 2] >= 0)

        # Dynamics for velocity - more efficient formulation
        constraints.append(V[1:, :2] == V[:-1, :2] + h_over_m * F[:, :2])
        constraints.append(V[1:, 2] == V[:-1, 2] + h_over_m * F[:, 2] - g_h)

        # Dynamics for position - more efficient formulation
        constraints.append(P[1:] == P[:-1] + h_over_2 * (V[:-1] + V[1:]))

        # Maximum thrust constraint
        constraints.append(cp.norm(F, 2, axis=1) <= F_max)

        # Objective: minimize fuel consumption
        fuel_consumption = gamma * cp.sum(cp.norm(F, axis=1))
        objective = cp.Minimize(fuel_consumption)

        # Solve the problem with optimized settings
        prob = cp.Problem(objective, constraints)
        try:
            # Try different solvers in order of expected speed
            for solver_name, solver_kwargs in [
                (cp.OSQP, {"max_iter": 1500, "eps_abs": 1e-5, "eps_rel": 1e-5, "polish": True}),
                (cp.ECOS, {"max_iters": 500, "feastol": 1e-5, "reltol": 1e-5}),
                (cp.SCS, {"max_iters": 1000, "eps": 1e-4})
            ]:
                try:
                    prob.solve(verbose=False, solver=solver_name, **solver_kwargs)
                    if prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                        break
                except:
                    continue
            
            # If all solvers failed, try default
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                prob.solve(verbose=False)
        except Exception:
            return {"position": [], "velocity": [], "thrust": []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or P.value is None:
            return {"position": [], "velocity": [], "thrust": []}
        
        # Return solution
        return {
            "position": P.value.tolist(),
            "velocity": V.value.tolist(),
            "thrust": F.value.tolist(),
            "fuel_consumption": float(prob.value),
        }