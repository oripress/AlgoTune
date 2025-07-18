from typing import Any, Dict, Tuple
import cvxpy as cp
import numpy as np

class Solver:
    def __init__(self):
        """
        Initializes the solver, setting up a cache for compiled problems.
        The cache will store CVXPY problems keyed by the number of time steps K.
        """
        self._cached_problems: Dict[int, Tuple[cp.Problem, Dict[str, cp.Parameter], Dict[str, cp.Variable]]] = {}

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the rocket landing optimization problem using CVXPY with a caching strategy.

        For each unique value of K (number of time steps), a CVXPY problem is constructed
        once using Parameters and then cached. Subsequent solves for the same K are much
        faster as they only require updating parameter values, avoiding repeated problem
        construction and canonicalization.
        """
        K = int(problem["K"])

        if K not in self._cached_problems:
            # This is the first time we see this value of K.
            # We construct the problem symbolically and cache it.
            p0 = cp.Parameter(3, name='p0')
            v0 = cp.Parameter(3, name='v0')
            p_target = cp.Parameter(3, name='p_target')
            g = cp.Parameter(nonneg=True, name='g')
            m = cp.Parameter(nonneg=True, name='m')
            h = cp.Parameter(nonneg=True, name='h')
            F_max = cp.Parameter(nonneg=True, name='F_max')
            gamma = cp.Parameter(nonneg=True, name='gamma')

            params = {
                'p0': p0, 'v0': v0, 'p_target': p_target, 'g': g, 'm': m, 
                'h': h, 'F_max': F_max, 'gamma': gamma
            }

            V = cp.Variable((K + 1, 3), name='V')
            P = cp.Variable((K + 1, 3), name='P')
            F = cp.Variable((K, 3), name='F')
            
            vars = {'P': P, 'V': V, 'F': F}

            constraints = [
                V[0] == v0,
                P[0] == p0,
                V[K] == np.zeros(3),
                P[K] == p_target,
                P[:, 2] >= 0,
                V[1:, :2] == V[:-1, :2] + h * (F[:, :2] / m),
                V[1:, 2] == V[:-1, 2] + h * (F[:, 2] / m - g),
                P[1:] == P[:-1] + h / 2 * (V[:-1] + V[1:]),
                cp.norm(F, 2, axis=1) <= F_max
            ]

            fuel_consumption = gamma * cp.sum(cp.norm(F, axis=1))
            objective = cp.Minimize(fuel_consumption)
            
            prob = cp.Problem(objective, constraints)
            
            self._cached_problems[K] = (prob, params, vars)
        
        # Retrieve the cached problem.
        prob, params, vars = self._cached_problems[K]

        # Update the parameter values with the current problem's data.
        params['p0'].value = np.array(problem["p0"])
        params['v0'].value = np.array(problem["v0"])
        params['p_target'].value = np.array(problem["p_target"])
        params['g'].value = float(problem["g"])
        params['m'].value = float(problem["m"])
        params['h'].value = float(problem["h"])
        params['F_max'].value = float(problem["F_max"])
        params['gamma'].value = float(problem["gamma"])

        try:
            # Use the ECOS solver with warm start and tuned tolerances for reliability and speed.
            prob.solve(solver=cp.ECOS, warm_start=True, abstol=1e-4, reltol=1e-4, feastol=1e-6, verbose=False)
        except (cp.SolverError, Exception):
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": 0.0}
        P, V, F = vars['P'], vars['V'], vars['F']
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or P.value is None:
            return {"position": [], "velocity": [], "thrust": [], "fuel_consumption": 0.0}

        # Return the solution in the required format.
        return {
            "position": P.value.tolist(),
            "velocity": V.value.tolist(),
            "thrust": F.value.tolist(),
            "fuel_consumption": float(prob.value),
        }