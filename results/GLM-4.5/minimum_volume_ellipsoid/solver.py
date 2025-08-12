from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, Any]:
        """
        Solves a given minimum volume covering ellipsoid problem using CVXPY.

        Args:
            problem: A dictionary with problem parameter:
                - points: list of given points to be contained in the ellipsoid.

        Returns:
            A dictionary containing the problem solution:
                - objective_value: the optimal objective value, which is proportional to logarithm of ellipsoid volume,
                - ellipsoid: a dictionary containing symmetric matrix X and ellipsoid center Y.
        """

        points = np.array(problem["points"])
        (n, d) = points.shape

        X = cp.Variable((d, d), symmetric=True)
        Y = cp.Variable((d,))

        constraint = []
        for i in range(n):
            constraint += [cp.SOC(1, X @ points[i] + Y)]

        problem_cvxpy = cp.Problem(cp.Minimize(-cp.log_det(X)), constraint)

        try:
            problem_cvxpy.solve(solver=cp.CLARABEL, verbose=False)

            # Check if a solution was found
            if problem_cvxpy.status not in ["optimal", "optimal_inaccurate"]:
                return {
                    "objective_value": float("inf"),
                    "ellipsoid": {"X": np.nan * np.ones((d, d)), "Y": np.nan * np.ones((d,))},
                }

            return {"objective_value": problem_cvxpy.value, "ellipsoid": {"X": X.value, "Y": Y.value}}

        except Exception as e:
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.nan * np.ones((d, d)), "Y": np.nan * np.ones((d,))},
            }