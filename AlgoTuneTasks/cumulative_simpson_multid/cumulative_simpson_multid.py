import logging

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_simpson

from AlgoTuneTasks.base import register_task, Task


@register_task("cumulative_simpson_multid")
class CumulativeSimpsonMultiD(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_problem(self, n: int = 1000, random_seed: int = 1) -> dict:
        """
        Generate a multi-dimensional problem for cumulative Simpson integration.
        Creates a 1D array of n points sampled from sin(2πx) over [0, 5],
        and tiles this array to form a (100, 100, n) array.
        Returns this multi-dimensional array along with the spacing (dx).
        """
        logging.debug(
            f"Generating multi-dimensional cumulative Simpson problem with n={n} and random_seed={random_seed}."
        )
        x, dx = np.linspace(0, 5, n, retstep=True)
        y = np.sin(2 * np.pi * x)
        y2 = np.tile(y, (100, 100, 1))
        return {"y2": y2, "dx": dx}

    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral along the last axis of the multi-dimensional array using Simpson's rule.
        """
        y2 = problem["y2"]
        dx = problem["dx"]
        result = cumulative_simpson(y2, dx=dx)
        return result

    def is_solution(self, problem: dict, solution: NDArray) -> bool:
        """
        Check if the multi-dimensional cumulative Simpson solution is valid and optimal.

        A valid solution must match the reference implementation (scipy's cumulative_simpson)
        within a small tolerance.

        :param problem: A dictionary containing the multi-dimensional input array and dx.
        :param solution: The computed cumulative integral.
        :return: True if the solution is valid and optimal, False otherwise.
        """
        y2 = problem["y2"]
        dx = problem["dx"]
        reference = cumulative_simpson(y2, dx=dx)
        tol = 1e-6
        error = np.linalg.norm(solution - reference) / (np.linalg.norm(reference) + 1e-12)
        if error > tol:
            logging.error(
                f"Cumulative Simpson MultiD relative error {error} exceeds tolerance {tol}."
            )
            return False
        return True
