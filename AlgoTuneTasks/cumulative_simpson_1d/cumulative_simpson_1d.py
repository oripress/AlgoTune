import logging

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_simpson

from AlgoTuneTasks.base import register_task, Task


@register_task("cumulative_simpson_1d")
class CumulativeSimpson1D(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_problem(self, n: int = 1000, random_seed: int = 1) -> dict:
        """
        Generate a 1D problem for cumulative Simpson integration.
        Creates an array of n points sampled from sin(2πx) over the interval [0, 5],
        and returns the array along with the spacing (dx) between points.
        """
        logging.debug(
            f"Generating 1D cumulative Simpson problem with n={n} and random_seed={random_seed}."
        )
        x, dx = np.linspace(0, 5, n, retstep=True)
        y = np.sin(2 * np.pi * x)
        return {"y": y, "dx": dx}

    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral of the 1D array using Simpson's rule.
        """
        y = problem["y"]
        dx = problem["dx"]
        result = cumulative_simpson(y, dx=dx)
        return result

    def is_solution(self, problem: dict, solution: NDArray) -> bool:
        """
        Check if the cumulative Simpson solution is valid and optimal.

        A valid solution must match the reference implementation (scipy's cumulative_simpson)
        within a small tolerance.

        :param problem: A dictionary containing the input array and dx.
        :param solution: The computed cumulative integral.
        :return: True if the solution is valid and optimal, False otherwise.
        """
        y = problem["y"]
        dx = problem["dx"]
        reference = cumulative_simpson(y, dx=dx)
        tol = 1e-6
        error = np.linalg.norm(solution - reference) / (np.linalg.norm(reference) + 1e-12)
        if error > tol:
            logging.error(f"Cumulative Simpson 1D relative error {error} exceeds tolerance {tol}.")
            return False
        return True
