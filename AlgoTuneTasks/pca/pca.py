# Copyright (c) 2025 Ori Press and the AlgoTune contributors
# https://github.com/oripress/AlgoTune
import logging
from typing import Any

import numpy as np
import sklearn

from AlgoTuneTasks.base import register_task, Task


@register_task("pca")
class PCA(Task):
    def __init__(self, **kwargs):
        """
        Initialize the Principal component analysis (PCA) Task.

        Finds the components V given the data matrix X. Uses
        `sklearn.decomposition.PCA`.
        """
        super().__init__(**kwargs)

    def generate_problem(self, n: int, random_seed: int = 1) -> dict[str, Any]:
        """
        Generate random data matrix using n to control the hardness
        """
        np.random.seed(random_seed)
        # 50 * n samples
        m = 50 * n

        r = max(2, n * 5)  # factorization rank
        # Step 1: Generate non-negative W and H
        W = np.random.rand(m, r)  # m x r
        H = np.random.rand(r, 10 * n)  # r x (10 n)

        # Step 2: Generate Y = W H + small noise
        Y = W @ H
        noise_level = 0.01

        Y += noise_level * np.random.rand(
            m, 10 * n
        )  # additive small noise to simulate imperfection

        return dict(X=Y.tolist(), n_components=r)

    def solve(self, problem: dict[str, Any]) -> list[list[float]]:
        try:
            # use sklearn.decomposition.PCA to solve the task
            model = sklearn.decomposition.PCA(n_components=problem["n_components"])
            X = np.array(problem["X"])
            X = X - np.mean(X, axis=0)
            model.fit(X)
            V = model.components_
            return V
        except Exception as e:
            logging.error(f"Error: {e}")
            n_components = problem["n_components"]
            n, d = np.array(problem["X"]).shape
            V = np.zeros((n_components, n))
            id = np.eye(n_components)
            V[:, :n_components] = id
            return V  # return trivial answer

    def is_solution(self, problem: dict[str, Any], solution: list[list[float]]) -> bool:
        try:
            n_components = problem["n_components"]
            V = np.array(solution, dtype=float)
            X = np.array(problem["X"], dtype=float)
            X = X - np.mean(X, axis=0)

            r, n = V.shape
            # make sure that the number of components is satisfied
            if n_components != r:
                return False
            # check shape
            if n != X.shape[1]:
                return False

            tol = 1e-4
            # check if the matrix V is orthonormal
            VVT = V @ V.T
            if not np.allclose(VVT, np.eye(n_components), rtol=tol, atol=tol / 10):
                return False

            # Match sklearn's ordered principal components up to per-row sign flips.
            # A rotated basis for the same top-k subspace is not equivalent to the
            # reference PCA output, even if it preserves the aggregate objective.
            V_solver = np.array(self.solve(problem), dtype=float)
            if V_solver.shape != V.shape:
                return False

            component_alignment = np.abs(V @ V_solver.T)
            return np.allclose(
                component_alignment,
                np.eye(n_components),
                rtol=tol,
                atol=tol / 10,
            )

        except Exception as e:
            logging.error(f"Error when verifying solution: {e}")
            return False
