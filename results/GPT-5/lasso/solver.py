from __future__ import annotations

from typing import Any, List

import numpy as np
from sklearn import linear_model

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[float]:
        """
        Solve Lasso:
            min_w (1/(2n))||y - Xw||_2^2 + alpha * ||w||_1

        Uses the same alpha and config as the reference for validity, with
        small optimizations:
          - early screening for zero solution
          - univariate closed form
          - efficient numpy conversion and memory layout
          - Gram precompute when beneficial
        """
        try:
            X_in = problem["X"]
            y_in = problem["y"]

            # Convert inputs efficiently
            X = np.asfortranarray(np.asarray(X_in, dtype=np.float64))
            y = np.asarray(y_in, dtype=np.float64).ravel()

            if X.ndim != 2 or y.ndim != 1:
                return []
            n, d = X.shape
            if d == 0:
                return []
            if n == 0:
                return [0.0] * d
            if y.shape[0] != n:
                if y.shape[0] > n:
                    y = y[:n]
                else:
                    y = np.pad(y, (0, n - y.shape[0]), mode="constant")

            # Fixed alpha as per task/reference
            alpha = 0.1

            # Early screening: if alpha >= max |(1/n) X^T y| then w=0 is optimal
            xty = X.T @ y
            lam_max = float(np.max(np.abs(xty))) / n
            if alpha >= lam_max:
                return [0.0] * d

            # Univariate closed form
            if d == 1:
                x = X[:, 0]
                s = float(x @ x) / n
                if s == 0.0:
                    return [0.0]
                rho = float(x @ y) / n
                if rho > alpha:
                    w0 = (rho - alpha) / s
                elif rho < -alpha:
                    w0 = (rho + alpha) / s
                else:
                    w0 = 0.0
                return [float(w0)]

            # Use Gram precompute when n >= d and d is modest
            precompute = True if (n >= d and d <= 256) else False

            # Fall back to sklearn Lasso with same settings as reference
            clf = linear_model.Lasso(
                alpha=alpha,
                fit_intercept=False,
                copy_X=False,
                precompute=precompute,
                selection="cyclic",
                max_iter=1000,
                tol=1e-4,
            )
            clf.fit(X, y)
            return clf.coef_.astype(float).tolist()
        except Exception:
            try:
                d = int(np.asarray(problem.get("X", [])).shape[1])
            except Exception:
                d = 0
            return [0.0] * d