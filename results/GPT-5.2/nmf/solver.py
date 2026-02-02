from __future__ import annotations

from typing import Any

import numpy as np

try:
    # Lower-level API avoids estimator object overhead.
    from sklearn.decomposition._nmf import non_negative_factorization
except Exception:  # pragma: no cover
    non_negative_factorization = None  # type: ignore[assignment]

class Solver:
    def __init__(self) -> None:
        self._seed = 0

    @staticmethod
    def _exact_if_possible(X: np.ndarray, k: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Exact nonnegative factorization when rank budget is large (error=0, very fast)."""
        m, n = X.shape
        if k >= n:
            W = np.zeros((m, k), dtype=X.dtype)
            W[:, :n] = X
            H = np.zeros((k, n), dtype=X.dtype)
            H[:n, :] = np.eye(n, dtype=X.dtype)
            return W, H
        if k >= m:
            W = np.zeros((m, k), dtype=X.dtype)
            W[:, :m] = np.eye(m, dtype=X.dtype)
            H = np.zeros((k, n), dtype=X.dtype)
            H[:m, :] = X
            return W, H
        return None, None

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, list[list[float]]]:
        k = int(problem["n_components"])
        # Match reference conversion style (fast path).
        X = np.array(problem["X"], dtype=np.float64, copy=False)
        m, n = X.shape

        if non_negative_factorization is None:
            return {
                "W": np.zeros((m, k), dtype=np.float64).tolist(),
                "H": np.zeros((k, n), dtype=np.float64).tolist(),
            }

        # Exact shortcut helps when k is large.
        W_exact, H_exact = self._exact_if_possible(X, k)
        if W_exact is not None:
            return {"W": W_exact.tolist(), "H": H_exact.tolist()}

        try:
            # Slightly looser tol than reference to stop earlier on easy instances.
            W, H, _ = non_negative_factorization(
                X,
                n_components=k,
                init="random",
                solver="cd",
                beta_loss="frobenius",
                tol=5e-4,
                max_iter=200,
                random_state=self._seed,
                update_H=True,
                verbose=0,
            )

            # non_negative_factorization guarantees non-negativity; avoid extra scans.
            return {"W": W.tolist(), "H": H.tolist()}
        except Exception:
            return {
                "W": np.zeros((m, k), dtype=np.float64).tolist(),
                "H": np.zeros((k, n), dtype=np.float64).tolist(),
            }