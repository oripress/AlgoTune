from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import linear_model

def _soft_threshold(rho: float, alpha: float) -> float:
    if rho > alpha:
        return rho - alpha
    if rho < -alpha:
        return rho + alpha
    return 0.0

class Solver:
    __slots__ = (
        "alpha",
        "tol",
        "max_iter",
        "gram_max_d",
        "_cd",
        "_cdg",
        "_lasso",
        "_lasso_pre",
    )

    def __init__(self) -> None:
        self.alpha = 0.1
        self.tol = 1e-4
        self.max_iter = 1000
        self.gram_max_d = 64
        self._cd = None
        self._cdg = None
        self._lasso = linear_model.Lasso(
            alpha=0.1,
            fit_intercept=False,
            copy_X=False,
        )
        self._lasso_pre = linear_model.Lasso(
            alpha=0.1,
            fit_intercept=False,
            copy_X=False,
            precompute=True,
        )

        try:
            from sklearn.linear_model._cd_fast import (
                enet_coordinate_descent,
                enet_coordinate_descent_gram,
            )

            coef = np.zeros(1, dtype=np.float64)
            X = np.zeros((1, 1), dtype=np.float64, order="F")
            y = np.zeros(1, dtype=np.float64)
            gram = np.zeros((1, 1), dtype=np.float64)
            Xy = np.zeros(1, dtype=np.float64)

            try:
                enet_coordinate_descent(coef.copy(), 1.0, 0.0, X, y, 1, 1e-4, None, False, False)

                def _cd(
                    coef_: np.ndarray,
                    l1_reg_: float,
                    X_: np.ndarray,
                    y_: np.ndarray,
                    max_iter_: int,
                    tol_: float,
                ) -> np.ndarray:
                    out = enet_coordinate_descent(
                        coef_,
                        l1_reg_,
                        0.0,
                        X_,
                        y_,
                        max_iter_,
                        tol_,
                        None,
                        False,
                        False,
                    )
                    return out[0] if isinstance(out, tuple) else out

            except TypeError:
                try:
                    enet_coordinate_descent(coef.copy(), 1.0, 0.0, X, y, 1, 1e-4, None, False)

                    def _cd(
                        coef_: np.ndarray,
                        l1_reg_: float,
                        X_: np.ndarray,
                        y_: np.ndarray,
                        max_iter_: int,
                        tol_: float,
                    ) -> np.ndarray:
                        out = enet_coordinate_descent(
                            coef_,
                            l1_reg_,
                            0.0,
                            X_,
                            y_,
                            max_iter_,
                            tol_,
                            None,
                            False,
                        )
                        return out[0] if isinstance(out, tuple) else out

                except TypeError:

                    def _cd(
                        coef_: np.ndarray,
                        l1_reg_: float,
                        X_: np.ndarray,
                        y_: np.ndarray,
                        max_iter_: int,
                        tol_: float,
                    ) -> np.ndarray:
                        out = enet_coordinate_descent(
                            coef_,
                            l1_reg_,
                            0.0,
                            X_,
                            y_,
                            max_iter_,
                            tol_,
                            None,
                        )
                        return out[0] if isinstance(out, tuple) else out

            self._cd = _cd

            try:
                enet_coordinate_descent_gram(
                    coef.copy(),
                    1.0,
                    0.0,
                    gram,
                    Xy,
                    y,
                    1,
                    1e-4,
                    None,
                    False,
                    False,
                )

                def _cdg(
                    coef_: np.ndarray,
                    l1_reg_: float,
                    gram_: np.ndarray,
                    Xy_: np.ndarray,
                    y_: np.ndarray,
                    max_iter_: int,
                    tol_: float,
                ) -> np.ndarray:
                    out = enet_coordinate_descent_gram(
                        coef_,
                        l1_reg_,
                        0.0,
                        gram_,
                        Xy_,
                        y_,
                        max_iter_,
                        tol_,
                        None,
                        False,
                        False,
                    )
                    return out[0] if isinstance(out, tuple) else out

            except TypeError:
                try:
                    enet_coordinate_descent_gram(
                        coef.copy(),
                        1.0,
                        0.0,
                        gram,
                        Xy,
                        y,
                        1,
                        1e-4,
                        None,
                        False,
                    )

                    def _cdg(
                        coef_: np.ndarray,
                        l1_reg_: float,
                        gram_: np.ndarray,
                        Xy_: np.ndarray,
                        y_: np.ndarray,
                        max_iter_: int,
                        tol_: float,
                    ) -> np.ndarray:
                        out = enet_coordinate_descent_gram(
                            coef_,
                            l1_reg_,
                            0.0,
                            gram_,
                            Xy_,
                            y_,
                            max_iter_,
                            tol_,
                            None,
                            False,
                        )
                        return out[0] if isinstance(out, tuple) else out

                except TypeError:

                    def _cdg(
                        coef_: np.ndarray,
                        l1_reg_: float,
                        gram_: np.ndarray,
                        Xy_: np.ndarray,
                        y_: np.ndarray,
                        max_iter_: int,
                        tol_: float,
                    ) -> np.ndarray:
                        out = enet_coordinate_descent_gram(
                            coef_,
                            l1_reg_,
                            0.0,
                            gram_,
                            Xy_,
                            y_,
                            max_iter_,
                            tol_,
                            None,
                        )
                        return out[0] if isinstance(out, tuple) else out

            self._cdg = _cdg
        except Exception:
            pass

    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        try:
            X = np.asarray(problem["X"], dtype=np.float64)
            if X.ndim != 2:
                return []
            n, d = X.shape
            if d == 0:
                return []
            if n == 0:
                return [0.0] * d

            y = np.asarray(problem["y"], dtype=np.float64).reshape(-1)
            if not np.any(y):
                return [0.0] * d

            alpha = self.alpha

            if d == 1:
                x = X[:, 0]
                z = float(np.dot(x, x)) / n
                if z <= 0.0:
                    return [0.0]
                rho = float(np.dot(x, y)) / n
                return [_soft_threshold(rho, alpha) / z]

            if n == 1:
                row = X[0]
                j = int(np.argmax(np.abs(row)))
                xj = float(row[j])
                axj = xj if xj >= 0.0 else -xj
                if axj <= 0.0:
                    return [0.0] * d
                coef = np.zeros(d, dtype=np.float64)
                coef[j] = _soft_threshold(float(y[0]), alpha / axj) / xj
                return coef.tolist()

            coef = np.zeros(d, dtype=np.float64)
            l1_reg = alpha * n

            try:
                cdg = self._cdg
                if cdg is not None and d <= self.gram_max_d and n >= d:
                    gram = X.T @ X
                    Xy = X.T @ y
                    coef = cdg(coef, l1_reg, gram, Xy, y, self.max_iter, self.tol)
                    return coef.tolist()

                cd = self._cd
                if cd is not None:
                    Xf = X if X.flags.f_contiguous else np.asfortranarray(X)
                    coef = cd(coef, l1_reg, Xf, y, self.max_iter, self.tol)
                    return coef.tolist()
            except Exception as e:
                print("LOW_LEVEL_ERR", repr(e))
                pass

            if d <= self.gram_max_d and n >= d:
                self._lasso_pre.fit(X, y)
                return self._lasso_pre.coef_.tolist()
            self._lasso.fit(X, y)
            return self._lasso.coef_.tolist()
        except Exception:
            try:
                X = problem["X"]
                try:
                    d = X.shape[1]
                except Exception:
                    d = len(X[0]) if X else 0
                return np.zeros(d).tolist()
            except Exception:
                return []