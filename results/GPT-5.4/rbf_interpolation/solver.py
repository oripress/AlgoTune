from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import RBFInterpolator

_HAVE_PRIVATE_RBF = False
try:
    from scipy.interpolate._rbfinterp import (  # type: ignore
        _build_and_solve_system,
        _build_evaluation_coefficients,
        _monomial_powers,
    )

    _HAVE_PRIVATE_RBF = True
except Exception:
    try:
        from scipy.interpolate._rbfinterp import _monomial_powers  # type: ignore
        from scipy.interpolate._rbfinterp_pythran import (  # type: ignore
            _build_evaluation_coefficients,
        )

        try:
            from scipy.interpolate._rbfinterp import _build_and_solve_system  # type: ignore
        except Exception:
            from scipy.interpolate._rbfinterp_pythran import _build_system  # type: ignore
            from scipy.linalg.lapack import dgesv

            def _build_and_solve_system(y, d, smoothing, kernel, epsilon, powers):
                lhs, rhs, shift, scale = _build_system(
                    y, d, smoothing, kernel, epsilon, powers
                )
                _, _, coeffs, info = dgesv(lhs, rhs, overwrite_a=True, overwrite_b=True)
                if info != 0:
                    raise np.linalg.LinAlgError("RBF system solve failed")
                return shift, scale, coeffs

        _HAVE_PRIVATE_RBF = True
    except Exception:
        _HAVE_PRIVATE_RBF = False

_DEFAULT_DEGREE = {
    "multiquadric": 0,
    "linear": 0,
    "thin_plate_spline": 1,
    "cubic": 1,
    "quintic": 2,
}

_POWERS_CACHE: dict[tuple[int, int], np.ndarray] = {}

def _degree_for_kernel(kernel: str) -> int:
    return _DEFAULT_DEGREE.get(kernel, 0)

def _get_powers(ndim: int, degree: int) -> np.ndarray:
    key = (ndim, degree)
    powers = _POWERS_CACHE.get(key)
    if powers is None:
        powers = _monomial_powers(ndim, degree)
        _POWERS_CACHE[key] = powers
    return powers

def _pairwise_r2(xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    aa = np.einsum("ij,ij->i", xa, xa)
    bb = np.einsum("ij,ij->i", xb, xb)
    r2 = aa[:, None] + bb[None, :] - 2.0 * (xa @ xb.T)
    np.maximum(r2, 0.0, out=r2)
    return r2

def _kernel_matrix(r2: np.ndarray, kernel: str, epsilon: float) -> np.ndarray:
    if kernel == "gaussian":
        return np.exp(-r2 / (epsilon * epsilon))
    if kernel == "inverse_multiquadric":
        return 1.0 / np.sqrt(r2 + epsilon * epsilon)
    if kernel == "multiquadric":
        return np.sqrt(r2 + epsilon * epsilon)

    r = np.sqrt(r2)
    if kernel == "linear":
        return r
    if kernel == "cubic":
        return r2 * r
    if kernel == "quintic":
        return (r2 * r2) * r
    if kernel == "thin_plate_spline":
        out = np.zeros_like(r2)
        mask = r2 > 0.0
        out[mask] = 0.5 * r2[mask] * np.log(r2[mask])
        return out

    return np.exp(-r2 / (epsilon * epsilon))

def _polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    n, d = x.shape
    if degree < 0:
        return np.empty((n, 0), dtype=x.dtype)

    parts = [np.ones((n, 1), dtype=x.dtype)]
    if degree >= 1:
        parts.append(x)

    if degree >= 2:
        cols = []
        for i in range(d):
            xi = x[:, i : i + 1]
            cols.append(xi * xi)
            for j in range(i + 1, d):
                cols.append(xi * x[:, j : j + 1])
        if cols:
            parts.append(np.hstack(cols))

    if len(parts) == 1:
        return parts[0]
    return np.hstack(parts)

def _solve_fallback(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    kernel: str,
    epsilon: float,
    smoothing: float,
) -> np.ndarray:
    n = x_train.shape[0]
    degree = _degree_for_kernel(kernel)
    p_train = _polynomial_features(x_train, degree)
    p = p_train.shape[1]

    k = _kernel_matrix(_pairwise_r2(x_train, x_train), kernel, epsilon)
    if smoothing != 0.0:
        k = k.copy()
        k.flat[:: n + 1] += smoothing

    if p:
        a = np.empty((n + p, n + p), dtype=np.float64)
        a[:n, :n] = k
        a[:n, n:] = p_train
        a[n:, :n] = p_train.T
        a[n:, n:] = 0.0

        b = np.empty(n + p, dtype=np.float64)
        b[:n] = y_train
        b[n:] = 0.0
    else:
        a = k
        b = y_train

    try:
        coeffs = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(a, b, rcond=None)[0]

    weights = coeffs[:n]
    y_pred = _kernel_matrix(_pairwise_r2(x_test, x_train), kernel, epsilon) @ weights
    if p:
        y_pred += _polynomial_features(x_test, degree) @ coeffs[n:]
    return y_pred

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        x_train = np.asarray(problem["x_train"], dtype=np.float64)
        y_train = np.asarray(problem["y_train"], dtype=np.float64).reshape(-1)
        x_test = np.asarray(problem["x_test"], dtype=np.float64)

        if x_train.ndim == 1:
            x_train = x_train[:, None]
        if x_test.ndim == 1:
            x_test = x_test[:, None]

        rbf_config = problem.get("rbf_config") or {}
        kernel = rbf_config.get("kernel", "thin_plate_spline")
        epsilon = float(rbf_config.get("epsilon", 1.0))
        smoothing = float(rbf_config.get("smoothing", 0.0))

        if _HAVE_PRIVATE_RBF:
            try:
                degree = _degree_for_kernel(kernel)
                powers = _get_powers(x_train.shape[1], degree)
                smoothing_vec = np.full(x_train.shape[0], smoothing, dtype=np.float64)
                shift, scale, coeffs = _build_and_solve_system(
                    x_train, y_train[:, None], smoothing_vec, kernel, epsilon, powers
                )

                n_coeff = coeffs.shape[0]
                n_test = x_test.shape[0]
                y_pred = np.empty(n_test, dtype=np.float64)

                max_elements = 8_000_000
                chunk = max(1, min(n_test, max_elements // max(n_coeff, 1)))
                start = 0
                while start < n_test:
                    end = min(start + chunk, n_test)
                    vec = _build_evaluation_coefficients(
                        x_test[start:end],
                        x_train,
                        kernel,
                        epsilon,
                        powers,
                        shift,
                        scale,
                    )
                    y_pred[start:end] = (vec @ coeffs).ravel()
                    start = end

                return {"y_pred": y_pred}
            except Exception:
                pass

        try:
            y_pred = RBFInterpolator(
                x_train,
                y_train,
                kernel=kernel,
                epsilon=epsilon,
                smoothing=smoothing,
            )(x_test)
            return {"y_pred": np.asarray(y_pred, dtype=np.float64).reshape(-1)}
        except Exception:
            pass

        y_pred = _solve_fallback(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            kernel=kernel,
            epsilon=epsilon,
            smoothing=smoothing,
        )
        return {"y_pred": y_pred}

        y_pred = _solve_fallback(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            kernel=kernel,
            epsilon=epsilon,
            smoothing=smoothing,
        )
        return {"y_pred": y_pred.tolist()}