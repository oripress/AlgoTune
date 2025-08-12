from typing import Any, Dict, Callable, Tuple
import numpy as np
import warnings

# Optional SciPy optimizers (may be missing in some environments)
try:
    from scipy.optimize import least_squares  # type: ignore
except Exception:
    least_squares = None

try:
    from scipy.optimize import leastsq  # type: ignore
except Exception:
    leastsq = None

def _safe_exp(z: np.ndarray | float) -> np.ndarray | float:
    """Exponentiation clipped to avoid overflow."""
    return np.exp(np.clip(z, -50.0, 50.0))

def _create_model(problem: Dict[str, Any]) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """
    Returns (model_fn, residual_fn, jac_fn, initial_guess).
    residual_fn returns y - model(p).
    jac_fn returns Jacobian of residuals (n_samples x n_params).
    """
    x = np.asarray(problem.get("x_data", []), dtype=float)
    y = np.asarray(problem.get("y_data", []), dtype=float)
    model_type = problem.get("model_type", "polynomial")

    if model_type == "polynomial":
        deg = int(problem.get("degree", 1))

        def model_fn(p: np.ndarray) -> np.ndarray:
            return np.polyval(p, x)

        def residual_fn(p: np.ndarray) -> np.ndarray:
            return y - np.polyval(p, x)

        def jac_fn(p: np.ndarray) -> np.ndarray:
            return -np.vander(x, N=p.size)

        guess = np.ones(deg + 1, dtype=float)
        return model_fn, residual_fn, jac_fn, guess

    elif model_type == "exponential":

        def model_fn(p: np.ndarray) -> np.ndarray:
            a, b, c = p
            return a * _safe_exp(b * x) + c

        def residual_fn(p: np.ndarray) -> np.ndarray:
            a, b, c = p
            return y - (a * _safe_exp(b * x) + c)

        def jac_fn(p: np.ndarray) -> np.ndarray:
            a, b, c = p
            E = _safe_exp(b * x)
            return np.column_stack((-E, -a * x * E, -np.ones_like(x)))

        # sensible data-driven initial guess
        a0 = float(np.nanmax(y) - np.nanmin(y)) if y.size else 1.0
        if a0 == 0.0:
            a0 = 1.0
        b0 = 0.05 if np.ptp(x) > 0 else 0.0
        c0 = float(np.nanmin(y)) if y.size else 0.0
        guess = np.array([a0, b0, c0], dtype=float)
        return model_fn, residual_fn, jac_fn, guess

    elif model_type == "logarithmic":

        def model_fn(p: np.ndarray) -> np.ndarray:
            a, b, c, d = p
            return a * np.log(np.clip(b * x + c, 1e-12, None)) + d

        def residual_fn(p: np.ndarray) -> np.ndarray:
            a, b, c, d = p
            return y - (a * np.log(np.clip(b * x + c, 1e-12, None)) + d)

        def jac_fn(p: np.ndarray) -> np.ndarray:
            a, b, c, d = p
            bx_c = np.clip(b * x + c, 1e-12, None)
            return np.column_stack((-np.log(bx_c), -a * x / bx_c, -a / bx_c, -np.ones_like(x)))

        a0 = float(np.nanmax(y) - np.nanmin(y)) if y.size else 1.0
        if a0 == 0.0:
            a0 = 1.0
        d0 = float(np.nanmin(y)) if y.size else 0.0
        guess = np.array([a0, 1.0, 1.0, d0], dtype=float)
        return model_fn, residual_fn, jac_fn, guess

    elif model_type == "sigmoid":

        def model_fn(p: np.ndarray) -> np.ndarray:
            a, b, c, d = p
            z = -b * (x - c)
            s = 1.0 / (1.0 + _safe_exp(z))
            return a * s + d

        def residual_fn(p: np.ndarray) -> np.ndarray:
            a, b, c, d = p
            z = -b * (x - c)
            s = 1.0 / (1.0 + _safe_exp(z))
            return y - (a * s + d)

        def jac_fn(p: np.ndarray) -> np.ndarray:
            a, b, c, d = p
            z = -b * (x - c)
            expz = _safe_exp(z)
            s = 1.0 / (1.0 + expz)
            # Correct derivatives of s = 1/(1+exp(z)) where z = -b*(x-c)
            # ds/db = (x - c) * s * (1 - s)
            ds_db = (x - c) * s * (1.0 - s)
            # ds/dc = -b * s * (1 - s)
            ds_dc = -b * s * (1.0 - s)
            return np.column_stack((-s, -a * ds_db, -a * ds_dc, -np.ones_like(x)))

        a0 = float(np.nanmax(y) - np.nanmin(y)) if y.size else 1.0
        if a0 == 0.0:
            a0 = 1.0
        c0 = float(np.median(x)) if x.size else 0.0
        b0 = 1.0 / (np.std(x) + 1e-6) if x.size > 1 else 0.5
        d0 = float(np.nanmin(y)) if y.size else 0.0
        guess = np.array([a0, b0, c0, d0], dtype=float)
        return model_fn, residual_fn, jac_fn, guess

    elif model_type == "sinusoidal":

        def model_fn(p: np.ndarray) -> np.ndarray:
            a, b, c, d = p
            return a * np.sin(b * x + c) + d

        def residual_fn(p: np.ndarray) -> np.ndarray:
            a, b, c, d = p
            return y - (a * np.sin(b * x + c) + d)

        def jac_fn(p: np.ndarray) -> np.ndarray:
            a, b, c, d = p
            arg = b * x + c
            return np.column_stack((-np.sin(arg), -a * x * np.cos(arg), -a * np.cos(arg), -np.ones_like(x)))

        # amplitude ~= half peak-to-peak, offset ~ mean, try FFT for frequency
        if x.size and y.size:
            amp0 = (np.nanmax(y) - np.nanmin(y)) / 2.0
            amp0 = amp0 if amp0 != 0 else 1.0
            offset0 = float(np.nanmean(y))
            dx = float(np.median(np.diff(x))) if x.size > 1 else 1.0
            b0 = 1.0
            if dx > 0 and x.size >= 4:
                try:
                    y0 = y - offset0
                    fft = np.fft.rfft(y0 - np.mean(y0))
                    freqs = np.fft.rfftfreq(x.size, d=dx)
                    if freqs.size > 1:
                        idx = int(np.argmax(np.abs(fft[1:])) + 1)
                        freq = float(freqs[idx])
                        btmp = 2.0 * np.pi * freq
                        if np.isfinite(btmp) and btmp != 0:
                            b0 = btmp
                except Exception:
                    b0 = 1.0
            guess = np.array([amp0, b0, 0.0, offset0], dtype=float)
        else:
            guess = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)
        return model_fn, residual_fn, jac_fn, guess

    else:
        raise ValueError(f"Unknown model type: {model_type}")

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        model_fn, residual_fn, jac_fn, guess = _create_model(problem)

        x = np.asarray(problem.get("x_data", []), dtype=float)
        y = np.asarray(problem.get("y_data", []), dtype=float)
        model_type = problem.get("model_type", "polynomial")

        # Fast return for empty data
        if x.size == 0 or y.size == 0:
            return {
                "params": guess.tolist(),
                "residuals": [],
                "mse": float("nan"),
                "convergence_info": {
                    "success": True,
                    "status": 1,
                    "message": "empty-data",
                    "num_function_calls": 0,
                    "final_cost": 0.0,
                },
            }

        params_opt = None
        message = ""
        status = 0
        nfev = 0

        if model_type == "polynomial":
            deg = int(problem.get("degree", max(1, guess.size - 1)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", np.RankWarning)
                p0 = np.asarray(guess, dtype=float)
                # Prefer scipy.optimize.leastsq to match the reference solver behavior
                params_opt = None
                if leastsq is not None:
                    try:
                        p_opt, cov_x, info_ls, mesg_ls, ier_ls = leastsq(residual_fn, p0, full_output=True, maxfev=10000)
                        params_opt = np.asarray(p_opt, dtype=float)
                        message = mesg_ls
                        try:
                            status = int(ier_ls)
                        except Exception:
                            status = 1 if getattr(ier_ls, None) else 0
                        try:
                            nfev = int(info_ls.get("nfev", 0)) if isinstance(info_ls, dict) else 0
                        except Exception:
                            nfev = 0
                    except Exception:
                        params_opt = None
                # Fallbacks: np.polyfit then np.linalg.lstsq
                if params_opt is None:
                    try:
                        params_opt = np.polyfit(x, y, deg).astype(float)
                        message = "polyfit"
                        status = 1
                        nfev = 0
                    except Exception:
                        try:
                            V = np.vander(x, N=deg + 1)
                            params_opt, *_ = np.linalg.lstsq(V, y, rcond=None)
                            params_opt = np.asarray(params_opt, dtype=float)
                            message = "lstsq"
                            status = 1
                            nfev = 0
                        except Exception as e:
                            params_opt = np.asarray(p0, dtype=float)
                            message = f"polyfit-failed: {e}"
                            status = 0
                            nfev = 0
            p0 = np.asarray(guess, dtype=float)
            # Try least_squares first (analytic jacobian speeds this up)
            if least_squares is not None:
                method = "lm" if x.size >= p0.size else "trf"
                try:
                    res = least_squares(lambda p: residual_fn(p), p0, jac=(lambda p: jac_fn(p)), method=method, max_nfev=10000)
                    params_opt = np.asarray(res.x, dtype=float)
                    message = getattr(res, "message", "")
                    try:
                        status = int(getattr(res, "status", 0))
                    except Exception:
                        status = 1 if getattr(res, "success", False) else 0
                    try:
                        nfev = int(getattr(res, "nfev", 0))
                    except Exception:
                        nfev = 0
                except Exception:
                    params_opt = None

            # Fallback to leastsq if needed
            if params_opt is None and leastsq is not None:
                try:
                    p_opt, cov_x, info_ls, mesg_ls, ier_ls = leastsq(residual_fn, p0, full_output=True, maxfev=10000)
                    params_opt = np.asarray(p_opt, dtype=float)
                    message = mesg_ls
                    try:
                        status = int(ier_ls)
                    except Exception:
                        status = int(ier_ls) if isinstance(ier_ls, int) else 0
                    try:
                        nfev = int(info_ls.get("nfev", 0)) if isinstance(info_ls, dict) else 0
                    except Exception:
                        nfev = 0
                except Exception as e:
                    params_opt = p0.copy()
                    message = f"leastsq-failed: {e}"
                    status = 0
                    nfev = 0

            if params_opt is None:
                params_opt = p0.copy()
                if message == "":
                    message = "no-optimizer-available"
                status = 0
                nfev = 0

        # Evaluate fitted model safely
        try:
            y_fit = np.asarray(model_fn(params_opt), dtype=float)
        except Exception:
            # fallback to initial guess if evaluation fails
            _, residual_fn2, jac_fn2, guess2 = _create_model(problem)
            params_opt = np.asarray(guess2, dtype=float)
            y_fit = np.asarray(model_fn(params_opt), dtype=float)

        residuals = (y - y_fit).astype(float)
        mse = float(np.mean(residuals ** 2)) if residuals.size else float("nan")
        final_cost = float(np.sum(residuals ** 2))

        success_flag = bool(status in {1, 2, 3, 4})

        return {
            "params": np.asarray(params_opt, dtype=float).tolist(),
            "residuals": residuals.tolist(),
            "mse": mse,
            "convergence_info": {
                "success": bool(success_flag),
                "status": int(status) if isinstance(status, (int, np.integer)) else int(status) if isinstance(status, int) else 0,
                "message": str(message),
                "num_function_calls": int(nfev),
                "final_cost": final_cost,
            },
        }