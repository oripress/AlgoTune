from typing import Any
import numpy as np

# Try to use numba for a fast native integrator. Fallback to scipy if numba isn't available.
_try_numba = True
try:
    import numba
    from numba import njit
except Exception:
    _try_numba = False
    njit = lambda *a, **k: (lambda f: f)  # identity decorator if numba missing

# Dormand-Prince coefficients (float constants)
_c2 = 1.0 / 5.0
_c3 = 3.0 / 10.0
_c4 = 4.0 / 5.0
_c5 = 8.0 / 9.0
_c6 = 1.0
_c7 = 1.0

_a21 = 1.0 / 5.0

_a31 = 3.0 / 40.0
_a32 = 9.0 / 40.0

_a41 = 44.0 / 45.0
_a42 = -56.0 / 15.0
_a43 = 32.0 / 9.0

_a51 = 19372.0 / 6561.0
_a52 = -25360.0 / 2187.0
_a53 = 64448.0 / 6561.0
_a54 = -212.0 / 729.0

_a61 = 9017.0 / 3168.0
_a62 = -355.0 / 33.0
_a63 = 46732.0 / 5247.0
_a64 = 49.0 / 176.0
_a65 = -5103.0 / 18656.0

_a71 = 35.0 / 384.0
_a72 = 0.0
_a73 = 500.0 / 1113.0
_a74 = 125.0 / 192.0
_a75 = -2187.0 / 6784.0
_a76 = 11.0 / 84.0

_b1 = 35.0 / 384.0
_b2 = 0.0
_b3 = 500.0 / 1113.0
_b4 = 125.0 / 192.0
_b5 = -2187.0 / 6784.0
_b6 = 11.0 / 84.0
_b7 = 0.0

_bs1 = 5179.0 / 57600.0
_bs2 = 0.0
_bs3 = 7571.0 / 16695.0
_bs4 = 393.0 / 640.0
_bs5 = -92097.0 / 339200.0
_bs6 = 187.0 / 2100.0
_bs7 = 1.0 / 40.0

# Safe guards
_DEFAULT_RTOL = 1e-8
_DEFAULT_ATOL = 1e-8
_MAX_STEPS = 2000000

# Implement integrator with numba if available
if _try_numba:
    @njit(fastmath=True)
    def _rk45_brusselator(y0, t0, t1, A, B, rtol, atol):
        # small helper for the RHS
        def f(y):
            x = y[0]
            yv = y[1]
            dx = A + x * x * yv - (B + 1.0) * x
            dy = B * x - x * x * yv
            return np.array((dx, dy), dtype=np.float64)

        # handle trivial case
        if t1 == t0:
            return y0.copy()

        t = t0
        y = y0.copy()
        dt = t1 - t0
        h = abs(dt) * 1e-3 + 1e-6
        if h == 0.0:
            h = 1e-6
        if dt < 0:
            h = -h

        safety = 0.9
        min_fac = 0.2
        max_fac = 10.0

        k1 = np.zeros(2, dtype=np.float64)
        k2 = np.zeros(2, dtype=np.float64)
        k3 = np.zeros(2, dtype=np.float64)
        k4 = np.zeros(2, dtype=np.float64)
        k5 = np.zeros(2, dtype=np.float64)
        k6 = np.zeros(2, dtype=np.float64)
        k7 = np.zeros(2, dtype=np.float64)
        ytmp = np.zeros(2, dtype=np.float64)
        y5 = np.zeros(2, dtype=np.float64)
        y4 = np.zeros(2, dtype=np.float64)
        e = np.zeros(2, dtype=np.float64)

        steps = 0
        while True:
            if (dt > 0 and t >= t1) or (dt < 0 and t <= t1):
                break
            if steps > _MAX_STEPS:
                break
            steps += 1

            # Adjust step to not overstep
            if dt > 0:
                if t + h > t1:
                    h = t1 - t
            else:
                if t + h < t1:
                    h = t1 - t

            # k1
            k1 = f(y)

            # k2
            ytmp[0] = y[0] + h * (_a21 * k1[0])
            ytmp[1] = y[1] + h * (_a21 * k1[1])
            k2 = f(ytmp)

            # k3
            ytmp[0] = y[0] + h * (_a31 * k1[0] + _a32 * k2[0])
            ytmp[1] = y[1] + h * (_a31 * k1[1] + _a32 * k2[1])
            k3 = f(ytmp)

            # k4
            ytmp[0] = y[0] + h * (_a41 * k1[0] + _a42 * k2[0] + _a43 * k3[0])
            ytmp[1] = y[1] + h * (_a41 * k1[1] + _a42 * k2[1] + _a43 * k3[1])
            k4 = f(ytmp)

            # k5
            ytmp[0] = y[0] + h * (_a51 * k1[0] + _a52 * k2[0] + _a53 * k3[0] + _a54 * k4[0])
            ytmp[1] = y[1] + h * (_a51 * k1[1] + _a52 * k2[1] + _a53 * k3[1] + _a54 * k4[1])
            k5 = f(ytmp)

            # k6
            ytmp[0] = y[0] + h * (_a61 * k1[0] + _a62 * k2[0] + _a63 * k3[0] + _a64 * k4[0] + _a65 * k5[0])
            ytmp[1] = y[1] + h * (_a61 * k1[1] + _a62 * k2[1] + _a63 * k3[1] + _a64 * k4[1] + _a65 * k5[1])
            k6 = f(ytmp)

            # 5th order solution (y5)
            y5[0] = y[0] + h * (_b1 * k1[0] + _b2 * k2[0] + _b3 * k3[0] + _b4 * k4[0] + _b5 * k5[0] + _b6 * k6[0] + _b7 * 0.0)
            y5[1] = y[1] + h * (_b1 * k1[1] + _b2 * k2[1] + _b3 * k3[1] + _b4 * k4[1] + _b5 * k5[1] + _b6 * k6[1] + _b7 * 0.0)

            # k7 (for some formulations; here k7 uses a7 coefficients)
            ytmp[0] = y[0] + h * (_a71 * k1[0] + _a72 * k2[0] + _a73 * k3[0] + _a74 * k4[0] + _a75 * k5[0] + _a76 * k6[0])
            ytmp[1] = y[1] + h * (_a71 * k1[1] + _a72 * k2[1] + _a73 * k3[1] + _a74 * k4[1] + _a75 * k5[1] + _a76 * k6[1])
            k7 = f(ytmp)

            # 4th order solution (y4) using b* coefficients
            y4[0] = y[0] + h * (_bs1 * k1[0] + _bs2 * k2[0] + _bs3 * k3[0] + _bs4 * k4[0] + _bs5 * k5[0] + _bs6 * k6[0] + _bs7 * k7[0])
            y4[1] = y[1] + h * (_bs1 * k1[1] + _bs2 * k2[1] + _bs3 * k3[1] + _bs4 * k4[1] + _bs5 * k5[1] + _bs6 * k6[1] + _bs7 * k7[1])

            # error estimate
            e[0] = y5[0] - y4[0]
            e[1] = y5[1] - y4[1]

            # scaled error
            sc0 = atol + rtol * max(abs(y[0]), abs(y5[0]))
            sc1 = atol + rtol * max(abs(y[1]), abs(y5[1]))
            err_norm = np.sqrt(0.5 * ((e[0] / sc0) ** 2 + (e[1] / sc1) ** 2))

            if err_norm <= 1.0:
                # accept
                t = t + h
                y[0] = y5[0]
                y[1] = y5[1]

                # adjust step
                if err_norm == 0.0:
                    h = h * max_fac
                else:
                    fac = safety * (err_norm ** (-0.2))
                    if fac < min_fac:
                        fac = min_fac
                    elif fac > max_fac:
                        fac = max_fac
                    h = h * fac
            else:
                # reject and reduce step
                fac = safety * (err_norm ** (-0.2))
                if fac < min_fac:
                    fac = min_fac
                elif fac > max_fac:
                    fac = max_fac
                h = h * fac

            # prevent underflow of step
            if abs(h) < 1e-16:
                # step too small, break to avoid infinite loop
                break

        return y

    # attempt to precompile with a trivial call to reduce first-call overhead in solve()
    try:
        # call once with tiny interval to trigger compilation (ignore any exception)
        _ = _rk45_brusselator(np.array((1.0, 1.0), dtype=np.float64), 0.0, 1e-6, 1.0, 3.0, _DEFAULT_RTOL, _DEFAULT_ATOL)
    except Exception:
        pass

else:
    _rk45_brusselator = None
    # Fallback using scipy
    def _scipy_solve(problem):
        from scipy.integrate import solve_ivp

        y0 = np.array(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem.get("params", {})
        A = float(params.get("A", 1.0))
        B = float(params.get("B", 3.0))

        def brusselator(t, y):
            X, Y = y
            dX_dt = A + X * X * Y - (B + 1.0) * X
            dY_dt = B * X - X * X * Y
            return np.array((dX_dt, dY_dt), dtype=np.float64)

        sol = solve_ivp(brusselator, (t0, t1), y0, method="RK45", rtol=_DEFAULT_RTOL, atol=_DEFAULT_ATOL)
        if not sol.success:
            raise RuntimeError("scipy solver failed: " + getattr(sol, "message", ""))
        return sol.y[:, -1]

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Solve the Brusselator ODE and return the final state [X, Y] at t1 as a numpy array.
        Uses a fast RK45 implementation compiled with numba when available; otherwise falls back to scipy.
        """
        # Validate and parse input
        try:
            y0 = np.array(problem["y0"], dtype=np.float64)
            t0 = float(problem["t0"])
            t1 = float(problem["t1"])
            params = problem.get("params", {})
            A = float(params.get("A", 1.0))
            B = float(params.get("B", 3.0))
        except Exception as e:
            raise ValueError(f"Invalid problem input: {e}")

        if y0.shape != (2,):
            # Try to shape it, but enforce 2 elements
            y0 = y0.reshape(-1)
            if y0.size != 2:
                raise ValueError("y0 must have length 2")

        if t1 == t0:
            return y0

        # Choose tolerances - we keep them tightly aligned with reference but allow kwargs to override
        rtol = float(kwargs.get("rtol", _DEFAULT_RTOL))
        atol = float(kwargs.get("atol", _DEFAULT_ATOL))

        if _rk45_brusselator is not None:
            # Use the fast compiled integrator
            try:
                y_final = _rk45_brusselator(y0, t0, t1, A, B, rtol, atol)
                return np.array(y_final, dtype=np.float64)
            except Exception:
                # fallback to scipy on unexpected errors
                pass

        # Fallback: use scipy's solve_ivp
        return _scipy_solve({"y0": y0, "t0": t0, "t1": t1, "params": {"A": A, "B": B}})