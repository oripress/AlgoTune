from typing import Any, Dict, List
import math

# Suzuki-Yoshida coefficients for 4th-order composition
_A1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_A2 = -2.0 ** (1.0 / 3.0) / (2.0 - 2.0 ** (1.0 / 3.0))

def _safe_exp(x: float) -> float:
    # Prevent overflow in exp
    if x > 700.0:
        x = 700.0
    elif x < -700.0:
        x = -700.0
    return math.exp(x)

def _flow_A(x0: float, y0: float, h: float, alpha: float, gamma: float):
    # Linear part: dx/dt = alpha*x, dy/dt = -gamma*y
    ex = _safe_exp(alpha * h)
    ey = _safe_exp(-gamma * h)
    return x0 * ex, y0 * ey

def _flow_B(x0: float, y0: float, h: float, beta: float, delta: float):
    # Interaction part: dx/dt = -beta*x*y, dy/dt = delta*x*y
    # This flow conserves K = y + (delta/beta)*x when beta!=0
    # Handle degenerate and trivial cases efficiently.
    if x0 == 0.0 or y0 == 0.0:
        # With either population zero, the pure-interaction flow does nothing.
        return x0, y0

    if beta == 0.0:
        # dx/dt = 0, dy/dt = delta * x0 * y => exponential growth/decay for y
        return x0, y0 * _safe_exp(delta * x0 * h)
    if delta == 0.0:
        # dy/dt = 0, dx/dt = -beta * x * y0 => exponential for x
        return x0 * _safe_exp(-beta * y0 * h), y0

    c = delta / beta
    K = y0 + c * x0
    if K == 0.0:
        return 0.0, 0.0

    r = beta * K
    exp_neg = _safe_exp(-r * h)
    # Avoid division by zero; denom should be positive in normal cases
    denom = 1.0 + (K / y0 - 1.0) * exp_neg
    if denom == 0.0:
        y1 = K
    else:
        y1 = K / denom
    x1 = (beta / delta) * (K - y1)

    # Clamp tiny negatives due to roundoff
    if x1 < 0.0 and abs(x1) < 1e-16:
        x1 = 0.0
    if y1 < 0.0 and abs(y1) < 1e-16:
        y1 = 0.0

    return x1, y1

def _s2(x: float, y: float, h: float, alpha: float, beta: float, delta: float, gamma: float):
    # Strang splitting: A(h/2) B(h) A(h/2)
    xa, ya = _flow_A(x, y, 0.5 * h, alpha, gamma)
    xb, yb = _flow_B(xa, ya, h, beta, delta)
    xf, yf = _flow_A(xb, yb, 0.5 * h, alpha, gamma)
    return xf, yf

def _s4(x: float, y: float, h: float, alpha: float, beta: float, delta: float, gamma: float):
    # 4th-order composition via Suzuki-Yoshida
    x1, y1 = _s2(x, y, _A1 * h, alpha, beta, delta, gamma)
    x2, y2 = _s2(x1, y1, _A2 * h, alpha, beta, delta, gamma)
    x3, y3 = _s2(x2, y2, _A1 * h, alpha, beta, delta, gamma)
    return x3, y3

def _integrate_s4_adaptive(t0: float, t1: float, x0: float, y0: float,
                           alpha: float, beta: float, delta: float, gamma: float,
                           rtol: float, atol: float):
    """
    Adaptive integrator using S4 splitting with step-doubling error estimate.
    """
    t = t0
    x = x0
    y = y0
    if t1 == t0:
        return x, y

    direction = 1.0 if t1 > t0 else -1.0
    total_dt = t1 - t0
    # initial guess for step size
    h = total_dt / 200.0
    if h == 0.0:
        h = direction * 1e-6
    if h * direction <= 0.0:
        h = direction * abs(h)

    safety = 0.9
    max_factor = 5.0
    min_factor = 0.2
    p = 4.0
    adapt_exp = 1.0 / (p + 1.0)
    tiny = 1e-300
    max_iters = 1000000
    iters = 0

    while (direction > 0.0 and t < t1 - 1e-16) or (direction < 0.0 and t > t1 + 1e-16):
        iters += 1
        if iters > max_iters:
            # Safety: return current approximation if too many iterations
            return x, y

        # adjust final step to hit t1 exactly
        if (direction > 0.0 and t + h > t1) or (direction < 0.0 and t + h < t1):
            h = t1 - t

        # one S4 step of size h
        xc, yc = _s4(x, y, h, alpha, beta, delta, gamma)
        # two S4 steps of size h/2
        xm, ym = _s4(x, y, 0.5 * h, alpha, beta, delta, gamma)
        xr, yr = _s4(xm, ym, 0.5 * h, alpha, beta, delta, gamma)

        err_x = abs(xr - xc)
        err_y = abs(yr - yc)

        axr = abs(xr)
        axc = abs(xc)
        max_x = axr if axr > axc else axc
        sc_x = atol + rtol * max_x
        if sc_x <= tiny:
            sc_x = tiny

        ayr = abs(yr)
        ayc = abs(yc)
        max_y = ayr if ayr > ayc else ayc
        sc_y = atol + rtol * max_y
        if sc_y <= tiny:
            sc_y = tiny

        err_norm = err_x / sc_x
        ey = err_y / sc_y
        if ey > err_norm:
            err_norm = ey

        # robustify
        if err_norm != err_norm or not math.isfinite(err_norm):
            err_norm = 1e6

        if err_norm <= 1.0:
            # accept step
            t = t + h
            x = xr
            y = yr
            # clamp tiny negatives
            if x < 0.0 and abs(x) < 1e-16:
                x = 0.0
            if y < 0.0 and abs(y) < 1e-16:
                y = 0.0

            # compute factor for next h
            if err_norm == 0.0:
                factor = max_factor
            else:
                factor = safety * (err_norm ** (-adapt_exp))
                if factor < min_factor:
                    factor = min_factor
                elif factor > max_factor:
                    factor = max_factor
            h = h * factor
        else:
            # reject step, shrink h
            factor = safety * (err_norm ** (-adapt_exp))
            if factor < min_factor:
                factor = min_factor
            elif factor > max_factor:
                factor = max_factor
            h = h * factor

        # avoid underflow of step size
        if abs(h) < 1e-24:
            h = 1e-24 * direction

    return x, y

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        """
        Solve Lotka-Volterra and return [x(t1), y(t1)].

        Optional kwargs:
          - rtol, atol: tolerances (defaults chosen to match reference accuracy)
        """
        if not isinstance(problem, dict):
            raise ValueError("problem must be a dict")

        try:
            y0_in = problem["y0"]
            if not hasattr(y0_in, "__len__") or len(y0_in) != 2:
                raise ValueError("y0 must be length-2")
            x0 = float(y0_in[0])
            y0 = float(y0_in[1])
            t0 = float(problem["t0"])
            t1 = float(problem["t1"])
            params = problem["params"]
            alpha = float(params["alpha"])
            beta = float(params["beta"])
            delta = float(params["delta"])
            gamma = float(params["gamma"])
        except Exception as e:
            raise ValueError(f"Invalid problem input: {e}")

        rtol = float(kwargs.get("rtol", 1e-10))
        atol = float(kwargs.get("atol", 1e-10))

        if t1 == t0:
            return [max(0.0, x0), max(0.0, y0)]

        # Use splitting integrator
        try:
            xf, yf = _integrate_s4_adaptive(t0, t1, x0, y0, alpha, beta, delta, gamma, rtol, atol)
        except Exception:
            xf, yf = x0, y0

        # final clamping and checks
        if xf < 0.0 and abs(xf) < 1e-14:
            xf = 0.0
        if yf < 0.0 and abs(yf) < 1e-14:
            yf = 0.0

        if not (math.isfinite(xf) and math.isfinite(yf)):
            # Fallback to SciPy if something went wrong
            try:
                from scipy.integrate import solve_ivp  # type: ignore

                def _f(t, z):
                    xx, yy = z
                    return [alpha * xx - beta * xx * yy, delta * xx * yy - gamma * yy]

                sol = solve_ivp(_f, (t0, t1), [x0, y0], method="RK45", rtol=1e-10, atol=1e-10)
                if not sol.success:
                    raise RuntimeError("fallback solver failed")
                arr = sol.y[:, -1]
                xf, yf = float(arr[0]), float(arr[1])
            except Exception as ee:
                raise RuntimeError(f"Integration failed: {ee}")

        return [float(xf), float(yf)]