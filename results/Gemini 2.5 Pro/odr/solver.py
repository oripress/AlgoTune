from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Fits a linear model using Orthogonal Distance Regression (ODR)
        based on the iterative algorithm by York et al. (2004).
        This implementation uses a robust WLS-based initial guess for the slope,
        swaps axes for near-vertical lines to ensure numerical stability, and
        uses the corrected York iterative formula.
        """
        x_orig = np.asarray(problem["x"], dtype=np.float64)
        y_orig = np.asarray(problem["y"], dtype=np.float64)
        sx_orig = np.asarray(problem["sx"], dtype=np.float64).copy()
        sy_orig = np.asarray(problem["sy"], dtype=np.float64).copy()

        epsilon = np.finfo(np.float64).tiny
        sx_orig[sx_orig == 0] = epsilon
        sy_orig[sy_orig == 0] = epsilon

        # Use an OLS fit to decide if axes should be swapped for stability
        x_mean_ols = np.mean(x_orig)
        num_ols = np.sum((x_orig - x_mean_ols) * (y_orig - np.mean(y_orig)))
        den_ols = np.sum((x_orig - x_mean_ols)**2)
        m_ols = num_ols / den_ols if den_ols != 0 else np.inf

        if abs(m_ols) > 1.0:
            swapped = True
            x, y, sx, sy = y_orig, x_orig, sy_orig, sx_orig
        else:
            swapped = False
            x, y, sx, sy = x_orig, y_orig, sx_orig, sy_orig

        # Initial guess for m using Weighted Least Squares in the fitting frame.
        w_dep = 1.0 / sy**2
        try:
            if not np.any(np.isfinite(w_dep)) or np.all(w_dep <= 0):
                raise ValueError
            x_mean_wls = np.average(x, weights=w_dep)
            y_mean_wls = np.average(y, weights=w_dep)
            num = np.sum(w_dep * (x - x_mean_wls) * (y - y_mean_wls))
            den = np.sum(w_dep * (x - x_mean_wls)**2)
            m = num / den if den != 0 else 0.0
        except (ValueError, ZeroDivisionError): # Fallback to OLS
            x_mean_frame = np.mean(x)
            num = np.sum((x - x_mean_frame) * (y - np.mean(y)))
            den = np.sum((x - x_mean_frame)**2)
            m = num / den if den != 0 else 0.0

        # Iteratively refine the slope `m` using the York et al. algorithm
        rtol = np.finfo(float).eps**(2/3)
        omega_x = 1.0 / sx**2
        omega_y = 1.0 / sy**2

        for _ in range(50):
            m_old = m
            W = omega_x * omega_y / (m**2 * omega_y + omega_x)
            sum_W = np.sum(W)
            if sum_W <= 0: break
            x_bar = np.sum(W * x) / sum_W
            y_bar = np.sum(W * y) / sum_W
            U = x - x_bar
            V = y - y_bar
            
            # Corrected York iteration factor. This was the site of the bug.
            beta = W * (U / omega_y + m * V / omega_x)
            
            num_m = np.sum(W * beta * V)
            den_m = np.sum(W * beta * U)
            
            m = num_m / den_m if den_m != 0 else 0.0
            if abs(m - m_old) < rtol * (1 + abs(m_old)): break
        
        # Recalculate centroid and intercept with the final slope
        W = omega_x * omega_y / (m**2 * omega_y + omega_x)
        sum_W = np.sum(W)
        if sum_W > 0:
            x_bar = np.sum(W * x) / sum_W
            y_bar = np.sum(W * y) / sum_W
        else:
            x_bar, y_bar = np.mean(x), np.mean(y)
        c = y_bar - m * x_bar
        
        if swapped:
            with np.errstate(divide='ignore', invalid='ignore'):
                final_m = 1.0 / m
                final_c = -c / m
        else:
            final_m = m
            final_c = c
            
        return {"beta": [final_m, final_c]}