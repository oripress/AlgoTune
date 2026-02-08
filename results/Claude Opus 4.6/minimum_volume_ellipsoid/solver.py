import numpy as np
from typing import Any
import numba


@numba.njit(cache=True)
def khachiyan_core(B, d, tol, max_iter):
    """Core Khachiyan algorithm with Sherman-Morrison updates."""
    m = B.shape[0]
    dp1 = d + 1
    
    # Initialize uniform weights
    u = np.ones(m) / m
    
    # Compute initial M = B^T diag(u) B
    M = np.zeros((dp1, dp1))
    for i in range(m):
        for r in range(dp1):
            for c in range(dp1):
                M[r, c] += u[i] * B[i, r] * B[i, c]
    
    # Compute M^{-1}
    Minv = np.linalg.inv(M)
    
    # Compute g_i = b_i^T M^{-1} b_i
    g = np.zeros(m)
    for i in range(m):
        tmp = np.zeros(dp1)
        for r in range(dp1):
            s = 0.0
            for c in range(dp1):
                s += Minv[r, c] * B[i, c]
            tmp[r] = s
        s = 0.0
        for r in range(dp1):
            s += B[i, r] * tmp[r]
        g[i] = s
    
    for it in range(max_iter):
        # Find argmax
        j = 0
        g_max = g[0]
        for i in range(1, m):
            if g[i] > g_max:
                g_max = g[i]
                j = i
        
        # Convergence check
        if g_max <= dp1 * (1 + tol):
            break
        
        # Step size
        step = (g_max - dp1) / (dp1 * (g_max - 1))
        
        # Update weights
        for i in range(m):
            u[i] *= (1 - step)
        u[j] += step
        
        # Sherman-Morrison update
        w = np.zeros(dp1)
        for r in range(dp1):
            s = 0.0
            for c in range(dp1):
                s += Minv[r, c] * B[j, c]
            w[r] = s
        
        denom = (1 - step) + step * g_max
        f1 = 1.0 / (1 - step)
        f2 = step / denom
        
        # Update Minv
        for r in range(dp1):
            for c in range(dp1):
                Minv[r, c] = f1 * (Minv[r, c] - f2 * w[r] * w[c])
        
        # Compute q = B @ w
        q = np.zeros(m)
        for i in range(m):
            s = 0.0
            for r in range(dp1):
                s += B[i, r] * w[r]
            q[i] = s
        
        # Update g
        for i in range(m):
            g[i] = f1 * (g[i] - f2 * q[i] * q[i])
        
        # Periodic recomputation
        if (it + 1) % 300 == 0:
            M = np.zeros((dp1, dp1))
            for i in range(m):
                for r in range(dp1):
                    for c in range(dp1):
                        M[r, c] += u[i] * B[i, r] * B[i, c]
            Minv = np.linalg.inv(M)
            for i in range(m):
                tmp = np.zeros(dp1)
                for r in range(dp1):
                    s = 0.0
                    for c in range(dp1):
                        s += Minv[r, c] * B[i, c]
                    tmp[r] = s
                s = 0.0
                for r in range(dp1):
                    s += B[i, r] * tmp[r]
                g[i] = s
    
    return u


class Solver:
    def __init__(self):
        # Warm up numba
        B_dummy = np.array([[0.0, 1.0], [1.0, 1.0], [0.5, 1.0]])
        khachiyan_core(B_dummy, 1, 1e-6, 10)
    
    def solve(self, problem, **kwargs) -> Any:
        points = np.array(problem["points"], dtype=np.float64)
        n, d = points.shape
        
        # Use convex hull to reduce point set
        pts = points
        if d <= 10 and n > 2 * (d + 1):
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(points)
                pts = points[hull.vertices]
            except Exception:
                pass
        
        m = len(pts)
        
        # Lifted points b_i = [a_i, 1]
        B = np.empty((m, d + 1), dtype=np.float64)
        B[:, :d] = pts
        B[:, d] = 1.0
        
        tol = 1e-8
        max_iter = max(5000, 100 * d)
        
        # Run Khachiyan algorithm
        u = khachiyan_core(B, d, tol, max_iter)
        
        # Recover ellipsoid parameters
        c = pts.T @ u  # center
        diff = pts - c
        S = (diff * u[:, np.newaxis]).T @ diff
        
        # X = (S^{-1} / d)^{1/2}
        eigvals, eigvecs = np.linalg.eigh(S)
        eigvals = np.maximum(eigvals, 1e-20)
        x_eigvals = 1.0 / np.sqrt(d * eigvals)
        X = (eigvecs * x_eigvals) @ eigvecs.T
        X = (X + X.T) / 2
        Y = -X @ c
        
        objective_value = float(-np.log(np.linalg.det(X)))
        
        return {
            "objective_value": objective_value,
            "ellipsoid": {"X": X, "Y": Y}
        }