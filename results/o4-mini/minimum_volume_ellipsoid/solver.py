import os
threads = os.cpu_count() or 1
threads = str(threads)
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["OMP_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
import numpy as np
import numba
from numba import njit

@njit(fastmath=True)
def compute_u(P, tol, max_iter):
    n, d = P.shape
    d1 = d + 1
    # Build augmented matrix Q
    Q = np.empty((n, d1), dtype=np.float64)
    for i in range(n):
        for j in range(d):
            Q[i, j] = P[i, j]
        Q[i, d] = 1.0
    # Initialize weights
    inv_n = 1.0 / n
    u = np.full(n, inv_n, dtype=np.float64)
    # Initial scatter matrix
    X_scatter = np.zeros((d1, d1), dtype=np.float64)
    for i2 in range(d1):
        for j2 in range(d1):
            s = 0.0
            for k in range(n):
                s += Q[k, i2] * u[k] * Q[k, j2]
            X_scatter[i2, j2] = s
    invX = np.linalg.inv(X_scatter)
    # Khachiyan iterations
    for _ in range(max_iter):
        M = np.empty(n, dtype=np.float64)
        for k in range(n):
            s1 = 0.0
            for i2 in range(d1):
                t = 0.0
                for j2 in range(d1):
                    t += invX[i2, j2] * Q[k, j2]
                s1 += t * Q[k, i2]
            M[k] = s1
        max_M = M[0]
        j = 0
        for k in range(1, n):
            if M[k] > max_M:
                max_M = M[k]
                j = k
        if max_M <= d1 * (1.0 + tol):
            break
        step = (max_M - d1) / (d1 * (max_M - 1.0))
        alpha = 1.0 - step
        beta = step
        # Compute z = invX @ Q[j]
        z = np.empty(d1, dtype=np.float64)
        for i2 in range(d1):
            s2 = 0.0
            for k2 in range(d1):
                s2 += invX[i2, k2] * Q[j, k2]
            z[i2] = s2
        vvz = 0.0
        for i2 in range(d1):
            vvz += Q[j, i2] * z[i2]
        denom = alpha + beta * vvz
        coef = beta / denom
        # Rank-1 update of invX
        for i2 in range(d1):
            for j2 in range(d1):
                invX[i2, j2] = (invX[i2, j2] - coef * z[i2] * z[j2]) / alpha
        # Update weights
        for k2 in range(n):
            u[k2] *= alpha
        u[j] += beta
    return u
class Solver:
    def solve(self, problem, **kwargs):
        # Read and prepare data
        P = np.array(problem["points"], dtype=np.float64)
        n, d = P.shape
        # Filter to convex hull points for 2D to speed up
        if d == 2 and n > 200:
            pts_list = P.tolist()
            # monotone chain convex hull
            def cross(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            pts_list.sort()
            lower = []
            for p in pts_list:
                while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                    lower.pop()
                lower.append(p)
            upper = []
            for p in reversed(pts_list):
                while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                    upper.pop()
                upper.append(p)
            hull = lower[:-1] + upper[:-1]
            P = np.array(hull, dtype=np.float64)
            n = P.shape[0]
        if n == 0:
            return {
                "objective_value": float("inf"),
                "ellipsoid": {
                    "X": np.nan * np.ones((d, d)),
                    "Y": np.nan * np.ones((d,)),
                },
            }
        # Augment points with homogeneous coordinate
        Q = np.hstack((P, np.ones((n, 1))))
        d1 = d + 1
        # Initialize weights uniformly
        # Initialize weights uniformly
        u = np.ones(n) / n
        tol = 1e-6
        max_iter = 1000
        # Initial scatter matrix and its inverse
        X_scatter = Q.T @ (Q * u[:, None])
        try:
            invX = np.linalg.inv(X_scatter)
        except np.linalg.LinAlgError:
            invX = np.linalg.pinv(X_scatter)
        # Khachiyan iterations using rank-1 invX updates
        for _ in range(max_iter):
            # Compute Mahalanobis distances for all points
            M = np.sum((Q @ invX) * Q, axis=1)
            max_M = M.max()
            j = np.argmax(M)
            # Check convergence
            if max_M <= d1 * (1 + tol):
                break
            # Compute update step parameters
            step = (max_M - d1) / (d1 * (max_M - 1))
            alpha = 1 - step
            beta = step
            v = Q[j]
            # Rank-1 update of invX via Sherman–Morrison
            z = invX @ v
            denom = alpha + beta * (v @ z)
            invX = (invX - (beta / denom) * np.outer(z, z)) / alpha
            # Update weights
            u *= alpha
            u[j] += beta
        # end iterations
        # end iterations
        # Compute ellipsoid center
        c = u @ P
        # Compute covariance-like matrix for shape
        P_centered = P - c
        M2 = P_centered.T @ (P_centered * u[:, None])
        # Invert (or pseudo-invert) and normalize
        try:
            A = np.linalg.inv(M2) / d
        except np.linalg.LinAlgError:
            A = np.linalg.pinv(M2) / d
        # Matrix square root: X = sqrt(A)
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.clip(eigvals, 0, None)
        sqrt_vals = np.sqrt(eigvals)
        X_mat = (eigvecs * sqrt_vals) @ eigvecs.T
        # Translation vector
        Y_vec = -X_mat @ c
        # scale to ensure feasibility
        d2 = np.sum((P_centered @ A) * P_centered, axis=1)
        max_d2 = d2.max()
        if max_d2 > 1.0:
            factor = np.sqrt(max_d2)
            X_mat /= factor
            Y_vec /= factor
        # Objective: -log det(X)
        sign, logdet = np.linalg.slogdet(X_mat)
        obj = -logdet
        return {
            "objective_value": obj,
            "ellipsoid": {"X": X_mat.tolist(), "Y": Y_vec.tolist()},
        }